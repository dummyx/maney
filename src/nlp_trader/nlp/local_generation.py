from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from nlp_trader.utils.device import select_llama_cpp_device


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    request_id: str
    prompt: str

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.prompt.strip():
            raise ValueError("generation request_id and prompt must not be empty")


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    request_id: str
    generated_text: str | None = None
    input_too_long: bool = False
    output_truncated: bool = False
    input_token_count: int | None = None
    output_token_count: int | None = None
    generation_latency_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("generation response request_id must not be empty")
        if self.generated_text is not None and not isinstance(self.generated_text, str):
            raise ValueError("generation response text must be a string when supplied")
        if type(self.input_too_long) is not bool or type(self.output_truncated) is not bool:
            raise ValueError("generation response state flags must be booleans")
        if self.input_too_long:
            if self.generated_text is not None or self.output_truncated:
                raise ValueError("input-too-long responses cannot contain generated output")
        elif self.generated_text is None or not self.generated_text.strip():
            raise ValueError("generation response text must not be empty")
        for field_name in ("input_token_count", "output_token_count"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer")
        latency = self.generation_latency_seconds
        if latency is not None and (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency < 0.0
        ):
            raise ValueError("generation_latency_seconds must be finite and non-negative")


RawGenerator = Callable[[list[GenerationRequest]], list[GenerationResponse]]


@dataclass(frozen=True, slots=True)
class LocalGenerationConfig:
    """Validated settings; model bytes are verified at the load boundary."""

    model_path: Path
    expected_model_sha256: str
    context_tokens: int
    prompt_batch_tokens: int
    max_input_tokens: int
    max_new_tokens: int
    gpu_layers: int = -1
    flash_attention: bool = True
    use_mmap: bool = True
    seed: int = 7
    decoding: Literal["greedy"] = "greedy"

    def __post_init__(self) -> None:
        if not self.model_path.is_file() or self.model_path.suffix.lower() != ".gguf":
            raise ValueError(f"local model must be one existing .gguf file: {self.model_path}")
        if re.fullmatch(r"[0-9a-f]{64}", self.expected_model_sha256) is None:
            raise ValueError("expected_model_sha256 must be a lowercase SHA-256 digest")
        if self.max_input_tokens < 1 or self.max_new_tokens < 1:
            raise ValueError("generation token limits must be positive")
        if self.context_tokens < self.max_input_tokens + self.max_new_tokens:
            raise ValueError("context_tokens must cover max_input_tokens + max_new_tokens")
        if not 1 <= self.prompt_batch_tokens <= self.context_tokens:
            raise ValueError("prompt_batch_tokens must be within the configured context")
        if self.gpu_layers < -1:
            raise ValueError("gpu_layers must be -1, 0, or a positive layer count")
        if self.seed < 1:
            raise ValueError("generation seed must be positive")
        if self.decoding != "greedy":
            raise ValueError("only deterministic greedy generation is supported")


@dataclass(frozen=True, slots=True)
class LocalGenerationDiagnostics:
    backend: Literal["llama_cpp_gguf"]
    runtime_version: str
    device: Literal["metal", "cpu"]
    requested_gpu_layers: int
    effective_gpu_layers: int
    chat_template_sha256: str
    model_load_latency_seconds: float


class LlamaCppGenerationSession:
    """One local, structured llama.cpp session with no cache or network behavior."""

    def __init__(
        self,
        config: LocalGenerationConfig,
        response_schema: dict[str, Any],
        *,
        module_loader: Callable[[str], Any] = import_module,
        runtime_version: str | None = None,
    ) -> None:
        self.config = config
        self.response_schema = response_schema
        try:
            llama_cpp = module_loader("llama_cpp")
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Local GGUF generation requires llama-cpp-python; run `uv sync --extra llm` first"
            ) from exc

        if runtime_version is None:
            try:
                runtime_version = version("llama-cpp-python")
            except PackageNotFoundError:  # pragma: no cover - malformed optional install
                runtime_version = "not-installed"
        if file_sha256(config.model_path) != config.expected_model_sha256:
            raise ValueError("local model changed before model loading")
        placement = select_llama_cpp_device(llama_cpp, config.gpu_layers)
        llama_type = getattr(llama_cpp, "Llama", None)
        if llama_type is None:
            raise RuntimeError("installed llama-cpp-python does not expose llama_cpp.Llama")
        load_options = dict(
            model_path=str(config.model_path),
            n_ctx=config.context_tokens,
            n_batch=config.prompt_batch_tokens,
            n_gpu_layers=placement.gpu_layers,
            seed=config.seed,
            flash_attn=config.flash_attention,
            use_mmap=config.use_mmap,
            offload_kqv=placement.name == "metal",
            op_offload=placement.name == "metal",
            chat_format="chat_template.default",
            verbose=False,
        )
        started_at = time.perf_counter()
        try:
            model: Any = llama_type(**load_options)
        except ValueError as exc:
            if placement.name == "metal" and "Failed to create llama_context" in str(exc):
                raise RuntimeError(
                    "llama.cpp could not create a Metal context; check available unified "
                    "memory and context settings, or reinstall llama-cpp-python with "
                    "GGML_METAL=OFF for a CPU-only runtime"
                ) from exc
            raise
        model_load_latency = time.perf_counter() - started_at
        if file_sha256(config.model_path) != config.expected_model_sha256:
            raise ValueError("local model changed while model files were loading")
        metadata = getattr(model, "metadata", None)
        if not isinstance(metadata, dict):
            raise ValueError("local GGUF model does not expose readable metadata")
        chat_template = metadata.get("tokenizer.chat_template") or metadata.get(
            "tokenizer.chat_template.default"
        )
        if not isinstance(chat_template, str) or not chat_template.strip():
            raise ValueError("local GGUF model must contain an embedded chat template")
        template_hash = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        chat_format_module = getattr(llama_cpp, "llama_chat_format", None)
        formatter_type = getattr(chat_format_module, "Jinja2ChatFormatter", None)
        if formatter_type is None:
            raise RuntimeError("installed llama-cpp-python cannot render GGUF chat templates")
        eos_token_id = model.token_eos()
        bos_token_id = model.token_bos()
        eos_token = model.detokenize([eos_token_id], special=True).decode("utf-8")
        bos_token = model.detokenize([bos_token_id], special=True).decode("utf-8")
        self._model = model
        self._chat_formatter = formatter_type(
            template=chat_template,
            eos_token=eos_token,
            bos_token=bos_token,
            stop_token_ids=[eos_token_id],
        )
        self.diagnostics = LocalGenerationDiagnostics(
            backend="llama_cpp_gguf",
            runtime_version=runtime_version,
            device=placement.name,
            requested_gpu_layers=config.gpu_layers,
            effective_gpu_layers=placement.gpu_layers,
            chat_template_sha256=template_hash,
            model_load_latency_seconds=model_load_latency,
        )

    def generate(self, batch: list[GenerationRequest]) -> list[GenerationResponse]:
        results: list[GenerationResponse] = []
        for request in batch:
            messages = [{"role": "user", "content": request.prompt}]
            formatted = self._chat_formatter(messages=messages)
            formatted_prompt = getattr(formatted, "prompt", None)
            added_special = getattr(formatted, "added_special", None)
            if not isinstance(formatted_prompt, str) or type(added_special) is not bool:
                raise ValueError("llama.cpp chat formatter returned an invalid prompt")
            token_ids = self._model.tokenize(
                formatted_prompt.encode("utf-8"),
                add_bos=not added_special,
                special=True,
            )
            if not isinstance(token_ids, list) or any(
                type(value) is not int for value in token_ids
            ):
                raise ValueError("llama.cpp tokenizer returned an invalid token sequence")
            input_tokens = len(token_ids)
            if (
                input_tokens > self.config.max_input_tokens
                or input_tokens + self.config.max_new_tokens > self.config.context_tokens
            ):
                results.append(
                    GenerationResponse(
                        request_id=request.request_id,
                        input_too_long=True,
                        input_token_count=input_tokens,
                        output_token_count=0,
                    )
                )
                continue
            started_at = time.perf_counter()
            try:
                raw = self._model.create_chat_completion(
                    messages=messages,
                    response_format={"type": "json_object", "schema": self.response_schema},
                    temperature=0.0,
                    top_p=1.0,
                    top_k=1,
                    min_p=0.0,
                    typical_p=1.0,
                    repeat_penalty=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    seed=self.config.seed,
                    max_tokens=self.config.max_new_tokens,
                    stream=False,
                )
            except ValueError as exc:
                if (
                    re.fullmatch(r"Requested tokens \(\d+\) exceed context window of \d+", str(exc))
                    is None
                ):
                    raise
                results.append(
                    GenerationResponse(
                        request_id=request.request_id,
                        input_too_long=True,
                        input_token_count=input_tokens,
                        output_token_count=0,
                        generation_latency_seconds=time.perf_counter() - started_at,
                    )
                )
                continue
            latency = time.perf_counter() - started_at
            if not isinstance(raw, dict):
                raise ValueError("llama.cpp chat completion returned a non-object response")
            choices = raw.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("llama.cpp chat completion must return exactly one choice")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise ValueError("llama.cpp chat completion choice is invalid")
            message = choice.get("message")
            if not isinstance(message, dict):
                raise ValueError("llama.cpp chat completion message is invalid")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("llama.cpp chat completion content is empty")
            finish_reason = choice.get("finish_reason")
            if finish_reason not in {"stop", "length"}:
                raise ValueError(
                    f"llama.cpp chat completion has unsupported finish_reason: {finish_reason!r}"
                )
            usage = raw.get("usage")
            if not isinstance(usage, dict):
                raise ValueError("llama.cpp chat completion did not return token usage")
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if (
                type(prompt_tokens) is not int
                or prompt_tokens < 0
                or type(completion_tokens) is not int
                or completion_tokens < 0
            ):
                raise ValueError("llama.cpp chat completion token usage is invalid")
            if prompt_tokens != input_tokens:
                raise ValueError(
                    "llama.cpp formatted-prompt token count changed between preflight "
                    "and generation"
                )
            results.append(
                GenerationResponse(
                    request_id=request.request_id,
                    generated_text=content,
                    output_truncated=finish_reason == "length",
                    input_token_count=prompt_tokens,
                    output_token_count=completion_tokens,
                    generation_latency_seconds=latency,
                )
            )
        return results
