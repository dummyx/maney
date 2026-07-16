from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, field_validator, model_validator

from nlp_trader.nlp.local_generation import file_sha256
from nlp_trader.research_agents.contracts import (
    Identifier,
    NonBlankText,
    Sha256,
    StrictModel,
    content_sha256,
)


class ResearchAgentConfig(StrictModel):
    config_schema_version: Literal["research-agent-config-v1"] = "research-agent-config-v1"
    enabled: bool = False
    role: Literal["analyst"] = "analyst"
    backend: Literal["llama_cpp_gguf"] = "llama_cpp_gguf"
    model_path: Path | None = None
    model_logical_id: Identifier
    model_revision: Identifier
    model_expected_sha256: Sha256 | None = None
    model_license_or_terms_ref: NonBlankText
    prompt_version: Identifier
    action_schema_version: Identifier
    proposal_schema_version: Identifier
    tool_catalog_version: Identifier
    verifier_version: Identifier
    runtime_version: Identifier
    decoding: Literal["greedy"] = "greedy"
    seed: int = Field(default=7, ge=1)
    context_tokens: int = Field(default=8192, ge=512)
    prompt_batch_tokens: int = Field(default=512, ge=1)
    max_input_tokens: int = Field(default=6144, ge=1)
    max_output_tokens: int = Field(default=1024, ge=1)
    gpu_layers: int = Field(default=-1, ge=-1)
    flash_attention: bool = True
    use_mmap: bool = True
    max_steps: int = Field(default=8, ge=1, le=32)
    max_tool_calls: int = Field(default=7, ge=0, le=31)
    max_evidence_results: int = Field(default=40, ge=1, le=100)
    max_evidence_pages: int = Field(default=8, ge=1, le=20)
    max_metric_reads: int = Field(default=3, ge=0, le=20)
    max_tool_result_bytes_per_step: int = Field(default=65_536, ge=1)
    max_tool_result_bytes_per_run: int = Field(default=262_144, ge=1)
    max_wall_time_seconds: float = Field(default=600.0, gt=0.0, le=7_200.0)
    max_retained_artifact_bytes: int = Field(default=16_777_216, ge=1)
    artifact_root: Path
    environment_scrub_policy_version: Identifier
    feasibility_diagnostics: bool = True

    @field_validator("model_path", "artifact_root", mode="before")
    @classmethod
    def parse_path(cls, value: object) -> Path | None:
        if value is None:
            return None
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            return Path(value).expanduser()
        raise ValueError("research-agent paths must be strings, Path values, or null")

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if not self.artifact_root.is_absolute():
            raise ValueError("research-agent artifact_root must be absolute")
        if self.context_tokens < self.max_input_tokens + self.max_output_tokens:
            raise ValueError("context_tokens must cover max input and output tokens")
        if self.prompt_batch_tokens > self.context_tokens:
            raise ValueError("prompt_batch_tokens cannot exceed context_tokens")
        if self.max_tool_calls >= self.max_steps:
            raise ValueError("max_tool_calls must leave at least one terminal model step")
        if self.max_tool_result_bytes_per_step > self.max_tool_result_bytes_per_run:
            raise ValueError("per-step tool bytes cannot exceed the per-run bound")
        if self.enabled:
            if self.model_path is None:
                raise ValueError("enabled research agent requires model_path")
            if not self.model_path.is_absolute() or not self.model_path.is_file():
                raise ValueError("enabled research agent requires one existing absolute GGUF path")
            if self.model_path.suffix.casefold() != ".gguf":
                raise ValueError("enabled research agent model_path must be one GGUF file")
            if self.model_expected_sha256 is None:
                raise ValueError("enabled research agent requires model_expected_sha256")
            if file_sha256(self.model_path) != self.model_expected_sha256:
                raise ValueError("research-agent model bytes do not match configured SHA-256")
        return self

    def content_hash(self) -> str:
        return content_sha256(self.model_dump(mode="json"))


def load_research_agent_config(path: str | Path) -> ResearchAgentConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("research-agent configuration cannot be read") from exc
    if not isinstance(payload, dict):
        raise ValueError("research-agent configuration must contain one mapping")
    for field_name in ("artifact_root", "model_path"):
        value = payload.get(field_name)
        if isinstance(value, str):
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                payload[field_name] = str((config_path.parent / candidate).resolve())
    return ResearchAgentConfig.model_validate(payload)


def require_enabled(config: ResearchAgentConfig) -> None:
    if not config.enabled:
        raise ValueError("research-agent propose requires enabled: true")
