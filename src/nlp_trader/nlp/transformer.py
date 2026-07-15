from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from nlp_trader.nlp.simple import normalize_text
from nlp_trader.utils.device import get_torch_device


@dataclass(frozen=True, slots=True)
class TransformerSentimentConfig:
    model_name: str
    model_version: str
    cache_dir: Path
    batch_size: int = 32
    max_sequence_length: int = 256
    local_files_only: bool = True


@dataclass(frozen=True, slots=True)
class TransformerSentimentResult:
    score: float
    label: str
    confidence: float

    def __post_init__(self) -> None:
        if self.label not in {"positive", "negative", "neutral"}:
            raise ValueError(f"unsupported transformer sentiment label: {self.label}")
        if not math.isfinite(self.score) or not -1.0 <= self.score <= 1.0:
            raise ValueError("transformer sentiment score must be finite and between -1 and 1")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("transformer confidence must be finite and between 0 and 1")


Predictor = Callable[[list[str]], list[TransformerSentimentResult]]


def _cache_key(text: str, config: TransformerSentimentConfig) -> str:
    payload = (
        f"{config.model_name}\0{config.model_version}\0"
        f"max_length={config.max_sequence_length}\0{normalize_text(text)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedTransformerSentiment:
    """Batched, disk-cached optional transformer inference.

    Model loading is explicitly local-only by default. Tests can inject a small
    predictor and therefore never download a model or require PyTorch.
    """

    def __init__(
        self,
        config: TransformerSentimentConfig,
        *,
        predictor: Predictor | None = None,
    ) -> None:
        if config.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if config.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        self.config = config
        self._predictor = predictor

    def _read(self, key: str) -> TransformerSentimentResult | None:
        path = self.config.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return TransformerSentimentResult(
            score=float(raw["score"]),
            label=str(raw["label"]),
            confidence=float(raw["confidence"]),
        )

    def _write(self, key: str, result: TransformerSentimentResult) -> None:
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.cache_dir / f"{key}.json"
        if path.exists():
            return
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "score": result.score,
                    "label": result.label,
                    "confidence": result.confidence,
                    "model_name": self.config.model_name,
                    "model_version": self.config.model_version,
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")

    def _default_predictor(self) -> Predictor:
        try:
            pipeline = import_module("transformers").pipeline
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Transformer sentiment is optional; run `uv sync --extra nlp` first"
            ) from exc
        device = get_torch_device()
        device_index = "mps" if str(device) == "mps" else -1
        classifier: Any = pipeline(
            "text-classification",
            model=self.config.model_name,
            tokenizer=self.config.model_name,
            device=device_index,
            truncation=True,
            max_length=self.config.max_sequence_length,
            local_files_only=self.config.local_files_only,
        )

        def predict(batch: list[str]) -> list[TransformerSentimentResult]:
            outputs = classifier(batch, batch_size=self.config.batch_size)
            results: list[TransformerSentimentResult] = []
            for output in outputs:
                raw_label = str(output["label"]).lower()
                confidence = float(output["score"])
                label = (
                    "negative"
                    if "neg" in raw_label
                    else "positive"
                    if "pos" in raw_label
                    else "neutral"
                )
                direction = -1.0 if label == "negative" else 1.0 if label == "positive" else 0.0
                results.append(
                    TransformerSentimentResult(
                        score=direction * confidence,
                        label=label,
                        confidence=confidence,
                    )
                )
            return results

        return predict

    def predict(self, texts: Iterable[str]) -> list[TransformerSentimentResult]:
        values = list(texts)
        results: list[TransformerSentimentResult | None] = [None] * len(values)
        missing: list[tuple[int, str, str]] = []
        for index, text in enumerate(values):
            key = _cache_key(text, self.config)
            cached = self._read(key)
            if cached is None:
                missing.append((index, key, text))
            else:
                results[index] = cached
        if missing:
            predictor = self._predictor or self._default_predictor()
            for start in range(0, len(missing), self.config.batch_size):
                chunk = missing[start : start + self.config.batch_size]
                predicted = predictor([text for _, _, text in chunk])
                if len(predicted) != len(chunk):
                    raise ValueError("transformer predictor returned the wrong number of rows")
                for (index, key, _), result in zip(chunk, predicted, strict=True):
                    self._write(key, result)
                    results[index] = result
        if any(result is None for result in results):
            raise RuntimeError("transformer inference did not produce every requested result")
        return [result for result in results if result is not None]


def evaluate_golden_set(
    engine: CachedTransformerSentiment,
    examples: Iterable[tuple[str, str]],
) -> dict[str, float | int]:
    """Evaluate a configured local model against a tiny labeled golden set."""

    rows = list(examples)
    expected = [label for _, label in rows]
    if any(label not in {"positive", "negative", "neutral"} for label in expected):
        raise ValueError("golden-set labels must be positive, negative, or neutral")
    predicted = engine.predict(text for text, _ in rows)
    correct = sum(result.label == label for result, label in zip(predicted, expected, strict=True))
    return {
        "examples": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
    }
