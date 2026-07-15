from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlp_trader.nlp.transformer import (
    CachedTransformerSentiment,
    TransformerSentimentConfig,
    TransformerSentimentResult,
    evaluate_golden_set,
)


def test_transformer_results_are_cached_without_optional_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def predictor(texts: list[str]) -> list[TransformerSentimentResult]:
        calls.append(texts)
        return [TransformerSentimentResult(0.8, "positive", 0.8) for _ in texts]

    config = TransformerSentimentConfig(
        model_name="local-test-model",
        model_version="v1",
        cache_dir=tmp_path,
        batch_size=1,
    )
    first = CachedTransformerSentiment(config, predictor=predictor).predict(["Strong growth"])
    cached = CachedTransformerSentiment(config)
    monkeypatch.setattr(
        cached,
        "_default_predictor",
        lambda: (_ for _ in ()).throw(AssertionError("cache hit loaded model")),
    )
    second = cached.predict(["Strong growth"])

    assert first == second
    assert calls == [["Strong growth"]]

    different_length = TransformerSentimentConfig(
        model_name="local-test-model",
        model_version="v1",
        cache_dir=tmp_path,
        batch_size=1,
        max_sequence_length=128,
    )
    CachedTransformerSentiment(different_length, predictor=predictor).predict(["Strong growth"])
    assert calls == [["Strong growth"], ["Strong growth"]]


def test_transformer_golden_set_harness_uses_local_injected_predictions(
    tmp_path: Path,
) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "transformer_sentiment_golden.jsonl"
    examples = [
        (record["text"], record["label"])
        for record in (
            json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()
        )
    ]

    def predictor(texts: list[str]) -> list[TransformerSentimentResult]:
        results: list[TransformerSentimentResult] = []
        for text in texts:
            label = (
                "positive" if "improved" in text else "negative" if "losses" in text else "neutral"
            )
            score = 0.8 if label == "positive" else -0.8 if label == "negative" else 0.0
            results.append(TransformerSentimentResult(score, label, 0.8))
        return results

    engine = CachedTransformerSentiment(
        TransformerSentimentConfig(
            model_name="local-golden-model",
            model_version="golden-v1",
            cache_dir=tmp_path,
        ),
        predictor=predictor,
    )

    assert evaluate_golden_set(engine, examples) == {
        "examples": 3,
        "correct": 3,
        "accuracy": 1.0,
    }
