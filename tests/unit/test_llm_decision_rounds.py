from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from nlp_trader.nlp.llm_decision_rounds import (
    CurrentSourceRetrieval,
    DecisionRound,
    DecisionRoundLedger,
    DecisionRoundLedgerError,
    InferenceUsage,
    ModelIdentity,
    RawGeneration,
    SamplingSettings,
    VerifierCheck,
    VerifierResult,
    VersionedContract,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _decision_round(
    *,
    run_id: str = "run-001",
    item_id: str = "item-001",
    source_available_at: datetime | str = datetime(
        2026,
        7,
        15,
        21,
        0,
        tzinfo=timezone(timedelta(hours=9)),
    ),
    decision_time: datetime | str = datetime(2026, 7, 16, 20, 0, tzinfo=UTC),
    structured_output: dict[str, object] | None = None,
    inference_source: str = "generated",
) -> DecisionRound:
    raw_output = structured_output or {
        "annotations": [
            {
                "asset_id": "asset_aaa",
                "evidence_span_ids": ["S2"],
                "stance_label": "positive",
            }
        ]
    }
    return DecisionRound(
        run_id=run_id,
        config_hash="e" * 64,
        input_snapshot_hash="f" * 64,
        item_id=item_id,
        source_text_hash="a" * 64,
        source_available_at=source_available_at,
        decision_time=decision_time,
        horizon_days=5,
        model=ModelIdentity(
            provider="transformers_causal_lm",
            model_id="local/test-model",
            revision="immutable-revision-1",
            sha256="b" * 64,
        ),
        prompt=VersionedContract(version="semantic-evidence-v2", sha256="c" * 64),
        schema_contract=VersionedContract(version="semantic-signal-v2", sha256="d" * 64),
        sampling=SamplingSettings(
            decoding="greedy",
            seed=7,
            max_input_tokens=2048,
            max_new_tokens=384,
            temperature=None,
            top_p=None,
        ),
        retrieval=CurrentSourceRetrieval(evidence_ids=("S1", "S2")),
        raw_generation=RawGeneration(
            request_id="request-001",
            generated_text=_canonical_json(raw_output),
            metadata={
                "backend": "injected_generator",
                "input_too_long": False,
                "output_truncated": False,
            },
        ),
        structured_output={"item_id": item_id, **raw_output},  # type: ignore[arg-type]
        verifier=VerifierResult(
            version="semantic-evidence-verifier-v1",
            passed=True,
            checks=(
                VerifierCheck(check_id="schema", passed=True),
                VerifierCheck(check_id="evidence", passed=True),
                VerifierCheck(check_id="temporal", passed=True),
            ),
        ),
        inference_source=inference_source,  # type: ignore[arg-type]
        usage=(
            InferenceUsage(
                input_tokens=91,
                output_tokens=37,
                latency_ms=12.5,
                estimated_usd_cost=0.0004,
            )
            if inference_source == "generated"
            else InferenceUsage(
                input_tokens=0,
                output_tokens=0,
                latency_ms=0.0,
                estimated_usd_cost=0.0,
            )
        ),
        application_mode="sidecar",
    )


def test_round_id_is_deterministic_over_normalized_canonical_content() -> None:
    first = _decision_round()
    equivalent = _decision_round(
        source_available_at="2026-07-15T12:00:00Z",
        decision_time="2026-07-17T05:00:00+09:00",
        structured_output={
            "annotations": [
                {
                    "stance_label": "positive",
                    "evidence_span_ids": ["S2"],
                    "asset_id": "asset_aaa",
                }
            ]
        },
    )

    assert first.round_id == equivalent.round_id
    assert len(first.round_id) == 64
    assert first.round_id == first.computed_round_id()
    payload = first.model_dump(mode="json")
    assert payload["artifact_schema_version"] == "llm-decision-round-v1"
    assert payload["source_available_at"] == "2026-07-15T12:00:00Z"
    assert payload["decision_time"] == "2026-07-16T20:00:00Z"
    assert payload["tool_calls"] == []
    assert payload["calibration"] == {}
    assert payload["portfolio"] == {}
    assert payload["risk"] == {}
    assert payload["orders"] == []
    assert "outcome" not in payload


def test_exclusive_canonical_jsonl_round_trip_preserves_exact_trace(tmp_path: Path) -> None:
    path = tmp_path / "decision_rounds.jsonl"
    rounds = (
        _decision_round(),
        _decision_round(
            item_id="item-002",
            inference_source="deduplicated",
        ),
    )
    ledger = DecisionRoundLedger(path)

    written = ledger.write_exclusive(rounds)

    assert written == rounds
    assert ledger.replay_and_verify() == rounds
    assert path.read_text(encoding="utf-8") == "".join(
        value.canonical_json() + "\n" for value in rounds
    )
    assert ledger.replay()[1].raw_generation.generated_text == (
        rounds[1].raw_generation.generated_text
    )
    assert ledger.replay()[1].usage.estimated_usd_cost == 0.0

    with pytest.raises(DecisionRoundLedgerError, match="already exists"):
        ledger.write_exclusive(rounds)


def test_replay_rejects_tampering_and_duplicate_round_ids(tmp_path: Path) -> None:
    tampered_path = tmp_path / "tampered.jsonl"
    tampered_ledger = DecisionRoundLedger(tampered_path)
    decision_round = _decision_round()
    tampered_ledger.write_exclusive((decision_round,))
    record = json.loads(tampered_path.read_text(encoding="utf-8"))
    record["raw_generation"]["generated_text"] = '{"annotations":[]}'
    tampered_path.write_text(_canonical_json(record) + "\n", encoding="utf-8")

    with pytest.raises(DecisionRoundLedgerError, match="raw generation"):
        tampered_ledger.replay_and_verify()

    duplicate_path = tmp_path / "duplicate.jsonl"
    duplicate_path.write_text(
        decision_round.canonical_json() + "\n" + decision_round.canonical_json() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DecisionRoundLedgerError, match="repeats round_id"):
        DecisionRoundLedger(duplicate_path).replay_and_verify()

    unwritten_path = tmp_path / "unwritten.jsonl"
    with pytest.raises(DecisionRoundLedgerError, match="repeats round_id"):
        DecisionRoundLedger(unwritten_path).write_exclusive((decision_round, decision_round))
    assert not unwritten_path.exists()


def test_temporal_and_identifier_invariants_reject_invalid_rounds() -> None:
    with pytest.raises(ValidationError, match="source_available_at must be no later"):
        _decision_round(
            source_available_at="2026-07-16T20:00:01Z",
            decision_time="2026-07-16T20:00:00Z",
        )

    with pytest.raises(ValidationError, match="timezone-aware"):
        _decision_round(source_available_at=datetime(2026, 7, 15, 12, 0))

    with pytest.raises(ValidationError):
        _decision_round(run_id="")

    with pytest.raises(ValidationError, match="evidence_ids must be unique"):
        CurrentSourceRetrieval(evidence_ids=("S1", "S1"))

    payload = _decision_round().model_dump(mode="json")
    payload["outcome"] = {"forward_return": 1.0}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DecisionRound.model_validate_json(_canonical_json(payload))


def test_replay_rejects_duplicate_keys_malformed_and_noncanonical_rows(
    tmp_path: Path,
) -> None:
    decision_round = _decision_round()

    duplicate_key_path = tmp_path / "duplicate-key.jsonl"
    duplicate_key_path.write_text(
        decision_round.canonical_json().replace(
            "{",
            '{"run_id":"duplicate",',
            1,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DecisionRoundLedgerError, match="repeats JSON key 'run_id'"):
        DecisionRoundLedger(duplicate_key_path).replay_and_verify()

    malformed_path = tmp_path / "malformed.jsonl"
    malformed_path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(DecisionRoundLedgerError, match="not valid JSON"):
        DecisionRoundLedger(malformed_path).replay_and_verify()

    noncanonical_path = tmp_path / "noncanonical.jsonl"
    noncanonical_path.write_text(
        json.dumps(decision_round.model_dump(mode="json"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DecisionRoundLedgerError, match="not canonical JSON"):
        DecisionRoundLedger(noncanonical_path).replay_and_verify()


def test_replay_rejects_missing_ledger(tmp_path: Path) -> None:
    with pytest.raises(DecisionRoundLedgerError, match="does not exist"):
        DecisionRoundLedger(tmp_path / "missing.jsonl").replay_and_verify()


def test_round_rejects_cross_field_generation_and_usage_contradictions() -> None:
    raw_mismatch = _decision_round().model_dump(mode="json")
    raw_mismatch["round_id"] = ""
    raw_mismatch["raw_generation"]["generated_text"] = '{"annotations":[]}'
    with pytest.raises(ValidationError, match="match the raw generation"):
        DecisionRound.model_validate_json(_canonical_json(raw_mismatch))

    cached_usage = _decision_round().model_dump(mode="json")
    cached_usage["round_id"] = ""
    cached_usage["inference_source"] = "cache"
    with pytest.raises(ValidationError, match="cannot report new inference usage"):
        DecisionRound.model_validate_json(_canonical_json(cached_usage))

    truncated = _decision_round().model_dump(mode="json")
    truncated["round_id"] = ""
    truncated["raw_generation"]["output_truncated"] = True
    with pytest.raises(ValidationError, match="truncated generation"):
        DecisionRound.model_validate_json(_canonical_json(truncated))


def test_round_accepts_only_canonical_verified_input_too_long_abstentions() -> None:
    payload = _decision_round().model_dump(mode="json")
    payload["round_id"] = ""
    payload["raw_generation"].update(
        {
            "generated_text": None,
            "input_too_long": True,
            "output_truncated": False,
        }
    )
    payload["structured_output"] = {
        "item_id": "item-001",
        "annotations": [
            {
                "asset_id": "asset_aaa",
                "stance_label": "abstain",
                "semantic_signal": 0,
                "raw_confidence": 0.0,
                "uncertainty": 1.0,
                "horizon_days": 5,
                "primary_event_type": None,
                "event_confidence": 0.0,
                "supporting_evidence_span_ids": [],
                "counterevidence_span_ids": [],
                "mechanism": None,
                "invalidation_conditions": [],
                "abstain_reason": "input_too_long",
            }
        ],
    }

    validated = DecisionRound.model_validate_json(_canonical_json(payload))
    assert validated.raw_generation.input_too_long is True

    payload["round_id"] = ""
    payload["structured_output"]["annotations"][0]["abstain_reason"] = "other"
    with pytest.raises(ValidationError, match="canonical abstentions"):
        DecisionRound.model_validate_json(_canonical_json(payload))
