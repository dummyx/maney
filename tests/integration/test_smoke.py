from __future__ import annotations

import json
from pathlib import Path

from nlp_trader.config import ResearchConfig
from nlp_trader.data.parquet import read_partitioned_parquet
from nlp_trader.pipeline import build_features, paper, smoke


def test_smoke_pipeline_uses_only_generated_local_data(
    generated_config: ResearchConfig,
) -> None:
    outputs = smoke(generated_config)

    report = outputs["report"]
    assert report.exists()
    assert "hypothetical research output" in report.read_text(encoding="utf-8")
    assert list(generated_config.paths.processed_dir.rglob("*.parquet"))
    manifest = json.loads(outputs["final_manifest"].read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["completed_stage"] == "report"
    assert all(entry["sha256"] for entry in manifest["data_manifest"])
    assert {entry["area"] for entry in manifest["artifact_manifest"]} == {
        "interim",
        "models",
        "processed",
        "reports",
    }


def _feature_rows(outputs: dict[str, object]) -> list[dict[str, object]]:
    return read_partitioned_parquet(Path(str(outputs["features"])))


def test_runtime_window_preserves_warmup_and_whole_cross_sections(
    generated_config: ResearchConfig,
) -> None:
    full_rows = _feature_rows(build_features(generated_config))
    runtime = generated_config.runtime.model_copy(
        update={"start_date": "2026-07-06", "end_date": "2026-07-10", "limit": 2}
    )
    bounded = generated_config.model_copy(update={"runtime": runtime})
    bounded_rows = _feature_rows(build_features(bounded))

    assert len({str(row["asof_ts"]) for row in bounded_rows}) == 2
    assert {str(row["symbol"]) for row in bounded_rows} == {"AAA", "BBB", "CCC"}
    bounded_aaa = next(row for row in bounded_rows if row["symbol"] == "AAA")
    full_aaa = next(
        row
        for row in full_rows
        if row["symbol"] == "AAA" and row["asof_ts"] == bounded_aaa["asof_ts"]
    )
    assert bounded_aaa["return_3d"] == full_aaa["return_3d"]
    assert bounded_aaa["text_count_5d"] == full_aaa["text_count_5d"]

    earliest = min(
        (row for row in full_rows if row["symbol"] == "AAA"),
        key=lambda row: str(row["asof_ts"]),
    )
    assert earliest["market_beta_60d_missing"] is True
    assert earliest["beta_fallback_used"] is True
    assert earliest["beta"] == generated_config.backtest.missing_beta_fallback
    assert earliest["realized_volatility_20d_missing"] is True
    assert earliest["volatility_fallback_used"] is True
    assert earliest["volatility"] == max(
        generated_config.backtest.missing_volatility_floor,
        float(earliest["realized_volatility_3d"]),
        float(earliest["high_low_volatility_20d"]),
    )


def test_optional_point_in_time_context_is_consumed_from_generated_files(
    generated_config: ResearchConfig,
) -> None:
    base = generated_config.path.parent
    fundamentals = base / "fundamentals.jsonl"
    earnings = base / "earnings.jsonl"
    actions = base / "actions.jsonl"
    fundamentals.write_text(
        json.dumps(
            {
                "asset_id": "asset_aaa",
                "symbol": "AAA",
                "period_end": "2026-03-31",
                "available_at": "2026-06-01T12:00:00Z",
                "values": {"book_to_market": 0.4, "return_on_equity": 0.2},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    earnings.write_text(
        json.dumps(
            {
                "asset_id": "asset_aaa",
                "symbol": "AAA",
                "event_ts": "2026-07-10T20:00:00Z",
                "available_at": "2026-06-15T12:00:00Z",
                "status": "confirmed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    actions.write_text(
        json.dumps(
            {
                "asset_id": "asset_aaa",
                "symbol": "AAA",
                "event_ts": "2026-07-13T13:30:00Z",
                "available_at": "2026-06-15T12:00:00Z",
                "action_type": "ex_dividend",
                "value": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = generated_config.paths.model_copy(
        update={
            "fundamentals": fundamentals,
            "earnings_calendar": earnings,
            "corporate_actions": actions,
        }
    )
    runtime = generated_config.runtime.model_copy(
        update={"start_date": "2026-07-06", "end_date": "2026-07-08"}
    )
    configured = generated_config.model_copy(update={"paths": paths, "runtime": runtime})

    rows = _feature_rows(build_features(configured))
    latest = max(
        (row for row in rows if row["symbol"] == "AAA"),
        key=lambda row: str(row["asof_ts"]),
    )

    assert latest["value_proxy"] == 0.4
    assert latest["quality_proxy"] == 0.2
    assert latest["earnings_proximity_missing"] is False
    assert latest["ex_dividend_proximity_missing"] is False


def test_pipeline_paper_output_is_unfilled_intent_only(
    generated_config: ResearchConfig,
) -> None:
    outputs = paper(generated_config)
    snapshot = json.loads(Path(outputs["paper"]).read_text(encoding="utf-8"))

    assert snapshot["status"] == "pending_unfilled_intents"
    assert snapshot["trades"] == []
    assert snapshot["positions"] == {}
    assert snapshot["equity"] == snapshot["initial_capital"]
