from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from nlp_trader.config import ResearchConfig, load_config
from nlp_trader.data.parquet import read_partitioned_parquet
from nlp_trader.paper.ledger import PaperEventLedger
from nlp_trader.pipeline import build_features, paper, smoke
from nlp_trader.portfolio.constraints import round_trip_entry_constraints


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


def test_xjpx_smoke_uses_after_close_cross_section_availability(tmp_path: Path) -> None:
    assets_path = tmp_path / "assets.csv"
    asset_rows = [
        {
            "asset_id": f"asset_{symbol}",
            "symbol": symbol,
            "exchange": "XJPX",
            "currency": "JPY",
            "name": name,
            "sector": sector,
            "active_from": "2026-07-07",
            "active_to": "",
            "trading_unit": 100,
        }
        for symbol, name, sector in (
            ("7203", "Synthetic Toyota", "Industrials"),
            ("6758", "Synthetic Sony", "Technology"),
        )
    ]
    with assets_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asset_rows[0]))
        writer.writeheader()
        writer.writerows(asset_rows)

    bars_path = tmp_path / "bars.csv"
    sessions = (7, 8, 9, 10, 13, 14, 15)
    bar_rows: list[dict[str, object]] = []
    for index, day in enumerate(sessions):
        for symbol, delay_minutes, base_price in (("7203", 60, 1000), ("6758", 75, 2000)):
            price = float(base_price + index * 10)
            available_hour = 7
            available_minute = delay_minutes
            if available_minute >= 60:
                available_hour += available_minute // 60
                available_minute %= 60
            available_at = f"2026-07-{day:02d}T{available_hour:02d}:{available_minute:02d}:00Z"
            bar_rows.append(
                {
                    "asset_id": f"asset_{symbol}",
                    "symbol": symbol,
                    "exchange": "XJPX",
                    "currency": "JPY",
                    "trading_unit": 100,
                    "session_date": f"2026-07-{day:02d}",
                    "ts": f"2026-07-{day:02d}T06:30:00Z",
                    "available_at": available_at,
                    "bar_size": "1d",
                    "open": price,
                    "high": price + 5.0,
                    "low": price - 5.0,
                    "close": price + 2.0,
                    "volume": 1_000_000,
                    "corporate_action_adjusted": "true",
                    "adjustment_vintage_at": available_at,
                    "return_adjustment_factor": 1.0,
                    "price_basis": "raw_tradable",
                }
            )
    with bars_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bar_rows[0]))
        writer.writeheader()
        writer.writerows(bar_rows)

    text_path = tmp_path / "text.jsonl"
    text_path.write_text(
        json.dumps(
            {
                "item_id": "post-close-before-bar-delivery",
                "source": "synthetic-news",
                "source_type": "news",
                "language": "en",
                "title": "Synthetic Toyota demand improved",
                "body": "A permitted synthetic test item.",
                "published_at": "2026-07-07T08:05:00Z",
                "vendor_received_at": "2026-07-07T08:05:00Z",
                "ingested_at": "2026-07-07T08:05:00Z",
                "available_at": "2026-07-07T08:05:00Z",
                "license_or_terms_ref": "synthetic-fixture-only",
                "entities": [
                    {
                        "name": "Synthetic Toyota",
                        "asset_id": "asset_7203",
                        "symbol": "7203",
                        "relevance": 1.0,
                        "mention_type": "primary",
                        "confidence": 1.0,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "xjpx.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "mode": "sample",
                "paths": {
                    "assets": str(assets_path),
                    "market_bars": str(bars_path),
                    "text_items": str(text_path),
                    "raw_dir": str(tmp_path / "artifacts/raw"),
                    "interim_dir": str(tmp_path / "artifacts/interim"),
                    "processed_dir": str(tmp_path / "artifacts/processed"),
                    "models_dir": str(tmp_path / "artifacts/models"),
                    "reports_dir": str(tmp_path / "artifacts/reports"),
                },
                "features": {
                    "windows_days": [1],
                    "market_warmup_sessions": 60,
                    "text_warmup_days": 2,
                    "horizon_days": 1,
                    "feature_set_version": "xjpx-test-v1",
                    "label_version": "xjpx-test-v1",
                    "model_version": "xjpx-test-v1",
                },
                "models": {
                    "families": ["traditional", "text", "combined"],
                    "min_train_rows": 2,
                    "embargo_periods": 0,
                    "final_holdout_periods": 1,
                    "top_k": 1,
                },
                "backtest": {
                    "commission_bps": 1.0,
                    "half_spread_bps": 2.0,
                    "slippage_bps": 3.0,
                    "borrow_bps_per_year": 0.0,
                    "max_position_weight": 0.5,
                    "max_gross_exposure": 1.0,
                    "max_net_exposure": 1.0,
                    "max_daily_turnover": 1.0,
                    "max_participation_rate": 0.05,
                    "min_price": 1.0,
                    "min_dollar_volume": 1.0,
                    "shorting_allowed": False,
                    "hard_to_borrow_allowed": False,
                    "rebalance_frequency": "1d",
                },
                "data": {
                    "calendar": "XJPX",
                    "market_contract": "japan_cash_equity_v1",
                    "schema_version": "synthetic-xjpx-test-v1",
                    "market_license_or_terms_ref": "synthetic-fixture-only",
                    "text_license_or_terms_ref": "synthetic-fixture-only",
                },
                "runtime": {"start_date": "2026-07-07", "limit": 5},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    outputs = smoke(config)
    rows = _feature_rows(outputs)

    assert len({str(row["asof_ts"]) for row in rows}) == 5
    assert {str(row["symbol"]) for row in rows} == {"6758", "7203"}
    assert {str(row["asof_ts"])[11:] for row in rows} == {"08:15:00Z"}
    assert all(str(row["session_close_ts"])[11:] == "06:30:00Z" for row in rows)
    first_toyota = next(
        row for row in rows if row["symbol"] == "7203" and row["asof_ts"] == "2026-07-07T08:15:00Z"
    )
    assert first_toyota["text_count_1d"] == 1
    assert first_toyota["latest_text_available_at_1d"] == "2026-07-07T08:05:00Z"
    labels = read_partitioned_parquet(Path(str(outputs["features"])).parent / "labels")
    assert all(str(row["label_end_ts"])[11:] == "06:30:00Z" for row in labels)
    assert all(str(row["label_available_at"])[11:] == "08:15:00Z" for row in labels)
    model = json.loads(Path(outputs["model"]).read_text(encoding="utf-8"))
    snapshots = {row["asof_ts"]: row for row in model["walk_forward_snapshots"]}
    assert snapshots["2026-07-08T08:15:00Z"]["eligible_training_rows"] == 0
    assert snapshots["2026-07-09T08:15:00Z"]["eligible_training_rows"] == 2
    manifest = json.loads(outputs["final_manifest"].read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["completed_stage"] == "report"

    empty_text_path = tmp_path / "empty-text.jsonl"
    empty_text_path.write_text("", encoding="utf-8")
    empty_config = config.model_copy(
        update={
            "paths": config.paths.model_copy(update={"text_items": empty_text_path}),
        }
    )
    empty_outputs = smoke(empty_config)
    empty_rows = _feature_rows(empty_outputs)
    assert all(row["text_count_1d"] == 0 for row in empty_rows)


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
    effective = round_trip_entry_constraints(
        generated_config.backtest,
        horizon_steps=generated_config.features.horizon_days,
    )
    selected = [intent for intent in snapshot["intents"] if intent["target_weight"]]
    assert selected
    assert sum(abs(intent["target_weight"]) for intent in selected) <= (
        effective.max_daily_turnover + 1e-12
    )
    for intent in selected:
        participation = (
            abs(intent["target_weight"])
            * snapshot["initial_capital"]
            / intent["decision_liquidity_proxy_dollar_volume"]
        )
        assert participation <= effective.max_participation_rate + 1e-12
    ledger_path = Path(snapshot["ledger"]["path"])
    replayed = PaperEventLedger(ledger_path).replay()
    assert replayed[0]["event_type"] == "paper_intent_batch"
    assert replayed[0]["event_hash"] == snapshot["ledger"]["event_hash"]
    assert replayed[0]["status"] == "pending_unfilled_intents"
    assert replayed[0]["research_protocol"] == snapshot["research_protocol"]
    assert snapshot["research_protocol"]["config_hash"] == generated_config.content_hash()
