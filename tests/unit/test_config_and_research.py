from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from nlp_trader.config import load_config, validate_config
from nlp_trader.research import create_run_context, finalize_run


def _config(tmp_path: Path) -> Path:
    for name in ("assets.csv", "market.csv", "text.jsonl"):
        (tmp_path / name).write_text("fixture\n", encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(
        """
mode: sample
paths:
  assets: assets.csv
  market_bars: market.csv
  text_items: text.jsonl
  raw_dir: raw
  interim_dir: interim
  processed_dir: processed
  models_dir: models
  reports_dir: reports
features:
  windows_days: [1, 3]
  horizon_days: 1
  feature_set_version: test-features
  label_version: test-labels
  model_version: test-model
backtest:
  commission_bps: 1
  half_spread_bps: 2
  slippage_bps: 3
  borrow_bps_per_year: 0
  max_position_weight: 0.5
  max_gross_exposure: 1
  max_net_exposure: 1
  max_daily_turnover: 1
  max_participation_rate: 0.05
  min_price: 1
  min_dollar_volume: 1
  shorting_allowed: false
  hard_to_borrow_allowed: false
""",
        encoding="utf-8",
    )
    return path


def test_yaml_config_is_typed_hashed_and_validated(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path))

    assert config.mode == "sample"
    assert config.paths.assets == (tmp_path / "assets.csv").resolve()
    assert len(config.content_hash()) == 64
    assert validate_config(config) == []


def test_daily_pipeline_rejects_open_decisions(tmp_path: Path) -> None:
    path = _config(tmp_path)
    value = path.read_text(encoding="utf-8").replace(
        "  windows_days: [1, 3]",
        "  windows_days: [1, 3]\n  decision_time: open",
    )
    path.write_text(value, encoding="utf-8")

    with pytest.raises(ValidationError, match="decision_time"):
        load_config(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["data"].update(storage_format="jsonl"), "storage_format"),
        (
            lambda value: value.setdefault("models", {}).update(
                families=["traditional", "combined"]
            ),
            "models.families",
        ),
        (
            lambda value: value["features"].update(horizon_days=2),
            "rebalance_frequency",
        ),
        (
            lambda value: value.setdefault("runtime", {}).update(
                start_date="2026-07-10", end_date="2026-07-01"
            ),
            "runtime.end_date",
        ),
    ],
)
def test_config_rejects_options_the_pipeline_cannot_honor(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    path = _config(tmp_path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    if "data" not in value:
        value["data"] = {}
    mutation(value)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(ValidationError, match=message):
        load_config(path)


def test_run_directories_and_final_manifest_are_immutable(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path))
    now = datetime(2026, 7, 14, 1, 2, 3, tzinfo=UTC)
    first = create_run_context(config, now=now)
    second = create_run_context(config, now=now)

    assert second.run_id == f"{first.run_id}-01"
    final_path = finalize_run(
        first,
        universe=["BBB", "AAA"],
        period={"start": "2026-01-01", "end": "2026-06-30"},
        metrics={"sharpe": 0.0},
        known_limitations=["synthetic only"],
        next_questions=["collect licensed data"],
    )
    manifest = json.loads(final_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["universe"] == ["AAA", "BBB"]
    assert manifest["data_manifest"][0]["sha256"]

    with pytest.raises(FileExistsError):
        finalize_run(
            first,
            universe=[],
            period={"start": None, "end": None},
            metrics={},
            known_limitations=[],
            next_questions=[],
        )
