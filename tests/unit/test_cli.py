from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from nlp_trader import pipeline
from nlp_trader.cli import app, run_cli
from nlp_trader.config import ResearchConfig, load_config


def _config(tmp_path: Path) -> Path:
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
  feature_set_version: cli-features
  label_version: cli-labels
  model_version: cli-model
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
transformer:
  enabled: false
  model_name: local-test-model
""",
        encoding="utf-8",
    )
    return path


def test_typer_help_lists_preserved_and_new_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for name in (
        "validate-config",
        "ingest-market",
        "ingest-text",
        "build-features",
        "build-labels",
        "train",
        "predict",
        "backtest",
        "paper",
        "report",
        "smoke",
        "generate-synthetic",
    ):
        assert name in result.stdout


def test_run_cli_applies_immutable_runtime_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _config(tmp_path)
    captured: list[ResearchConfig] = []

    def fake_ingest(config: ResearchConfig) -> dict[str, object]:
        captured.append(config)
        return {"stage": "ingest-market"}

    monkeypatch.setattr(pipeline, "ingest_market", fake_ingest)

    result = run_cli(
        [
            "ingest-market",
            "--config",
            str(config_path),
            "--limit",
            "7",
            "--start-date",
            "2026-07-01",
            "--end-date",
            "2026-07-10",
            "--symbol",
            "AAA",
            "--symbol",
            "BBB",
        ]
    )

    assert result == 0
    assert captured[0].runtime.limit == 7
    assert captured[0].runtime.start_date == "2026-07-01"
    assert captured[0].runtime.end_date == "2026-07-10"
    assert captured[0].runtime.symbols == ("AAA", "BBB")
    assert load_config(config_path).runtime.limit is None
    assert "stage: ingest-market" in capsys.readouterr().out


def test_smoke_forwards_optional_transformer_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _config(tmp_path)
    captured: dict[str, object] = {}

    def fake_smoke(config: ResearchConfig) -> dict[str, object]:
        captured["config"] = config
        captured["enabled"] = config.transformer.enabled
        return {"ok": True}

    monkeypatch.setattr(pipeline, "smoke", fake_smoke)

    assert (
        run_cli(
            [
                "smoke",
                "--config",
                str(config_path),
                "--enable-transformer-sentiment",
            ]
        )
        == 0
    )
    assert captured["enabled"] is True


def test_runtime_overrides_are_revalidated(tmp_path: Path) -> None:
    config_path = _config(tmp_path)

    with pytest.raises(ValueError, match="runtime.end_date must be on or after"):
        run_cli(
            [
                "ingest-market",
                "--config",
                str(config_path),
                "--start-date",
                "2026-07-10",
                "--end-date",
                "2026-07-01",
            ]
        )


def test_validate_config_returns_nonzero_and_prints_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _config(tmp_path)

    def fake_validate(config: ResearchConfig) -> dict[str, object]:
        assert config.runtime.limit == 2
        return {"ok": False, "errors": ["missing licensed input"]}

    monkeypatch.setattr(pipeline, "validate", fake_validate)

    assert (
        run_cli(
            [
                "validate-config",
                "--config",
                str(config_path),
                "--limit",
                "2",
            ]
        )
        == 1
    )
    assert "missing licensed input" in capsys.readouterr().err


def test_generate_synthetic_command_is_deterministic_and_local(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    arguments = ["--seed", "19", "--session-count", "3", "--symbol", "AAA"]

    assert run_cli(["generate-synthetic", "--output-dir", str(first), *arguments]) == 0
    assert run_cli(["generate-synthetic", "--output-dir", str(second), *arguments]) == 0

    for name in ("assets.csv", "market_bars.csv", "text_items.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
