from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nlp_trader.config import ResearchConfig, load_config
from nlp_trader.data.synthetic import generate_synthetic_fixture


@pytest.fixture
def generated_config(tmp_path: Path) -> ResearchConfig:
    """Build a complete test config from generated data only, with no network access."""

    fixture = generate_synthetic_fixture(
        tmp_path / "generated-data",
        seed=23,
        symbols=("AAA", "BBB", "CCC"),
        session_count=14,
    )
    config = {
        "mode": "sample",
        "paths": {
            "assets": str(fixture.assets),
            "market_bars": str(fixture.market_bars),
            "text_items": str(fixture.text_items),
            "raw_dir": str(tmp_path / "artifacts" / "raw"),
            "interim_dir": str(tmp_path / "artifacts" / "interim"),
            "processed_dir": str(tmp_path / "artifacts" / "processed"),
            "models_dir": str(tmp_path / "artifacts" / "models"),
            "reports_dir": str(tmp_path / "artifacts" / "reports"),
        },
        "features": {
            "windows_days": [1, 3, 5],
            "horizon_days": 1,
            "feature_set_version": "generated-test-features-v1",
            "label_version": "generated-test-labels-v1",
            "model_version": "generated-test-model-v1",
            "text_decay_half_life_days": 1.0,
            "decision_time": "close",
        },
        "models": {
            "families": ["traditional", "text", "combined"],
            "min_train_rows": 4,
            "embargo_periods": 1,
            "top_k": 2,
        },
        "backtest": {
            "commission_bps": 1.0,
            "half_spread_bps": 2.0,
            "slippage_bps": 3.0,
            "borrow_bps_per_year": 0.0,
            "max_position_weight": 0.4,
            "max_gross_exposure": 1.0,
            "max_net_exposure": 1.0,
            "max_sector_weight": 0.8,
            "max_beta_exposure": 1.0,
            "max_daily_turnover": 1.0,
            "max_participation_rate": 0.05,
            "min_price": 1.0,
            "min_dollar_volume": 1_000_000.0,
            "shorting_allowed": False,
            "hard_to_borrow_allowed": False,
        },
        "data": {
            "storage_format": "parquet",
            "compression": "zstd",
            "calendar": "XNYS",
            "schema_version": "generated-v1",
            "market_license_or_terms_ref": "synthetic-fixture-v1",
            "text_license_or_terms_ref": "synthetic-fixture-v1",
        },
        "runtime": {},
        "transformer": {"enabled": False},
    }
    config_path = tmp_path / "generated-test.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return load_config(config_path)
