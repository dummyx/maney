from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from nlp_trader.timestamps import parse_utc


class FrozenModel(BaseModel):
    """Strict immutable configuration base used by every runtime section."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class PathsConfig(FrozenModel):
    assets: Path
    market_bars: Path
    text_items: Path
    fundamentals: Path | None = None
    earnings_calendar: Path | None = None
    corporate_actions: Path | None = None
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path


class FeatureConfig(FrozenModel):
    windows_days: tuple[int, ...] = (1, 3, 5, 20)
    market_warmup_sessions: int = Field(default=60, ge=60)
    text_warmup_days: int = Field(default=40, ge=1)
    event_lookahead_days: int = Field(default=30, ge=1)
    horizon_days: int = Field(default=1, ge=1)
    feature_set_version: str = Field(min_length=1)
    label_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    text_decay_half_life_days: float = Field(default=1.0, gt=0)
    decision_time: Literal["close"] = "close"

    @model_validator(mode="after")
    def validate_windows(self) -> FeatureConfig:
        if not self.windows_days:
            raise ValueError("features.windows_days must not be empty")
        if any(day < 1 for day in self.windows_days):
            raise ValueError("features.windows_days values must be >= 1")
        if len(set(self.windows_days)) != len(self.windows_days):
            raise ValueError("features.windows_days values must be unique")
        minimum_text_history = 2 * max(self.windows_days)
        if self.text_warmup_days < minimum_text_history:
            raise ValueError(
                "features.text_warmup_days must be at least twice the largest text window "
                f"({minimum_text_history})"
            )
        return self


class ModelConfig(FrozenModel):
    families: tuple[Literal["traditional", "text", "combined"], ...] = (
        "traditional",
        "text",
        "combined",
    )
    min_train_rows: int = Field(default=4, ge=2)
    embargo_periods: int = Field(default=0, ge=0)
    top_k: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def validate_families(self) -> ModelConfig:
        if self.families != ("traditional", "text", "combined"):
            raise ValueError(
                "models.families must be [traditional, text, combined] for this baseline"
            )
        return self


class BacktestConfig(FrozenModel):
    commission_bps: float = Field(ge=0)
    half_spread_bps: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)
    volatility_slippage_multiplier: float = Field(default=0.05, ge=0)
    participation_slippage_bps: float = Field(default=50.0, ge=0)
    market_impact_multiplier: float = Field(default=0.10, ge=0)
    borrow_bps_per_year: float = Field(ge=0)
    max_position_weight: float = Field(gt=0, le=1)
    max_gross_exposure: float = Field(gt=0)
    max_net_exposure: float = Field(ge=0)
    max_sector_weight: float = Field(default=1.0, gt=0, le=1)
    max_beta_exposure: float = Field(default=1.0, ge=0)
    missing_beta_fallback: float = Field(default=1.0, ge=0)
    missing_volatility_floor: float = Field(default=0.03, gt=0)
    max_daily_turnover: float = Field(gt=0)
    same_day_exit_notional_buffer: float = Field(default=0.10, ge=0)
    max_participation_rate: float = Field(gt=0, le=1)
    min_price: float = Field(gt=0)
    min_dollar_volume: float = Field(ge=0)
    shorting_allowed: bool
    hard_to_borrow_allowed: bool
    initial_capital: float = Field(default=1_000_000.0, gt=0)
    rebalance_frequency: str = Field(default="1d", pattern=r"^[1-9][0-9]*d$")
    benchmark: Literal["equal_weight"] = "equal_weight"

    @model_validator(mode="after")
    def validate_exposures(self) -> BacktestConfig:
        if self.max_position_weight > self.max_gross_exposure:
            raise ValueError("max_position_weight cannot exceed max_gross_exposure")
        if self.max_net_exposure > self.max_gross_exposure:
            raise ValueError("max_net_exposure cannot exceed max_gross_exposure")
        return self


class DataConfig(FrozenModel):
    storage_format: Literal["parquet"] = "parquet"
    compression: Literal["zstd", "snappy", "uncompressed"] = "zstd"
    write_batch_rows: int = Field(default=10_000, ge=1)
    calendar: str = "XNYS"
    schema_version: str = "1"
    market_license_or_terms_ref: str = Field(default="user-provided-local", min_length=1)
    text_license_or_terms_ref: str = Field(default="user-provided-local", min_length=1)


class RuntimeConfig(FrozenModel):
    limit: int | None = Field(default=None, ge=1)
    start_date: str | None = None
    end_date: str | None = None
    symbols: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_filters(self) -> RuntimeConfig:
        parsed: dict[str, date] = {}
        for name in ("start_date", "end_date"):
            value = getattr(self, name)
            if value is None:
                continue
            try:
                parsed[name] = (
                    parse_utc(value).date() if "T" in value else date.fromisoformat(value)
                )
            except ValueError as exc:
                raise ValueError(
                    f"runtime.{name} must be an ISO date or timezone-aware timestamp"
                ) from exc
        if (
            "start_date" in parsed
            and "end_date" in parsed
            and parsed["end_date"] < parsed["start_date"]
        ):
            raise ValueError("runtime.end_date must be on or after runtime.start_date")
        if len(self.symbols) != len(set(self.symbols)):
            raise ValueError("runtime.symbols must be unique")
        if any(symbol != symbol.upper() or not symbol for symbol in self.symbols):
            raise ValueError("runtime.symbols must be non-empty uppercase symbols")
        return self


class TransformerConfig(FrozenModel):
    enabled: bool = False
    model_name: str | None = None
    model_version: str = "local-transformer-v1"
    batch_size: int = Field(default=32, ge=1)
    max_sequence_length: int = Field(default=256, ge=1)
    local_files_only: bool = True

    @model_validator(mode="after")
    def validate_model_name(self) -> TransformerConfig:
        if self.enabled and not self.model_name:
            raise ValueError("transformer.model_name is required when transformer.enabled is true")
        return self


class ResearchConfig(FrozenModel):
    path: Path
    mode: Literal["sample", "full"]
    paths: PathsConfig
    features: FeatureConfig
    models: ModelConfig = ModelConfig()
    backtest: BacktestConfig
    data: DataConfig = DataConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    transformer: TransformerConfig = TransformerConfig()

    @model_validator(mode="after")
    def validate_horizon_alignment(self) -> ResearchConfig:
        rebalance_days = int(self.backtest.rebalance_frequency.removesuffix("d"))
        if rebalance_days != self.features.horizon_days:
            raise ValueError("backtest.rebalance_frequency must match features.horizon_days")
        return self

    def content_hash(self) -> str:
        """Hash runtime-relevant config content, excluding its machine-local filename."""

        payload = self.model_dump(mode="json", exclude={"path"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class SecretSettings(BaseSettings):
    """Optional credentials; never serialized into run manifests or reports."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="NLP_TRADER_",
        extra="ignore",
    )

    news_api_key: str | None = None
    market_data_api_key: str | None = None


def _path(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _resolve_paths(base: Path, raw: dict[str, Any]) -> dict[str, Any]:
    values = dict(raw)
    values.setdefault("raw_dir", "../data/raw")
    required = {
        "assets",
        "market_bars",
        "text_items",
        "raw_dir",
        "interim_dir",
        "processed_dir",
        "models_dir",
        "reports_dir",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"missing paths configuration: {', '.join(missing)}")
    return {name: None if value is None else _path(base, value) for name, value in values.items()}


def load_config(path: str | Path) -> ResearchConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"config file does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in config {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"config root must be a mapping: {config_path}")
    raw = dict(loaded)
    paths = raw.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("config.paths must be a mapping")
    raw["path"] = config_path
    raw["paths"] = _resolve_paths(config_path.parent, paths)
    return ResearchConfig.model_validate(raw)


def validate_config(config: ResearchConfig, *, require_inputs: bool = True) -> list[str]:
    """Return actionable validation errors without performing network access."""

    errors: list[str] = []
    input_names = (
        "assets",
        "market_bars",
        "text_items",
        "fundamentals",
        "earnings_calendar",
        "corporate_actions",
    )
    if require_inputs:
        for name in ("assets", "market_bars", "text_items"):
            path = getattr(config.paths, name)
            if not path.is_file() and not (path.is_dir() and any(path.rglob("*.parquet"))):
                errors.append(f"missing {name}: {path}")
        for name in ("fundamentals", "earnings_calendar", "corporate_actions"):
            path = getattr(config.paths, name)
            if (
                path is not None
                and not path.is_file()
                and not (path.is_dir() and any(path.rglob("*.parquet")))
            ):
                errors.append(f"missing configured {name}: {path}")
    write_roots = {
        "raw_dir": config.paths.raw_dir,
        "interim_dir": config.paths.interim_dir,
        "processed_dir": config.paths.processed_dir,
        "models_dir": config.paths.models_dir,
        "reports_dir": config.paths.reports_dir,
    }
    write_items = list(write_roots.items())
    for index, (left_name, left) in enumerate(write_items):
        for right_name, right in write_items[index + 1 :]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                errors.append(f"write roots must not overlap: {left_name} and {right_name}")
    for input_name in input_names:
        input_path = getattr(config.paths, input_name)
        if input_path is None:
            continue
        for root_name, root in write_items:
            if (
                input_path == root
                or input_path.is_relative_to(root)
                or root.is_relative_to(input_path)
            ):
                errors.append(f"input {input_name} must not overlap write root {root_name}")
    if config.mode == "full" and config.runtime.limit is not None and config.runtime.limit < 1:
        errors.append("runtime.limit must be positive when set")
    return errors
