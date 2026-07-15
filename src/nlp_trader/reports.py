from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nlp_trader.config import ResearchConfig

DEFAULT_LIMITATIONS = [
    "Results are hypothetical and depend on the configured data, execution, and cost assumptions.",
    "The sample mode uses synthetic fixtures and is only an implementation smoke test.",
    "Intraday queue position, venue-level fills, taxes, and emergency exchange closures are not "
    "modeled.",
    "External vendor revisions and survivorship bias require provider-specific point-in-time "
    "datasets.",
]


def run_id(config: ResearchConfig, metrics: dict[str, Any]) -> str:
    """Return a deterministic content identifier for legacy report callers."""

    payload = json.dumps(
        {"config_hash": config.content_hash(), "metrics": metrics},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _metric_lines(metrics: dict[str, Any], *, prefix: str = "") -> list[str]:
    lines: list[str] = []
    for key, value in sorted(metrics.items()):
        label = f"{prefix}{key}"
        if isinstance(value, dict):
            lines.extend(_metric_lines(value, prefix=f"{label}."))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {label}: `{value}`")
    return lines


def _period_diagnostic_lines(periods: list[dict[str, Any]]) -> list[str]:
    if not periods:
        return ["- No replay periods were produced."]
    lines = [
        "| decision | execution | exit | net return | turnover | gross | net | beta | costs | "
        "sectors | flags |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in periods:
        sectors = json.dumps(row.get("sector_exposure", {}), sort_keys=True, separators=(",", ":"))
        flags = ", ".join(str(value) for value in row.get("risk_flags", [])) or "none"
        lines.append(
            f"| {row.get('asof_ts')} | {row.get('execution_ts')} | {row.get('exit_ts')} | "
            f"{row.get('net_return', 0.0):.8f} | {row.get('turnover', 0.0):.6f} | "
            f"{row.get('gross_exposure', 0.0):.6f} | {row.get('net_exposure', 0.0):.6f} | "
            f"{row.get('beta_exposure', 0.0):.6f} | {row.get('cost_return', 0.0):.8f} | "
            f"`{sectors}` | {flags} |"
        )
    return lines


def write_report(
    config: ResearchConfig,
    backtest: dict[str, Any],
    path: Path,
    *,
    report_run_id: str | None = None,
    created_at: datetime | None = None,
    code_version: dict[str, Any] | None = None,
    data_manifest: list[dict[str, Any]] | None = None,
    universe: list[str] | None = None,
    period: dict[str, str | None] | None = None,
    model_evaluation: dict[str, Any] | None = None,
    known_limitations: list[str] | None = None,
    next_questions: list[str] | None = None,
) -> Path:
    """Write a complete, human-readable and assumption-explicit research note."""

    metrics = backtest.get("metrics", {})
    evaluation_window = backtest.get("evaluation_window", {})
    evaluation_window_name = (
        str(evaluation_window.get("name", "unspecified"))
        if isinstance(evaluation_window, dict)
        else "unspecified"
    )
    rid = report_run_id or run_id(config, metrics)
    timestamp = (created_at or datetime.now(UTC)).astimezone(UTC)
    code = code_version or {"git_commit": None, "dirty": None}
    interval = period or {"start": None, "end": None}
    limitations = known_limitations or DEFAULT_LIMITATIONS
    questions = next_questions or [
        "Does the combined model improve out-of-sample results over traditional-only and naive "
        "baselines?",
        "Are results stable across liquidity, sector, and volatility regimes after costs?",
    ]
    lines = [
        "# Research Run Summary",
        "",
        "This report is hypothetical research output. It is assumption-dependent, is not financial "
        "advice, and does not authorize live trading.",
        "",
        "## Provenance",
        "",
        f"- run_id: `{rid}`",
        f"- created_at: `{timestamp.isoformat().replace('+00:00', 'Z')}`",
        f"- code_version/git_commit: `{code.get('git_commit') or 'uncommitted'}`",
        f"- dirty_worktree: `{code.get('dirty')}`",
        f"- config_hash: `{config.content_hash()}`",
        f"- mode: `{config.mode}`",
        f"- universe: `{', '.join(sorted(set(universe or []))) or 'not recorded'}`",
        f"- period: `{interval.get('start')} to {interval.get('end')}`",
        f"- rebalance_frequency: `{config.backtest.rebalance_frequency}`",
        f"- feature_set_version: `{config.features.feature_set_version}`",
        f"- label_version: `{config.features.label_version}`",
        f"- model_version: `{config.features.model_version}`",
        f"- horizon: `{config.features.horizon_days} sessions`",
        "",
        "## Data Manifest",
        "",
    ]
    if data_manifest:
        for entry in data_manifest:
            lines.append(
                f"- {entry.get('role')}: sha256=`{entry.get('sha256')}`, "
                f"bytes=`{entry.get('bytes')}`, exists=`{entry.get('exists')}`"
            )
    else:
        lines.append("- No data manifest was supplied by the caller.")
    lines.extend(
        [
            "",
            "## Cost and Fill Model",
            "",
            f"- commission_bps: `{config.backtest.commission_bps}`",
            f"- half_spread_bps: `{config.backtest.half_spread_bps}`",
            f"- base_slippage_bps: `{config.backtest.slippage_bps}`",
            "- slippage adjustment: volatility and participation-rate dependent",
            f"- volatility_slippage_multiplier: `{config.backtest.volatility_slippage_multiplier}`",
            f"- participation_slippage_bps: `{config.backtest.participation_slippage_bps}`",
            f"- market_impact_multiplier: `{config.backtest.market_impact_multiplier}`",
            f"- borrow_bps_per_year: `{config.backtest.borrow_bps_per_year}`",
            "- fills: decision after the official close, entry at the next session open, and "
            "forced exit at the configured horizon close",
            "",
            "## Portfolio Constraints",
            "",
            f"- max_position_weight: `{config.backtest.max_position_weight}`",
            f"- max_gross_exposure: `{config.backtest.max_gross_exposure}`",
            f"- max_net_exposure: `{config.backtest.max_net_exposure}`",
            f"- max_sector_weight: `{config.backtest.max_sector_weight}`",
            f"- max_beta_exposure: `{config.backtest.max_beta_exposure}`",
            f"- missing_beta_fallback: `{config.backtest.missing_beta_fallback}`",
            f"- missing_volatility_floor: `{config.backtest.missing_volatility_floor}`",
            f"- max_daily_turnover: `{config.backtest.max_daily_turnover}`",
            f"- same_day_exit_notional_buffer: `{config.backtest.same_day_exit_notional_buffer}`",
            f"- max_participation_rate: `{config.backtest.max_participation_rate}`",
            f"- min_price: `{config.backtest.min_price}`",
            f"- min_dollar_volume: `{config.backtest.min_dollar_volume}`",
            f"- shorting_allowed: `{config.backtest.shorting_allowed}`",
            f"- hard_to_borrow_allowed: `{config.backtest.hard_to_borrow_allowed}`",
            "",
            f"## Backtest Metrics ({evaluation_window_name})",
            "",
        ]
    )
    lines.extend(_metric_lines(metrics) or ["- No metrics were produced."])
    lines.extend(["", "## Per-Period Diagnostics", ""])
    lines.extend(_period_diagnostic_lines(list(backtest.get("periods", []))))
    lines.extend(
        [
            "",
            "## Detailed Replay Logs",
            "",
            f"- trade rows: `{len(backtest.get('trades', []))}`",
            f"- position rows: `{len(backtest.get('positions', []))}`",
            "- Complete trade, position, rejection, cost, and risk-flag records are retained in "
            "the run's processed backtest JSON artifacts.",
        ]
    )
    lines.extend(["", "## Model Evaluation", ""])
    lines.extend(_metric_lines(model_evaluation or {}) or ["- No model evaluation was supplied."])
    lines.extend(["", "## Known Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in limitations)
    lines.extend(["", "## Next Questions", ""])
    lines.extend(f"- {question}" for question in questions)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path
