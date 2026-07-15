from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time as clock
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from nlp_trader.backtest.engine import DeterministicBacktestEngine
from nlp_trader.calendars import USEquityCalendar
from nlp_trader.config import ResearchConfig, validate_config
from nlp_trader.data.local import asset_to_record, market_bar_to_record, text_item_to_record
from nlp_trader.data.parquet import write_partitioned_parquet
from nlp_trader.data.stores import (
    ContentAddressedRawStore,
    LocalModelRegistry,
    ParquetFeatureStore,
    RawIngestionRequest,
)
from nlp_trader.features.build import build_feature_rows, build_label_rows
from nlp_trader.market_timing import (
    market_bar_available_at,
    market_decision_time_for_bar,
    market_decision_times_by_session,
)
from nlp_trader.models.baselines import predict_all_families, train_baselines
from nlp_trader.models.evaluation import evaluate_predictions
from nlp_trader.nlp.llm_annotations import (
    AnnotationRequest,
    AnnotationResponse,
    AnnotationVerification,
    AssetCandidate,
    CachedLocalLLMAnnotator,
    EntityAnnotation,
    GenerationRequest,
    GenerationResponse,
    LLMAnnotationConfig,
    build_annotation_request,
)
from nlp_trader.nlp.llm_decision_rounds import (
    CurrentSourceRetrieval,
    DecisionRound,
    DecisionRoundLedger,
    InferenceUsage,
    ModelIdentity,
    RawGeneration,
    SamplingSettings,
    VerifierCheck,
    VerifierResult,
    VersionedContract,
)
from nlp_trader.nlp.simple import SOURCE_CREDIBILITY, build_text_signals, link_entities
from nlp_trader.nlp.transformer import (
    CachedTransformerSentiment,
    TransformerSentimentConfig,
)
from nlp_trader.paper.ledger import PaperEventLedger
from nlp_trader.paper.simulator import PaperOrderIntent
from nlp_trader.portfolio.constraints import constraint_snapshot, round_trip_entry_constraints
from nlp_trader.portfolio.construction import construct_portfolio
from nlp_trader.portfolio.risk import conservative_risk_estimates, risk_estimate_flags
from nlp_trader.providers import (
    LocalFundamentalsProvider,
    LocalMarketDataProvider,
    LocalTextDataProvider,
)
from nlp_trader.reports import DEFAULT_LIMITATIONS, write_report
from nlp_trader.research import (
    RunContext,
    create_run_context,
    fail_run,
    finalize_run,
)
from nlp_trader.schemas import (
    Asset,
    CorporateAction,
    EarningsCalendarEvent,
    FundamentalRecord,
    MarketBar,
    TextItem,
    TextSignal,
)
from nlp_trader.timestamps import format_utc, parse_utc

LOGGER = logging.getLogger(__name__)

LLMGenerator = Callable[[list[GenerationRequest]], list[GenerationResponse]]

STAGES = (
    "ingest_market",
    "ingest_text",
    "annotate_text",
    "build_features",
    "build_labels",
    "train",
    "predict",
    "backtest",
    "paper",
    "report",
)

DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "ingest_market": (),
    "ingest_text": (),
    "annotate_text": ("ingest_market", "ingest_text"),
    "build_features": ("annotate_text",),
    "build_labels": ("ingest_market",),
    "train": ("build_features", "build_labels"),
    "predict": ("train",),
    "backtest": ("predict",),
    "paper": ("predict",),
    "report": ("backtest",),
}


def _write_json_once(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _write_text_once(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        if not value.endswith("\n"):
            handle.write("\n")
    return path


def _numeric_metric_deltas(
    baseline: dict[str, Any],
    enhanced: dict[str, Any],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for name in sorted(set(baseline) & set(enhanced)):
        left = baseline[name]
        right = enhanced[name]
        if isinstance(left, bool) or isinstance(right, bool):
            continue
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            continue
        left_value = float(left)
        right_value = float(right)
        if math.isfinite(left_value) and math.isfinite(right_value):
            deltas[name] = right_value - left_value
    return deltas


def _runtime_bound(value: str | None, *, end: bool) -> datetime | None:
    if value is None:
        return None
    if "T" in value:
        return parse_utc(value)
    parsed = date.fromisoformat(value)
    return datetime.combine(parsed, time.max if end else time.min, tzinfo=UTC)


def _signal_to_record(signal: TextSignal) -> dict[str, Any]:
    return {
        "item_id": signal.item_id,
        "asset_id": signal.asset_id,
        "symbol": signal.symbol,
        "asof_ts": format_utc(signal.asof_ts),
        "available_at": format_utc(signal.available_at) if signal.available_at else None,
        "sentiment_score": signal.sentiment_score,
        "sentiment_label": signal.sentiment_label,
        "sentiment_confidence": signal.sentiment_confidence,
        "relevance": signal.relevance,
        "novelty": signal.novelty,
        "source_credibility": signal.source_credibility,
        "model_version": signal.model_version,
        "source": signal.source,
        "source_type": signal.source_type,
        "author_hash": signal.author_hash,
        "duplicate_cluster_id": signal.duplicate_cluster_id,
        "event_type": signal.event_type,
        "spam_score": signal.spam_score,
        "disagreement": signal.disagreement,
        "llm_semantic_signal": signal.llm_semantic_signal,
        "llm_raw_confidence": signal.llm_raw_confidence,
        "llm_uncertainty": signal.llm_uncertainty,
        "llm_event_type": signal.llm_event_type,
        "llm_event_confidence": signal.llm_event_confidence,
        "llm_supporting_evidence_count": signal.llm_supporting_evidence_count,
        "llm_counterevidence_count": signal.llm_counterevidence_count,
        "llm_abstained": signal.llm_abstained,
    }


def _fundamental_to_record(record: FundamentalRecord) -> dict[str, Any]:
    return {
        "asset_id": record.asset_id,
        "symbol": record.symbol,
        "period_end": record.period_end.isoformat(),
        "available_at": format_utc(record.available_at),
        "values": dict(record.values),
        "filing_id": record.filing_id,
        "year": record.available_at.year,
    }


def _earnings_to_record(event: EarningsCalendarEvent) -> dict[str, Any]:
    return {
        "asset_id": event.asset_id,
        "symbol": event.symbol,
        "event_ts": format_utc(event.event_ts),
        "available_at": format_utc(event.available_at),
        "status": event.status,
        "year": event.event_ts.year,
    }


def _corporate_action_to_record(action: CorporateAction) -> dict[str, Any]:
    return {
        "asset_id": action.asset_id,
        "symbol": action.symbol,
        "event_ts": format_utc(action.event_ts),
        "available_at": format_utc(action.available_at),
        "action_type": action.action_type,
        "value": action.value,
        "year": action.event_ts.year,
    }


class PipelineExecution:
    """One immutable, dependency-aware local research execution."""

    def __init__(
        self,
        config: ResearchConfig,
        context: RunContext,
        *,
        llm_generator: LLMGenerator | None = None,
    ) -> None:
        self.config = config
        self.context = context
        self.enable_transformer_sentiment = config.transformer.enabled
        self.llm_generator = llm_generator
        self.outputs: dict[str, Any] = {"run_id": context.run_id}
        self.completed: set[str] = set()
        self.assets: list[Asset] = []
        self.bars: list[MarketBar] = []
        self.items: list[TextItem] = []
        self.llm_annotations: dict[tuple[str, str], EntityAnnotation] = {}
        self.llm_annotation_summary: dict[str, Any] = {}
        self.signals: list[TextSignal] = []
        self.fundamentals: list[FundamentalRecord] = []
        self.earnings_events: list[EarningsCalendarEvent] = []
        self.corporate_actions: list[CorporateAction] = []
        self.selected_decision_times: frozenset[datetime] | None = None
        self.market_decision_times_by_session: dict[date, datetime] = {}
        self.features: list[dict[str, Any]] = []
        self.labels: list[dict[str, Any]] = []
        self.model: dict[str, Any] = {}
        self.predictions: dict[str, list[dict[str, Any]]] = {}
        self.evaluation: dict[str, Any] = {}
        self.backtests: dict[str, dict[str, Any]] = {}
        self.final_holdout_backtests: dict[str, dict[str, Any]] = {}

    @property
    def symbols(self) -> tuple[str, ...] | None:
        return self.config.runtime.symbols or None

    @property
    def start(self) -> datetime | None:
        return _runtime_bound(self.config.runtime.start_date, end=False)

    @property
    def end(self) -> datetime | None:
        return _runtime_bound(self.config.runtime.end_date, end=True)

    def _runtime_context_calendar(self, *extra: datetime) -> USEquityCalendar:
        anchors = [value for value in (*extra, self.start, self.end) if value is not None]
        if not anchors:
            raise ValueError("runtime context calendar requires a start or end bound")
        dates = [value.date() for value in anchors]
        return USEquityCalendar(
            calendar_name=self.config.data.calendar,
            start=min(dates) - timedelta(days=730),
            end=max(dates) + timedelta(days=730),
        )

    @staticmethod
    def _nearby_decision_closes(
        calendar: USEquityCalendar, value: datetime
    ) -> tuple[datetime, ...]:
        local_date = value.astimezone(calendar.timezone).date()
        return calendar.decision_times(
            local_date - timedelta(days=10),
            local_date + timedelta(days=10),
        )

    def _market_fetch_start(self) -> datetime | None:
        if self.start is None:
            return None
        calendar = self._runtime_context_calendar()
        nearby_closes = self._nearby_decision_closes(calendar, self.start)
        if self.config.data.market_contract == "japan_cash_equity_v1":
            first_decision = max(value for value in nearby_closes if value <= self.start)
        else:
            first_decision = min(value for value in nearby_closes if value >= self.start)
        session_date = first_decision.astimezone(calendar.timezone).date()
        for _ in range(self.config.features.market_warmup_sessions):
            session_date = calendar.previous_session(session_date)
        return calendar.session_close(session_date)

    def _market_fetch_end(self, decision_end: datetime | None = None) -> datetime | None:
        anchor = decision_end or self.end
        if anchor is None:
            return None
        calendar = self._runtime_context_calendar(anchor)
        last_decision = max(
            value for value in self._nearby_decision_closes(calendar, anchor) if value <= anchor
        )
        session_date = last_decision.astimezone(calendar.timezone).date()
        future_sessions = self.config.features.horizon_days
        if self.config.data.market_contract == "japan_cash_equity_v1":
            future_sessions += 1
        for _ in range(future_sessions):
            session_date = calendar.next_session(session_date)
        return calendar.session_close(session_date)

    def _text_fetch_start(self) -> datetime | None:
        if self.start is None:
            return None
        return self.start - timedelta(days=self.config.features.text_warmup_days)

    def _decision_data_end(self) -> datetime | None:
        if self.selected_decision_times:
            return max(self.selected_decision_times)
        return self.end

    def _decision_data_start(self) -> datetime | None:
        if self.selected_decision_times:
            return min(self.selected_decision_times)
        return self.start

    def _known_event_fetch_end(self) -> datetime | None:
        decision_end = self._decision_data_end()
        if decision_end is None:
            return None
        return decision_end + timedelta(days=self.config.features.event_lookahead_days)

    def _is_decision_timestamp(self, value: str | datetime) -> bool:
        timestamp = parse_utc(value) if isinstance(value, str) else value.astimezone(UTC)
        in_window = (self.start is None or timestamp >= self.start) and (
            self.end is None or timestamp <= self.end
        )
        return in_window and (
            self.selected_decision_times is None or timestamp in self.selected_decision_times
        )

    def _decision_bars(self) -> list[MarketBar]:
        if not self.bars:
            return []
        calendar = self._calendar()
        if not self.market_decision_times_by_session:
            self.market_decision_times_by_session = market_decision_times_by_session(
                self.bars,
                calendar.timezone,
            )
        return [
            bar
            for bar in self.bars
            if self._is_decision_timestamp(
                market_decision_time_for_bar(
                    bar,
                    self.market_decision_times_by_session,
                    calendar.timezone,
                )
            )
        ]

    def _assets_by_id(self) -> dict[str, Asset]:
        indexed: dict[str, Asset] = {}
        for asset in self.assets:
            if asset.asset_id in indexed:
                raise ValueError(f"duplicate asset_id in asset master: {asset.asset_id}")
            indexed[asset.asset_id] = asset
        return indexed

    @staticmethod
    def _validate_asset_reference(
        asset_id: str,
        symbol: str | None,
        assets_by_id: dict[str, Asset],
        *,
        role: str,
    ) -> Asset:
        asset = assets_by_id.get(asset_id)
        if asset is None:
            raise ValueError(f"{role} references unknown asset_id: {asset_id}")
        if symbol is not None and symbol != asset.symbol:
            raise ValueError(
                f"{role} symbol {symbol} does not match asset master {asset.symbol} for {asset_id}"
            )
        return asset

    def run(self, stage: str) -> None:
        if stage not in DEPENDENCIES:
            raise ValueError(f"unknown pipeline stage: {stage}")
        if stage in self.completed:
            return
        for dependency in DEPENDENCIES[stage]:
            self.run(dependency)
        started = clock.monotonic()
        LOGGER.info("stage_start run_id=%s stage=%s", self.context.run_id, stage)
        getattr(self, stage)()
        self.completed.add(stage)
        LOGGER.info(
            "stage_complete run_id=%s stage=%s elapsed_seconds=%.3f assets=%d bars=%d "
            "text_items=%d features=%d labels=%d",
            self.context.run_id,
            stage,
            clock.monotonic() - started,
            len(self.assets),
            len(self.bars),
            len(self.items),
            len(self.features),
            len(self.labels),
        )

    def _ensure_assets(self) -> list[dict[str, Any]]:
        if self.assets:
            return []
        references = self._ingest_raw(
            "asset_master",
            self.config.paths.assets,
            self.config.data.market_license_or_terms_ref,
        )
        provider = LocalMarketDataProvider(
            self._captured_input_path("assets", self.config.paths.assets, references),
            self.config.paths.market_bars,
            self.config.paths.corporate_actions,
            market_contract=self.config.data.market_contract,
        )
        self.assets = provider.fetch_assets(symbols=self.symbols)
        if not self.assets:
            raise ValueError("the configured asset universe is empty after runtime filters")
        return references

    def _ingest_raw(self, role: str, path: Path, license_ref: str) -> list[dict[str, Any]]:
        store = ContentAddressedRawStore(self.config.paths.raw_dir)
        source_paths = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
        if not source_paths:
            raise ValueError(f"configured {role} input contains no files: {path}")
        references: list[dict[str, Any]] = []
        for index, source_path in enumerate(source_paths):
            relative = (
                source_path.name if path.is_file() else source_path.relative_to(path).as_posix()
            )
            artifact = store.ingest_file(
                source_path,
                RawIngestionRequest(
                    source=role,
                    vendor="local-file",
                    license_or_terms_ref=license_ref,
                    ingested_at=self.context.created_at,
                    request_id=f"{self.context.run_id}-{role}-{index:06d}",
                    schema_version=self.config.data.schema_version,
                    fetch_params={
                        "symbols": list(self.config.runtime.symbols),
                        "start_date": self.config.runtime.start_date,
                        "end_date": self.config.runtime.end_date,
                        "limit": self.config.runtime.limit,
                        "market_contract": self.config.data.market_contract,
                        "input_relative_path": relative,
                    },
                ),
            )
            references.append(
                {
                    "role": role,
                    "input_relative_path": relative,
                    "payload_path": str(artifact.payload_path),
                    "metadata_path": str(artifact.metadata_path),
                    "sha256": artifact.metadata.sha256,
                }
            )
        manifest_role = "assets" if role == "asset_master" else role
        captured_manifest = next(
            (entry for entry in self.context.inputs if entry["role"] == manifest_role),
            None,
        )
        if captured_manifest is None:
            raise ValueError(f"run input manifest is missing role: {manifest_role}")
        if path.is_file():
            if len(references) != 1 or references[0]["sha256"] != captured_manifest.get("sha256"):
                raise ValueError(f"configured {manifest_role} changed after run capture")
        else:
            expected = {
                str(entry["relative_path"]): str(entry["sha256"])
                for entry in captured_manifest.get("files", [])
            }
            observed = {
                str(reference["input_relative_path"]): str(reference["sha256"])
                for reference in references
            }
            if observed != expected:
                raise ValueError(f"configured {manifest_role} directory changed after run capture")
        return references

    def _captured_input_path(
        self,
        role: str,
        configured_path: Path,
        references: list[dict[str, Any]],
    ) -> Path:
        """Expose immutable bronze bytes to providers without rereading mutable inputs."""

        if configured_path.is_file():
            if len(references) != 1:
                raise ValueError(f"captured file input {role} must have one raw artifact")
            return Path(str(references[0]["payload_path"]))
        root = self.context.paths.interim / "captured_inputs" / role
        for reference in references:
            destination = root / str(reference["input_relative_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(Path(str(reference["payload_path"])), destination)
        return root

    def ingest_market(self) -> None:
        asset_refs = self._ingest_raw(
            "asset_master",
            self.config.paths.assets,
            self.config.data.market_license_or_terms_ref,
        )
        bar_refs = self._ingest_raw(
            "market_bars",
            self.config.paths.market_bars,
            self.config.data.market_license_or_terms_ref,
        )
        raw_refs = [*asset_refs, *bar_refs]
        captured_optional: dict[str, Path] = {}
        optional_inputs = (
            (
                "corporate_actions",
                self.config.paths.corporate_actions,
                self.config.data.market_license_or_terms_ref,
            ),
            (
                "fundamentals",
                self.config.paths.fundamentals,
                self.config.data.market_license_or_terms_ref,
            ),
            (
                "earnings_calendar",
                self.config.paths.earnings_calendar,
                self.config.data.market_license_or_terms_ref,
            ),
        )
        for role, configured_path, license_ref in optional_inputs:
            if configured_path is None:
                continue
            references = self._ingest_raw(role, configured_path, license_ref)
            raw_refs.extend(references)
            captured_optional[role] = self._captured_input_path(role, configured_path, references)
        raw_path = _write_json_once(
            self.context.paths.interim / "bronze_refs" / "market.json",
            raw_refs,
        )
        provider = LocalMarketDataProvider(
            self._captured_input_path("assets", self.config.paths.assets, asset_refs),
            self._captured_input_path("market_bars", self.config.paths.market_bars, bar_refs),
            captured_optional.get("corporate_actions"),
            market_contract=self.config.data.market_contract,
        )
        if self.config.runtime.limit is not None:
            selected = provider.fetch_decision_times(
                symbols=self.symbols,
                start=self.start,
                end=self.end,
                bar_size="1d",
                limit=self.config.runtime.limit,
            )
            self.selected_decision_times = frozenset(selected)
        self.assets = provider.fetch_assets(symbols=self.symbols)
        self.bars = provider.fetch_bars(
            symbols=self.symbols,
            start=self._market_fetch_start(),
            end=self._market_fetch_end(self._decision_data_end()),
            bar_size="1d",
            limit=None,
        )
        self.bars.sort(key=lambda bar: (bar.ts, bar.symbol))
        market_calendar = self._calendar()
        self.market_decision_times_by_session = market_decision_times_by_session(
            self.bars,
            market_calendar.timezone,
        )
        self.corporate_actions = provider.fetch_corporate_actions(
            symbols=self.symbols,
            start=self._decision_data_start(),
            # Future-dated actions can be valid point-in-time features when they
            # were announced before a decision.  Feature construction enforces
            # available_at <= asof_ts before using them.
            end=self._known_event_fetch_end(),
        )
        fundamentals_provider = LocalFundamentalsProvider(
            captured_optional.get("fundamentals"),
            captured_optional.get("earnings_calendar"),
        )
        self.fundamentals = fundamentals_provider.fetch_fundamentals(
            symbols=self.symbols,
            # A filing available before the requested research window remains
            # the latest known filing at the first decision in that window.
            start=None,
            end=self._decision_data_end(),
        )
        self.earnings_events = fundamentals_provider.fetch_earnings_calendar(
            symbols=self.symbols,
            start=self._decision_data_start(),
            # Keep known future events beyond the last market bar so proximity
            # features at the end of the requested window are not truncated.
            end=self._known_event_fetch_end(),
        )
        if not self.assets:
            raise ValueError("the configured asset universe is empty after runtime filters")
        if not self._decision_bars():
            raise ValueError("no market bars remain after runtime filters")
        assets_by_id = self._assets_by_id()
        calendar = self._calendar()
        for bar in self.bars:
            asset = self._validate_asset_reference(
                bar.asset_id, bar.symbol, assets_by_id, role="market bar"
            )
            session_date = bar.ts.astimezone(calendar.timezone).date()
            if asset.active_from is not None and session_date < asset.active_from:
                raise ValueError(f"market bar for {bar.asset_id} predates active_from")
            if asset.active_to is not None and session_date > asset.active_to:
                raise ValueError(f"market bar for {bar.asset_id} is after active_to")
        for role, records in (
            ("fundamental record", self.fundamentals),
            ("earnings event", self.earnings_events),
            ("corporate action", self.corporate_actions),
        ):
            for record in records:
                self._validate_asset_reference(
                    record.asset_id,
                    record.symbol,
                    assets_by_id,
                    role=role,
                )

        asset_records = [asset_to_record(asset) for asset in self.assets]
        asset_paths = write_partitioned_parquet(
            asset_records,
            self.context.paths.interim / "silver" / "assets",
            compression=self.config.data.compression,
            max_rows_per_file=self.config.data.write_batch_rows,
        )
        bar_records: list[dict[str, Any]] = []
        for bar in self.bars:
            bar_record = market_bar_to_record(bar)
            bar_record["year"] = bar.ts.year
            bar_records.append(bar_record)
        bar_paths = write_partitioned_parquet(
            bar_records,
            self.context.paths.interim / "silver" / "market",
            partition_fields=("bar_size", "symbol", "year"),
            compression=self.config.data.compression,
            max_rows_per_file=self.config.data.write_batch_rows,
        )
        fundamental_paths = write_partitioned_parquet(
            [_fundamental_to_record(record) for record in self.fundamentals],
            self.context.paths.interim / "silver" / "fundamentals",
            partition_fields=("symbol", "year"),
            compression=self.config.data.compression,
            max_rows_per_file=self.config.data.write_batch_rows,
        )
        earnings_paths = write_partitioned_parquet(
            [_earnings_to_record(event) for event in self.earnings_events],
            self.context.paths.interim / "silver" / "earnings_calendar",
            partition_fields=("symbol", "year"),
            compression=self.config.data.compression,
            max_rows_per_file=self.config.data.write_batch_rows,
        )
        action_paths = write_partitioned_parquet(
            [_corporate_action_to_record(action) for action in self.corporate_actions],
            self.context.paths.interim / "silver" / "corporate_actions",
            partition_fields=("symbol", "year"),
            compression=self.config.data.compression,
            max_rows_per_file=self.config.data.write_batch_rows,
        )
        self.outputs.update(
            {
                "raw_market_manifest": raw_path,
                "assets": [str(path) for path in asset_paths],
                "market": [str(path) for path in bar_paths],
                "fundamentals": [str(path) for path in fundamental_paths],
                "earnings_calendar": [str(path) for path in earnings_paths],
                "corporate_actions": [str(path) for path in action_paths],
            }
        )

    def ingest_text(self) -> None:
        asset_refs = self._ensure_assets()
        text_refs = self._ingest_raw(
            "text_items",
            self.config.paths.text_items,
            self.config.data.text_license_or_terms_ref,
        )
        provider = LocalTextDataProvider(
            self._captured_input_path("text_items", self.config.paths.text_items, text_refs)
        )
        selected = provider.fetch_items(
            start=self._text_fetch_start(),
            end=self._decision_data_end(),
            limit=None,
        )
        assets_by_id = self._assets_by_id()
        item_dates = [item.available_at.date() for item in selected]
        calendar = USEquityCalendar(
            calendar_name=self.config.data.calendar,
            start=min(item_dates) - timedelta(days=14)
            if item_dates
            else self.context.created_at.date() - timedelta(days=14),
            end=max(item_dates) + timedelta(days=14)
            if item_dates
            else self.context.created_at.date() + timedelta(days=14),
        )
        normalized: list[TextItem] = []
        for item in selected:
            for entity in item.entities:
                if entity.asset_id is None:
                    continue
                asset = self._validate_asset_reference(
                    entity.asset_id,
                    entity.symbol,
                    assets_by_id,
                    role="text entity",
                )
                item_date = item.available_at.astimezone(calendar.timezone).date()
                if asset.active_from is not None and item_date < asset.active_from:
                    raise ValueError(f"text entity for {asset.asset_id} predates active_from")
                if asset.active_to is not None and item_date > asset.active_to:
                    raise ValueError(f"text entity for {asset.asset_id} is after active_to")
            normalized.append(
                item if item.entities else replace(item, entities=link_entities(item, self.assets))
            )
        self.items = normalized
        raw_path = _write_json_once(
            self.context.paths.interim / "bronze_refs" / "text.json",
            [*asset_refs, *text_refs],
        )
        text_records: list[dict[str, Any]] = []
        for item in self.items:
            record = text_item_to_record(item)
            record["date"] = item.available_at.date().isoformat()
            text_records.append(record)
        text_paths = write_partitioned_parquet(
            text_records,
            self.context.paths.interim / "silver" / "text",
            partition_fields=("source", "date"),
            compression=self.config.data.compression,
            max_rows_per_file=self.config.data.write_batch_rows,
        )
        self.outputs.update(
            {
                "raw_text_manifest": raw_path,
                "text": [str(path) for path in text_paths],
            }
        )

    def annotate_text(self) -> None:
        configured = self.config.llm_annotations
        if not configured.enabled:
            return
        model_path = self.config.paths.llm_model
        if (
            model_path is None
            or not configured.model_id
            or not configured.model_revision
            or not configured.model_license_or_terms_ref
        ):
            raise ValueError("enabled LLM annotations require a local model path and identity")

        annotator = CachedLocalLLMAnnotator(
            LLMAnnotationConfig(
                model_path=model_path,
                model_id=configured.model_id,
                model_revision=configured.model_revision,
                model_license_or_terms_ref=configured.model_license_or_terms_ref,
                prompt_version=configured.prompt_version,
                schema_version=configured.schema_version,
                verifier_version=configured.verifier_version,
                cache_dir=self.config.paths.models_dir / "_cache" / "llm_annotations",
                attempt_dir=(self.context.paths.models / "llm_annotations" / "generation_attempts"),
                batch_size=configured.batch_size,
                max_input_tokens=configured.max_input_tokens,
                max_new_tokens=configured.max_new_tokens,
                decoding=configured.decoding,
                seed=configured.seed,
                input_cost_per_million_tokens_usd=(configured.input_cost_per_million_tokens_usd),
                output_cost_per_million_tokens_usd=(configured.output_cost_per_million_tokens_usd),
                local_files_only=configured.local_files_only,
                trust_remote_code=configured.trust_remote_code,
            ),
            generator=self.llm_generator,
        )
        model_manifest = next(
            (entry for entry in self.context.inputs if entry["role"] == "llm_model"),
            None,
        )
        if model_manifest is None or not model_manifest.get("sha256"):
            raise ValueError("run input manifest is missing the enabled local LLM model hash")
        loaded_model_hash = annotator.provenance_payload.get("model_directory_sha256")
        if loaded_model_hash != model_manifest["sha256"]:
            raise ValueError("configured local LLM model changed after run input capture")
        assets_by_id = self._assets_by_id()
        calendar = self._calendar()
        market_decision_times = sorted(self.market_decision_times_by_session.values())
        requests: list[AnnotationRequest] = []
        expected_assets: dict[str, tuple[str, ...]] = {}
        for item in sorted(self.items, key=lambda value: (value.available_at, value.item_id)):
            item_date = item.available_at.astimezone(calendar.timezone).date()
            candidate_assets: set[str] = set()
            for entity in item.entities:
                if entity.asset_id is None:
                    continue
                asset = assets_by_id.get(entity.asset_id)
                if asset is None:
                    continue
                if asset.active_from is not None and item_date < asset.active_from:
                    continue
                if asset.active_to is not None and item_date > asset.active_to:
                    continue
                candidate_assets.add(entity.asset_id)
            if not candidate_assets:
                continue
            candidates = tuple(
                AssetCandidate(
                    asset_id=asset_id,
                    symbol=assets_by_id[asset_id].symbol,
                    name=assets_by_id[asset_id].name,
                )
                for asset_id in sorted(candidate_assets)
            )
            decision_index = bisect_left(
                market_decision_times,
                item.available_at.astimezone(UTC),
            )
            decision_time = (
                market_decision_times[decision_index]
                if decision_index < len(market_decision_times)
                else calendar.next_decision_time(item.available_at)
            )
            request = build_annotation_request(
                item,
                candidates,
                decision_time=decision_time,
                target_horizon_days=self.config.features.horizon_days,
                source_quality=SOURCE_CREDIBILITY.get(item.source_type, 0.5),
            )
            requests.append(request)
            expected_assets[item.item_id] = tuple(candidate.asset_id for candidate in candidates)

        responses = annotator.annotate(requests)
        responses_by_item: dict[str, AnnotationResponse] = {}
        for response in responses:
            if response.item_id in responses_by_item:
                raise ValueError(f"duplicate LLM annotation response for item: {response.item_id}")
            expected = expected_assets.get(response.item_id)
            if expected is None:
                raise ValueError(
                    f"LLM annotation response references unknown item: {response.item_id}"
                )
            observed = tuple(annotation.asset_id for annotation in response.annotations)
            if len(observed) != len(set(observed)) or set(observed) != set(expected):
                raise ValueError(
                    f"LLM annotation assets do not match candidates for item: {response.item_id}"
                )
            responses_by_item[response.item_id] = response
        if set(responses_by_item) != set(expected_assets):
            missing = sorted(set(expected_assets) - set(responses_by_item))
            raise ValueError("LLM annotation responses missing items: " + ", ".join(missing))

        item_by_id = {item.item_id: item for item in self.items}
        artifact_root = self.context.paths.models / "llm_annotations"
        prompt_path = _write_text_once(artifact_root / "prompt.txt", annotator.prompt_text)
        schema_path = _write_json_once(artifact_root / "schema.json", annotator.schema_payload)
        annotation_stage_completed_at = datetime.now(UTC)
        provenance = dict(annotator.provenance_payload)
        provenance.update(
            {
                "run_id": self.context.run_id,
                "model_directory_sha256": model_manifest["sha256"],
                "model_license_or_terms_ref": configured.model_license_or_terms_ref,
                "feature_mode": configured.feature_mode,
                "retrospective_parser": True,
                "source_availability_policy": (
                    "annotations inherit each source item's available_at; annotation stage "
                    "completion time is audit metadata and is not historical feature availability"
                ),
                "annotation_stage_completed_at": format_utc(annotation_stage_completed_at),
                "cache_hit_count": annotator.cache_hit_count,
                "generation_request_count": annotator.generation_request_count,
                "deduplicated_request_count": annotator.deduplicated_request_count,
                "generation_attempt_count": len(annotator.attempt_paths),
            }
        )
        provenance_path = _write_json_once(
            artifact_root / "provenance.json",
            provenance,
        )

        requests_by_cache_key: dict[str, list[AnnotationRequest]] = {}
        for request in requests:
            requests_by_cache_key.setdefault(annotator.cache_key_for(request), []).append(request)
        response_paths: list[str] = []
        cache_records_by_key: dict[str, dict[str, Any]] = {}
        for cache_key, matching_requests in sorted(requests_by_cache_key.items()):
            cache_record = annotator.cache_record_for(matching_requests[0])
            cache_records_by_key[cache_key] = cache_record
            cache_record["run_request"] = {
                "item_ids": [request.item_id for request in matching_requests]
            }
            response_path = _write_json_once(
                artifact_root / "responses" / f"{cache_key}.json",
                cache_record,
            )
            response_paths.append(str(response_path))

        annotation_records: list[dict[str, Any]] = []
        verifications: dict[str, AnnotationVerification] = {}
        for request in requests:
            response = responses_by_item[request.item_id]
            verification = annotator.verification_for(request, response)
            verifications[request.item_id] = verification
            item = item_by_id[request.item_id]
            for annotation in response.annotations:
                asset = assets_by_id[annotation.asset_id]
                key = (request.item_id, annotation.asset_id)
                if key in self.llm_annotations:
                    raise ValueError(
                        "duplicate LLM annotation for item and asset: "
                        f"{request.item_id}, {annotation.asset_id}"
                    )
                self.llm_annotations[key] = annotation
                annotation_records.append(
                    {
                        "item_id": request.item_id,
                        "asset_id": annotation.asset_id,
                        "symbol": asset.symbol,
                        "available_at": format_utc(item.available_at),
                        "decision_time": format_utc(request.decision_time),
                        "stance_label": annotation.stance_label,
                        "semantic_signal": annotation.semantic_signal,
                        "raw_confidence": annotation.raw_confidence,
                        "uncertainty": annotation.uncertainty,
                        "horizon_days": annotation.horizon_days,
                        "primary_event_type": annotation.primary_event_type,
                        "event_confidence": annotation.event_confidence,
                        "supporting_evidence_span_ids": (annotation.supporting_evidence_span_ids),
                        "counterevidence_span_ids": annotation.counterevidence_span_ids,
                        "mechanism": annotation.mechanism,
                        "invalidation_conditions": annotation.invalidation_conditions,
                        "abstain_reason": annotation.abstain_reason,
                        "source_type": request.source_type,
                        "source_quality": request.source_quality,
                        "model_id": configured.model_id,
                        "model_revision": configured.model_revision,
                        "prompt_version": configured.prompt_version,
                        "schema_version": configured.schema_version,
                        "verifier_version": configured.verifier_version,
                        "verification_valid": verification.valid,
                        "model_directory_sha256": model_manifest["sha256"],
                        "retrospective_parser": True,
                        "year": item.available_at.year,
                    }
                )

        decision_rounds: list[DecisionRound] = []
        input_snapshot_hash = hashlib.sha256(
            json.dumps(
                [
                    {
                        "role": entry["role"],
                        "sha256": entry.get("sha256"),
                        "exists": entry["exists"],
                    }
                    for entry in self.context.inputs
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for request in requests:
            cache_key = annotator.cache_key_for(request)
            cache_record = cache_records_by_key[cache_key]
            generation = cache_record["generation"]
            if not isinstance(generation, dict):
                raise ValueError("LLM cache generation metadata must be an object")
            response = responses_by_item[request.item_id]
            verification = verifications[request.item_id]
            inference_source = annotator.inference_source_for(request)
            generated_in_this_round = inference_source == "generated"
            token_rates_configured = (
                configured.input_cost_per_million_tokens_usd is not None
                and configured.output_cost_per_million_tokens_usd is not None
            )
            raw_latency = generation.get("generation_latency_seconds")
            decision_rounds.append(
                DecisionRound(
                    run_id=self.context.run_id,
                    config_hash=self.config.content_hash(),
                    input_snapshot_hash=input_snapshot_hash,
                    item_id=request.item_id,
                    source_text_hash=request.source_text_hash,
                    source_available_at=request.source_available_at,
                    decision_time=request.decision_time,
                    horizon_days=request.target_horizon_days,
                    model=ModelIdentity(
                        provider=configured.backend,
                        model_id=configured.model_id,
                        revision=configured.model_revision,
                        sha256=str(model_manifest["sha256"]),
                    ),
                    prompt=VersionedContract(
                        version=configured.prompt_version,
                        sha256=str(provenance["prompt_sha256"]),
                    ),
                    schema_contract=VersionedContract(
                        version=configured.schema_version,
                        sha256=str(provenance["schema_sha256"]),
                    ),
                    sampling=SamplingSettings(
                        decoding=configured.decoding,
                        seed=configured.seed,
                        max_input_tokens=configured.max_input_tokens,
                        max_new_tokens=configured.max_new_tokens,
                    ),
                    retrieval=CurrentSourceRetrieval(
                        evidence_ids=tuple(span.span_id for span in request.evidence_spans)
                    ),
                    raw_generation=RawGeneration(
                        request_id=str(generation["request_id"]),
                        generated_text=generation.get("generated_text"),
                        input_too_long=bool(generation.get("input_too_long", False)),
                        output_truncated=bool(generation.get("output_truncated", False)),
                        metadata={
                            "original_input_token_count": generation.get("input_token_count"),
                            "original_output_token_count": generation.get("output_token_count"),
                            "original_generation_latency_seconds": raw_latency,
                            "original_estimated_cost_usd": generation.get("estimated_cost_usd"),
                        },
                    ),
                    structured_output=response.to_dict(),
                    verifier=VerifierResult(
                        version=configured.verifier_version,
                        passed=verification.valid,
                        checks=tuple(
                            VerifierCheck(
                                check_id=check.name,
                                passed=check.passed,
                                detail=check.detail,
                            )
                            for check in verification.checks
                        ),
                    ),
                    inference_source=inference_source,
                    usage=InferenceUsage(
                        input_tokens=(
                            generation.get("input_token_count") if generated_in_this_round else 0
                        ),
                        output_tokens=(
                            generation.get("output_token_count") if generated_in_this_round else 0
                        ),
                        latency_ms=(
                            float(raw_latency) * 1000.0
                            if generated_in_this_round and raw_latency is not None
                            else 0.0
                        ),
                        estimated_usd_cost=(
                            generation.get("estimated_cost_usd")
                            if generated_in_this_round
                            else 0.0
                            if token_rates_configured
                            else None
                        ),
                    ),
                    application_mode=configured.feature_mode,
                )
            )
        decision_ledger_path = self.context.paths.models / "llm_decisions" / "rounds.jsonl"
        decision_ledger = DecisionRoundLedger(decision_ledger_path)
        decision_ledger.write_exclusive(decision_rounds)
        replayed_rounds = decision_ledger.replay_and_verify()
        if len(replayed_rounds) != len(decision_rounds):
            raise ValueError("LLM decision-round replay coverage mismatch")

        silver_paths = write_partitioned_parquet(
            annotation_records,
            self.context.paths.processed / "silver" / "llm_annotations",
            partition_fields=("symbol", "year"),
            compression=self.config.data.compression,
            max_rows_per_file=self.config.data.write_batch_rows,
        )
        abstentions = sum(record["stance_label"] == "abstain" for record in annotation_records)
        cited_annotations = sum(
            bool(record["supporting_evidence_span_ids"]) for record in annotation_records
        )
        counterevidence_annotations = sum(
            bool(record["counterevidence_span_ids"]) for record in annotation_records
        )
        verification_checks = [
            check for verification in verifications.values() for check in verification.checks
        ]
        verification_summary = {
            "artifact_schema_version": "llm-annotation-verification-summary-v1",
            "verifier_version": configured.verifier_version,
            "request_count": len(requests),
            "valid_request_count": sum(value.valid for value in verifications.values()),
            "invalid_request_count": sum(not value.valid for value in verifications.values()),
            "check_counts": {
                name: {
                    "passed": sum(
                        check.passed for check in verification_checks if check.name == name
                    ),
                    "failed": sum(
                        not check.passed for check in verification_checks if check.name == name
                    ),
                }
                for name in sorted({check.name for check in verification_checks})
            },
            "scope_note": (
                "deterministic checks validate identity, timing, horizon, evidence references, "
                "and cited numeric tokens; they do not prove that prose claims are semantically "
                "true"
            ),
        }
        verification_summary_path = _write_json_once(
            self.context.paths.processed / "evaluation" / "llm_verification_summary.json",
            verification_summary,
        )
        self.llm_annotation_summary = {
            "artifact_schema_version": "llm-semantic-signal-summary-v2",
            "enabled": True,
            "feature_mode": configured.feature_mode,
            "retrospective_parser": True,
            "text_item_count": len(self.items),
            "request_count": len(requests),
            "annotation_count": len(annotation_records),
            "non_abstention_count": len(annotation_records) - abstentions,
            "abstention_count": abstentions,
            "abstention_rate": (
                abstentions / len(annotation_records) if annotation_records else 0.0
            ),
            "cited_annotation_count": cited_annotations,
            "counterevidence_annotation_count": counterevidence_annotations,
            "event_annotation_count": sum(
                record["primary_event_type"] is not None for record in annotation_records
            ),
            "invalid_response_count": 0,
            "cache_hit_count": annotator.cache_hit_count,
            "generation_request_count": annotator.generation_request_count,
            "deduplicated_request_count": annotator.deduplicated_request_count,
            "generation_attempt_count": len(annotator.attempt_paths),
            "generated_input_token_count": annotator.generated_input_token_count,
            "generated_output_token_count": annotator.generated_output_token_count,
            "generation_latency_seconds": annotator.generation_latency_seconds,
            "estimated_inference_cost_usd": annotator.estimated_inference_cost_usd,
            "model_id": configured.model_id,
            "model_revision": configured.model_revision,
            "prompt_version": configured.prompt_version,
            "schema_version": configured.schema_version,
            "verifier_version": configured.verifier_version,
            "model_directory_sha256": model_manifest["sha256"],
        }
        summary_path = _write_json_once(
            self.context.paths.processed / "evaluation" / "llm_annotation_summary.json",
            self.llm_annotation_summary,
        )
        self.outputs.update(
            {
                "llm_annotation_prompt": prompt_path,
                "llm_annotation_schema": schema_path,
                "llm_annotation_provenance": provenance_path,
                "llm_annotation_responses": response_paths,
                "llm_annotation_generation_attempts": [
                    str(path) for path in annotator.attempt_paths
                ],
                "llm_annotations": [str(path) for path in silver_paths],
                "llm_annotation_summary": summary_path,
                "llm_verification_summary": verification_summary_path,
                "llm_decision_rounds": decision_ledger_path,
            }
        )

    def _calendar(self) -> USEquityCalendar:
        dates = [bar.ts.date() for bar in self.bars]
        dates.extend(market_bar_available_at(bar).date() for bar in self.bars)
        dates.extend(item.available_at.date() for item in self.items)
        if not dates:
            today = self.context.created_at.date()
            dates = [today]
        return USEquityCalendar(
            calendar_name=self.config.data.calendar,
            start=min(dates) - timedelta(days=14),
            end=max(dates) + timedelta(days=14),
        )

    def _transformer_results(self) -> dict[str, tuple[float, str, float]]:
        if not self.enable_transformer_sentiment:
            return {}
        transformer = self.config.transformer
        if not transformer.model_name:
            raise ValueError(
                "transformer sentiment was enabled but transformer.model_name is not configured"
            )
        engine = CachedTransformerSentiment(
            TransformerSentimentConfig(
                model_name=transformer.model_name,
                model_version=transformer.model_version,
                cache_dir=self.config.paths.models_dir / "_cache" / "transformer",
                batch_size=transformer.batch_size,
                max_sequence_length=transformer.max_sequence_length,
                local_files_only=transformer.local_files_only,
            )
        )
        values = [f"{item.title or ''}\n{item.body or ''}" for item in self.items]
        results = engine.predict(values)
        return {
            item.item_id: (result.score, result.label, result.confidence)
            for item, result in zip(self.items, results, strict=True)
        }

    def build_features(self) -> None:
        calendar = self._calendar()
        market_decision_times = sorted(self.market_decision_times_by_session.values())
        transformer_results = self._transformer_results()
        raw_signals = build_text_signals(self.items, self.assets)
        self.signals = []
        for signal in raw_signals:
            decision_index = bisect_left(
                market_decision_times,
                signal.asof_ts.astimezone(UTC),
            )
            decision_time = (
                market_decision_times[decision_index]
                if decision_index < len(market_decision_times)
                else calendar.next_decision_time(signal.asof_ts)
            )
            transformer_result = transformer_results.get(signal.item_id)
            llm_annotation = self.llm_annotations.get((signal.item_id, signal.asset_id))
            overrides: dict[str, Any] = {"asof_ts": decision_time}
            if transformer_result is not None:
                score, label, confidence = transformer_result
                overrides.update(
                    {
                        "sentiment_score": score,
                        "sentiment_label": label,
                        "sentiment_confidence": confidence,
                        "model_version": self.config.transformer.model_version,
                    }
                )
            if self.config.llm_annotations.feature_mode == "augment" and llm_annotation is not None:
                overrides.update(
                    {
                        "llm_semantic_signal": llm_annotation.semantic_signal,
                        "llm_raw_confidence": llm_annotation.raw_confidence,
                        "llm_uncertainty": llm_annotation.uncertainty,
                        "llm_event_type": llm_annotation.primary_event_type,
                        "llm_event_confidence": llm_annotation.event_confidence,
                        "llm_supporting_evidence_count": len(
                            llm_annotation.supporting_evidence_span_ids
                        ),
                        "llm_counterevidence_count": len(llm_annotation.counterevidence_span_ids),
                        "llm_abstained": llm_annotation.stance_label == "abstain",
                    }
                )
            self.signals.append(replace(signal, **overrides))
        feature_context = build_feature_rows(
            self.bars,
            self.signals,
            self.config,
            self.assets,
            fundamentals=self.fundamentals,
            earnings_events=self.earnings_events,
            corporate_actions=self.corporate_actions,
        )
        self.features = [
            row for row in feature_context if self._is_decision_timestamp(str(row["asof_ts"]))
        ]
        for row in self.features:
            row["beta"] = row.get("market_beta_60d")
            row["volatility"] = row.get("realized_volatility_20d")
            row.update(conservative_risk_estimates(row, self.config.backtest))
            row.setdefault("short_available", False)
            row.setdefault("hard_to_borrow", False)
        signal_records: list[dict[str, Any]] = []
        for signal in self.signals:
            record = _signal_to_record(signal)
            record["year"] = signal.asof_ts.year
            signal_records.append(record)
        signal_paths = write_partitioned_parquet(
            signal_records,
            self.context.paths.processed / "silver" / "text_signals",
            partition_fields=("symbol", "year"),
            compression=self.config.data.compression,
            max_rows_per_file=self.config.data.write_batch_rows,
        )
        store = ParquetFeatureStore(
            self.context.paths.processed / "gold" / "features",
            compression=self.config.data.compression,
        )
        store.write_features(self.features)
        self.outputs.update(
            {
                "signals": [str(path) for path in signal_paths],
                "features": str(self.context.paths.processed / "gold" / "features"),
            }
        )

    def build_labels(self) -> None:
        label_context = build_label_rows(self.bars, self.config, self.assets)
        self.labels = [
            row for row in label_context if self._is_decision_timestamp(str(row["asof_ts"]))
        ]
        records: list[dict[str, Any]] = []
        for row in self.labels:
            record = dict(row)
            record["year"] = parse_utc(str(row["asof_ts"])).year
            records.append(record)
        paths = write_partitioned_parquet(
            records,
            self.context.paths.processed / "gold" / "labels",
            partition_fields=("label_version", "year"),
            compression=self.config.data.compression,
            max_rows_per_file=self.config.data.write_batch_rows,
        )
        self.outputs["labels"] = [str(path) for path in paths]

    def train(self) -> None:
        self.model = train_baselines(
            self.features,
            self.labels,
            model_version=self.config.features.model_version,
            min_train_rows=self.config.models.min_train_rows,
            embargo_periods=self.config.models.embargo_periods,
            final_holdout_periods=self.config.models.final_holdout_periods,
            families=self.config.models.families,
        )
        registry = LocalModelRegistry(self.context.paths.models)
        path = registry.save_model(
            self.config.features.model_version,
            self.model,
            metadata={
                "run_id": self.context.run_id,
                "config_hash": self.config.content_hash(),
                "training_protocol": self.model["training_protocol"],
            },
        )
        self.outputs["model"] = path

    def predict(self) -> None:
        self.predictions = predict_all_families(self.features, self.model)
        prediction_paths: list[str] = []
        for _family, rows in sorted(self.predictions.items()):
            records: list[dict[str, Any]] = []
            for row in rows:
                record = dict(row)
                record["year"] = parse_utc(str(row["asof_ts"])).year
                records.append(record)
            paths = write_partitioned_parquet(
                records,
                self.context.paths.processed / "gold" / "predictions",
                partition_fields=("model_family", "year"),
                compression=self.config.data.compression,
                max_rows_per_file=self.config.data.write_batch_rows,
            )
            prediction_paths.extend(str(path) for path in paths)
        self.evaluation = evaluate_predictions(
            self.predictions,
            self.labels,
            context_rows=self.features,
            top_k=self.config.models.top_k,
            final_holdout_periods=self.config.models.final_holdout_periods,
            final_holdout_training=self.model["final_holdout_training"],
        )
        registry = LocalModelRegistry(self.context.paths.models)
        registry.record_metrics(self.config.features.model_version, self.evaluation)
        self.outputs["predictions"] = prediction_paths
        self.outputs["model_evaluation"] = _write_json_once(
            self.context.paths.processed / "evaluation" / "prediction_metrics.json",
            self.evaluation,
        )

    def backtest(self) -> None:
        engine = DeterministicBacktestEngine()
        protocol = self.evaluation.get("evaluation_protocol", {})
        holdout_start_value = protocol.get("final_holdout_start")
        if not isinstance(holdout_start_value, str):
            raise ValueError("backtesting requires a configured chronological final holdout")
        holdout_start = parse_utc(holdout_start_value)
        purged_times = {str(value) for value in protocol.get("purged_development_times", [])}
        pre_holdout_periods = int(protocol.get("pre_holdout_periods", 0))
        horizon_steps = self.config.features.horizon_days
        holdout_rebalance_offset = (-pre_holdout_periods) % horizon_steps
        for family, rows in sorted(self.predictions.items()):
            selection_depth = (
                None if family in {"equal_weight", "no_trade"} else self.config.models.top_k
            )
            development_rows = [
                row
                for row in rows
                if parse_utc(str(row["asof_ts"])) < holdout_start
                and str(row["asof_ts"]) not in purged_times
            ]
            holdout_rows = [row for row in rows if parse_utc(str(row["asof_ts"])) >= holdout_start]
            if not development_rows or not holdout_rows:
                raise ValueError("backtest holdout split must contain both evaluation windows")
            development_result = engine.run(
                development_rows,
                self.labels,
                self.config.backtest,
                top_k=selection_depth,
            )
            holdout_result = engine.run(
                holdout_rows,
                self.labels,
                self.config.backtest,
                top_k=selection_depth,
                rebalance_offset=holdout_rebalance_offset,
            )
            development_result["evaluation_window"] = {
                "name": "development",
                "end_exclusive": holdout_start_value,
            }
            holdout_result["evaluation_window"] = {
                "name": "final_holdout",
                "start_inclusive": holdout_start_value,
            }
            self.backtests[family] = development_result
            self.final_holdout_backtests[family] = holdout_result
            _write_json_once(
                self.context.paths.processed / "backtests" / family / "backtest.json",
                development_result,
            )
            _write_json_once(
                self.context.paths.processed / "backtests" / family / "final_holdout.json",
                holdout_result,
            )
        provenance = {
            "run_id": self.context.run_id,
            "created_at": format_utc(self.context.created_at),
            "config_hash": self.config.content_hash(),
            "code_version": self.context.code,
            "input_manifest": [
                {
                    "role": entry["role"],
                    "sha256": entry.get("sha256"),
                    "exists": entry["exists"],
                }
                for entry in self.context.inputs
            ],
            "feature_set_version": self.config.features.feature_set_version,
            "label_version": self.config.features.label_version,
            "model_version": self.config.features.model_version,
        }
        comparison_assumptions = {
            "horizon_steps": horizon_steps,
            "rebalance_frequency": self.config.backtest.rebalance_frequency,
            "top_k": self.config.models.top_k,
            "uncapped_families": ["equal_weight", "no_trade"],
            "portfolio_ranking": "direction eligibility, then absolute score, then asset_id",
            "costs_and_constraints": self.config.backtest.model_dump(mode="json"),
        }
        comparison = {
            "artifact_schema_version": "backtest-comparison-v2",
            "provenance": provenance,
            "assumptions": comparison_assumptions,
            "evaluation_window": {
                "name": "development",
                "end_exclusive": holdout_start_value,
            },
            "evaluation_protocol": protocol,
            "families": {
                family: result["metrics"] for family, result in sorted(self.backtests.items())
            },
        }
        holdout_comparison = {
            "artifact_schema_version": "backtest-comparison-v2",
            "provenance": provenance,
            "assumptions": comparison_assumptions,
            "evaluation_window": {
                "name": "final_holdout",
                "start_inclusive": holdout_start_value,
                "end_inclusive": protocol.get("final_holdout_end"),
            },
            "evaluation_protocol": protocol,
            "families": {
                family: result["metrics"]
                for family, result in sorted(self.final_holdout_backtests.items())
            },
        }
        self.outputs["backtests"] = str(self.context.paths.processed / "backtests")
        self.outputs["backtest_comparison"] = _write_json_once(
            self.context.paths.processed / "evaluation" / "backtest_comparison.json",
            comparison,
        )
        self.outputs["final_holdout_backtest_comparison"] = _write_json_once(
            self.context.paths.processed / "evaluation" / "final_holdout_backtest_comparison.json",
            holdout_comparison,
        )
        if self.config.llm_annotations.feature_mode == "augment":
            ablation_pairs = {
                "llm_only_vs_conventional_text": ("text", "llm"),
                "numeric_plus_llm_vs_numeric": ("traditional", "traditional_llm"),
                "all_vs_numeric_plus_conventional_text": ("combined", "all"),
            }

            def comparisons_for(
                results: dict[str, dict[str, Any]],
            ) -> dict[str, dict[str, Any]]:
                comparisons: dict[str, dict[str, Any]] = {}
                for name, (baseline_family, enhanced_family) in ablation_pairs.items():
                    baseline_metrics = results[baseline_family]["metrics"]
                    enhanced_metrics = results[enhanced_family]["metrics"]
                    comparisons[name] = {
                        "baseline_family": baseline_family,
                        "enhanced_family": enhanced_family,
                        "baseline_metrics": baseline_metrics,
                        "enhanced_metrics": enhanced_metrics,
                        "enhanced_minus_baseline": _numeric_metric_deltas(
                            baseline_metrics,
                            enhanced_metrics,
                        ),
                    }
                return comparisons

            llm_ablation = {
                "artifact_schema_version": "llm-ablation-comparison-v1",
                "run_id": self.context.run_id,
                "interpretation": (
                    "Arithmetic deltas only; positive values are not evidence of statistical "
                    "significance, causality, profitability, or successful promotion."
                ),
                "family_semantics": {
                    "traditional": "deterministic numeric market features",
                    "text": "conventional text features without LLM columns",
                    "combined": "traditional plus conventional text",
                    "llm": "LLM semantic/evidence features only",
                    "traditional_llm": "traditional plus LLM semantic/evidence features",
                    "all": "traditional, conventional text, and LLM semantic/evidence features",
                },
                "development": comparisons_for(self.backtests),
                "final_holdout": comparisons_for(self.final_holdout_backtests),
                "intelligence_cost": {
                    "generated_input_token_count": self.llm_annotation_summary.get(
                        "generated_input_token_count"
                    ),
                    "generated_output_token_count": self.llm_annotation_summary.get(
                        "generated_output_token_count"
                    ),
                    "generation_latency_seconds": self.llm_annotation_summary.get(
                        "generation_latency_seconds"
                    ),
                    "estimated_inference_cost_usd": self.llm_annotation_summary.get(
                        "estimated_inference_cost_usd"
                    ),
                },
            }
            self.outputs["llm_ablation_comparison"] = _write_json_once(
                self.context.paths.processed / "evaluation" / "llm_ablation_comparison.json",
                llm_ablation,
            )

    def paper(self) -> None:
        rows = self.predictions.get("combined", [])
        if not rows:
            raise ValueError("paper simulation requires combined-model predictions")
        latest_ts = max(str(row["asof_ts"]) for row in rows)
        latest = [row for row in rows if str(row["asof_ts"]) == latest_ts]
        intended_execution_ts = format_utc(
            self._calendar().next_open_decision_time(
                parse_utc(latest_ts) + timedelta(microseconds=1)
            )
        )
        horizon_steps = self.config.features.horizon_days
        entry_constraints = round_trip_entry_constraints(
            self.config.backtest,
            horizon_steps=horizon_steps,
        )
        paper_protocol = {
            "run_id": self.context.run_id,
            "config_hash": self.config.content_hash(),
            "feature_set_version": self.config.features.feature_set_version,
            "model_version": self.config.features.model_version,
            "horizon_steps": horizon_steps,
            "top_k": self.config.models.top_k,
            "same_day_exit_notional_buffer": (self.config.backtest.same_day_exit_notional_buffer),
            "effective_entry_constraints": constraint_snapshot(entry_constraints),
        }
        decision = construct_portfolio(
            latest,
            {},
            entry_constraints,
            equity=self.config.backtest.initial_capital,
            top_k=self.config.models.top_k,
        )
        desired = dict(decision.target_weights)
        intents = [
            PaperOrderIntent(
                strategy_id="combined-research-paper",
                asof_ts=intended_execution_ts,
                asset_id=str(row["asset_id"]),
                symbol=str(row["symbol"]),
                target_weight=decision.target_weights.get(str(row["asset_id"]), 0.0),
                side=(
                    "BUY"
                    if decision.target_weights.get(str(row["asset_id"]), 0.0) > 0
                    else "SHORT"
                    if decision.target_weights.get(str(row["asset_id"]), 0.0) < 0
                    else "FLAT"
                ),
                reason_codes=("combined_score_selected",)
                if str(row["asset_id"]) in desired
                else ("not_selected",),
            )
            for row in latest
        ]
        rows_by_asset = {str(row["asset_id"]): row for row in latest}
        snapshot = {
            "simulation_only": True,
            "status": "pending_unfilled_intents",
            "decision_ts": latest_ts,
            "intended_execution_ts": intended_execution_ts,
            "initial_capital": self.config.backtest.initial_capital,
            "equity": self.config.backtest.initial_capital,
            "research_protocol": paper_protocol,
            "positions": {},
            "trades": [],
            "intents": [
                {
                    "strategy_id": intent.strategy_id,
                    "decision_ts": latest_ts,
                    "intended_execution_ts": intent.asof_ts,
                    "asset_id": intent.asset_id,
                    "symbol": intent.symbol,
                    "target_weight": intent.target_weight,
                    "side": intent.side,
                    "reason_codes": list(intent.reason_codes),
                    "risk_flags": sorted(
                        set(decision.rejected.get(intent.asset_id, ()))
                        | set(risk_estimate_flags(rows_by_asset.get(intent.asset_id, {})))
                    ),
                    "decision_liquidity_proxy_dollar_volume": rows_by_asset.get(
                        intent.asset_id, {}
                    ).get("dollar_volume"),
                }
                for intent in intents
            ],
            "portfolio_risk_flags": sorted(
                set(decision.risk_flags)
                | {
                    flag
                    for row in latest
                    if str(row["asset_id"]) in decision.target_weights
                    for flag in risk_estimate_flags(row)
                }
            ),
            "execution_note": "simulation-only next-session-open intent; no broker order, "
            "fill, cost, cash, or position mutation has occurred",
        }
        ledger_path = self.context.paths.processed / "paper" / "events.jsonl"
        ledger_event = PaperEventLedger(ledger_path).append(
            {
                "event_type": "paper_intent_batch",
                "asof_ts": intended_execution_ts,
                "simulation_only": True,
                "status": snapshot["status"],
                "decision_ts": latest_ts,
                "research_protocol": paper_protocol,
                "intents": snapshot["intents"],
                "portfolio_risk_flags": snapshot["portfolio_risk_flags"],
            }
        )
        snapshot["ledger"] = {
            "path": str(ledger_path),
            "sequence": ledger_event["sequence"],
            "event_hash": ledger_event["event_hash"],
        }
        self.outputs["paper"] = _write_json_once(
            self.context.paths.processed / "paper" / "snapshot.json",
            snapshot,
        )

    def report(self) -> None:
        primary = self.backtests.get("combined")
        if primary is None:
            raise ValueError("reporting requires a combined-model backtest")
        evaluation = {
            "prediction": self.evaluation,
            "portfolio": {
                family: result["metrics"] for family, result in sorted(self.backtests.items())
            },
            "portfolio_final_holdout": {
                family: result["metrics"]
                for family, result in sorted(self.final_holdout_backtests.items())
            },
        }
        if self.llm_annotation_summary:
            evaluation["llm_annotations"] = self.llm_annotation_summary
        path = write_report(
            self.config,
            primary,
            self.context.paths.reports / "research_note.md",
            report_run_id=self.context.run_id,
            created_at=self.context.created_at,
            code_version=self.context.code,
            data_manifest=list(self.context.inputs),
            universe=[asset.symbol for asset in self.assets],
            period=self.period(),
            model_evaluation=evaluation,
            known_limitations=self.limitations(),
            next_questions=self.next_questions(),
        )
        self.outputs["report"] = path

    def period(self) -> dict[str, str | None]:
        decision_bars = self._decision_bars()
        if not decision_bars:
            return {"start": None, "end": None}
        calendar = self._calendar()
        timestamps = [
            market_decision_time_for_bar(
                bar,
                self.market_decision_times_by_session,
                calendar.timezone,
            )
            for bar in decision_bars
        ]
        return {"start": format_utc(min(timestamps)), "end": format_utc(max(timestamps))}

    def limitations(self) -> list[str]:
        values = list(DEFAULT_LIMITATIONS)
        if self.config.llm_annotations.enabled:
            values.append(
                "LLM annotations are retrospective, source-grounded parsing; the "
                "model may retain facts learned during pretraining after the historical period."
            )
        if (
            self.config.llm_annotations.feature_mode == "sidecar"
            and not self.enable_transformer_sentiment
        ):
            values.append("Text sentiment uses a deterministic finance dictionary baseline.")
        if (
            self.config.llm_annotations.enabled
            and self.config.llm_annotations.feature_mode == "sidecar"
        ):
            values.append(
                "LLM annotations are sidecar-only in this run and do not affect features or "
                "research results."
            )
        if self.config.llm_annotations.feature_mode == "augment":
            values.append(
                "LLM semantic features come from a retrospective pretrained parser; raw language "
                "confidence is uncalibrated and remains separate from semantic direction."
            )
        if self.config.mode == "sample":
            values.append("No sample metric is evidence of investment profitability.")
        return values

    @staticmethod
    def next_questions() -> list[str]:
        return [
            "Validate with licensed point-in-time data and a survivorship-aware universe.",
            "Test stability by sector, liquidity, volatility regime, source, and event type.",
            "Calibrate capacity and execution assumptions against user-provided quote or fill "
            "data.",
        ]

    def final_metrics(self) -> dict[str, Any]:
        if self.backtests:
            metrics: dict[str, Any] = {
                "prediction": self.evaluation,
                "portfolio": {
                    family: result["metrics"] for family, result in sorted(self.backtests.items())
                },
                "portfolio_final_holdout": {
                    family: result["metrics"]
                    for family, result in sorted(self.final_holdout_backtests.items())
                },
            }
            if self.llm_annotation_summary:
                metrics["llm_annotations"] = self.llm_annotation_summary
            return metrics
        metrics = {
            "assets": len(self.assets),
            "bars": len(self._decision_bars()),
            "text_items": len(self.items),
            "features": len(self.features),
            "labels": len(self.labels),
        }
        if self.llm_annotation_summary:
            metrics["llm_annotations"] = self.llm_annotation_summary
        return metrics


def validate(config: ResearchConfig) -> dict[str, Any]:
    errors = validate_config(config)
    return {"ok": not errors, "errors": errors}


def run_to_stage(
    config: ResearchConfig,
    stage: str,
    *,
    llm_generator: LLMGenerator | None = None,
) -> dict[str, Any]:
    target = "report" if stage == "smoke" else stage.replace("-", "_")
    if target not in DEPENDENCIES:
        raise ValueError(f"unknown pipeline stage: {stage}")
    errors = validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    context = create_run_context(config)
    LOGGER.info(
        "run_start run_id=%s target=%s mode=%s",
        context.run_id,
        target,
        config.mode,
    )
    execution = PipelineExecution(
        config,
        context,
        llm_generator=llm_generator,
    )
    try:
        execution.run(target)
        final_manifest = finalize_run(
            context,
            universe=[asset.symbol for asset in execution.assets],
            period=execution.period(),
            metrics=execution.final_metrics(),
            known_limitations=execution.limitations(),
            next_questions=execution.next_questions(),
            stage=target,
        )
        execution.outputs["final_manifest"] = final_manifest
        LOGGER.info("run_complete run_id=%s target=%s", context.run_id, target)
        return execution.outputs
    except Exception as error:
        LOGGER.exception("run_failed run_id=%s target=%s", context.run_id, target)
        fail_run(context, error, stage=target)
        raise


def ingest_market(config: ResearchConfig) -> dict[str, Any]:
    return run_to_stage(config, "ingest_market")


def ingest_text(config: ResearchConfig) -> dict[str, Any]:
    return run_to_stage(config, "ingest_text")


def annotate_text(
    config: ResearchConfig,
    *,
    llm_generator: LLMGenerator | None = None,
) -> dict[str, Any]:
    return run_to_stage(config, "annotate_text", llm_generator=llm_generator)


def build_features(
    config: ResearchConfig,
    *,
    llm_generator: LLMGenerator | None = None,
) -> dict[str, Any]:
    return run_to_stage(config, "build_features", llm_generator=llm_generator)


def build_labels(config: ResearchConfig) -> dict[str, Any]:
    return run_to_stage(config, "build_labels")


def train(
    config: ResearchConfig,
    *,
    llm_generator: LLMGenerator | None = None,
) -> dict[str, Any]:
    return run_to_stage(config, "train", llm_generator=llm_generator)


def predict(
    config: ResearchConfig,
    *,
    llm_generator: LLMGenerator | None = None,
) -> dict[str, Any]:
    return run_to_stage(config, "predict", llm_generator=llm_generator)


def backtest(
    config: ResearchConfig,
    *,
    llm_generator: LLMGenerator | None = None,
) -> dict[str, Any]:
    return run_to_stage(config, "backtest", llm_generator=llm_generator)


def paper(
    config: ResearchConfig,
    *,
    llm_generator: LLMGenerator | None = None,
) -> dict[str, Any]:
    return run_to_stage(config, "paper", llm_generator=llm_generator)


def report(
    config: ResearchConfig,
    *,
    llm_generator: LLMGenerator | None = None,
) -> dict[str, Any]:
    return run_to_stage(config, "report", llm_generator=llm_generator)


def smoke(
    config: ResearchConfig,
    *,
    llm_generator: LLMGenerator | None = None,
) -> dict[str, Any]:
    return run_to_stage(config, "smoke", llm_generator=llm_generator)
