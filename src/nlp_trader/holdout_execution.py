from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from nlp_trader.backtest.engine import DeterministicBacktestEngine
from nlp_trader.config import ResearchConfig
from nlp_trader.immutable.append import SafeFileError, write_bytes_exclusive_durable
from nlp_trader.models.baselines import complete_label_cross_sections, predict_all_families
from nlp_trader.models.evaluation import prediction_metrics
from nlp_trader.pipeline import PipelineExecution, PipelineExecutionScope
from nlp_trader.research import (
    RunContext,
    create_run_context,
    fail_run,
    finalize_run,
    sha256_file,
)
from nlp_trader.research_agents.approvals import CandidateFreezeRecord
from nlp_trader.research_agents.authority import load_authoritative_candidate_freeze
from nlp_trader.research_agents.compiler import (
    apply_definition_patch,
    load_compiled_execution_definition,
)
from nlp_trader.research_agents.contracts import (
    HoldoutIdentity,
    Sha256,
    StrictModel,
    TimeRange,
    canonical_json,
    content_sha256,
)
from nlp_trader.research_agents.holdouts import reserve_frozen_holdout
from nlp_trader.research_agents.registry import ResearchRegistryLedger
from nlp_trader.timestamps import parse_utc


class HoldoutResultManifest(StrictModel):
    artifact_schema_version: Literal["holdout-result-manifest-v1"] = "holdout-result-manifest-v1"
    manifest_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    study_id: Sha256
    reservation_id: Sha256
    reservation_event_hash: Sha256
    candidate_hash: Sha256
    holdout_identity: HoldoutIdentity
    frozen_model_manifest_hash: Sha256
    frozen_model_hash_before: Sha256
    frozen_model_hash_after: Sha256
    training_updates: Literal[0] = 0
    selected_family: str
    execution_definition_hash: Sha256
    base_config_hash: Sha256
    patched_config_hash: Sha256
    required_evaluation_contract_hash: Sha256
    cost_assumptions_hash: Sha256
    constraint_assumptions_hash: Sha256
    attempted_proposal_count: int = Field(ge=1)
    attempted_proposals_hash: Sha256
    feature_snapshot_hash: Sha256
    label_snapshot_hash: Sha256
    prediction_results_hash: Sha256
    backtest_results_hash: Sha256
    metrics_hash: Sha256
    pipeline_run_final_manifest_hash: Sha256
    input_snapshot_hashes: tuple[Sha256, ...]
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.frozen_model_hash_before != self.frozen_model_hash_after:
            raise ValueError("holdout execution mutated the frozen model")
        if self.holdout_identity.candidate_hash != self.candidate_hash:
            raise ValueError("holdout result candidate identity does not match")
        expected = content_sha256(self.model_dump(mode="json", exclude={"manifest_id"}))
        if self.manifest_id and self.manifest_id != expected:
            raise ValueError("holdout result manifest_id does not match canonical content")
        if not self.manifest_id:
            object.__setattr__(self, "manifest_id", expected)
        return self


def reveal_frozen_holdout(
    *,
    ledger: ResearchRegistryLedger,
    freeze: CandidateFreezeRecord,
    development_root: Path,
    base_config: ResearchConfig,
    actor_label: str,
) -> tuple[Path, HoldoutResultManifest]:
    authority = load_authoritative_candidate_freeze(
        ledger,
        freeze,
        development_root=development_root,
    )
    freeze = authority.record
    development_root = authority.development_root
    reservation = reserve_frozen_holdout(ledger, freeze, actor_label=actor_label)
    destination = ledger.artifact_root / "holdout_reveals" / reservation.reservation_id
    context: RunContext | None = None
    try:
        destination.mkdir(mode=0o700, parents=True, exist_ok=False)
        candidate_config = _read_object(
            ledger.artifact_root
            / "studies"
            / freeze.study_id
            / "candidate"
            / "candidate_config.json"
        )
        if content_sha256(candidate_config) != freeze.candidate_config_hash:
            raise ValueError("stored candidate config does not match the frozen candidate")
        definition = load_compiled_execution_definition(
            ledger.artifact_root,
            freeze.execution_definition_hash,
        )
        patched_config = apply_definition_patch(definition, base_config)
        backtest_contract_hash = content_sha256(patched_config.backtest.model_dump(mode="json"))
        if (
            backtest_contract_hash != definition.cost_assumptions_hash
            or backtest_contract_hash != definition.constraint_assumptions_hash
        ):
            raise ValueError("holdout cost or constraint contract changed after compilation")
        selected_family = candidate_config.get("selected_family")
        if not isinstance(selected_family, str) or not selected_family:
            raise ValueError("candidate config requires one selected model family")
        model_path = development_root / "frozen_model.json"
        model_manifest = authority.frozen_model_manifest
        model_bytes = model_path.read_bytes()
        model_hash_before = hashlib.sha256(model_bytes).hexdigest()
        if model_hash_before != model_manifest.model_artifact_hash:
            raise ValueError("frozen model bytes do not match their manifest")
        model = json.loads(model_bytes)
        if not isinstance(model, dict) or "development_training" not in model:
            raise ValueError("candidate is not a frozen development model")
        if "final_holdout_training" in model:
            raise ValueError("candidate model contains an unexpected alternate training state")
        snapshots = model.get("walk_forward_snapshots")
        if not isinstance(snapshots, list) or not snapshots:
            raise ValueError("frozen model has no causal development snapshots")
        if any(
            parse_utc(str(snapshot["asof_ts"])) >= freeze.holdout_identity.decision_interval.start
            for snapshot in snapshots
            if isinstance(snapshot, dict) and "asof_ts" in snapshot
        ):
            raise ValueError("frozen model contains training state from the reserved interval")
        context = create_run_context(
            patched_config,
            run_id=reservation.reservation_id,
        )
        inputs = context.inputs
        input_hashes = tuple(
            sorted(
                str(value["sha256"])
                for value in inputs
                if value.get("exists") and isinstance(value.get("sha256"), str)
            )
        )
        if input_hashes != freeze.holdout_identity.input_snapshot_hashes:
            raise ValueError("current input snapshots do not match the frozen holdout identity")
        execution = PipelineExecution(
            patched_config,
            context,
            execution_scope=PipelineExecutionScope(
                mode="holdout_evaluation",
                decision_start=freeze.holdout_identity.decision_interval.start,
                decision_end=freeze.holdout_identity.decision_interval.end,
            ),
        )
        execution.run("build_features")
        execution.run("build_labels")
        universe = tuple(sorted(value.asset_id for value in execution.assets))
        if universe != freeze.holdout_identity.universe_asset_ids:
            raise ValueError("holdout asset membership changed after candidate freeze")
        features, labels = _select_complete_holdout_rows(
            execution.features,
            execution.labels,
            outcome_interval=freeze.holdout_identity.outcome_interval,
        )
        predictions = predict_all_families(features, model)
        if selected_family not in predictions:
            raise ValueError("frozen selected family is not present in model predictions")
        if not predictions[selected_family]:
            raise ValueError("frozen selected family produced no holdout predictions")
        prediction_metrics_by_family = {
            family: prediction_metrics(rows, labels, top_k=patched_config.models.top_k)
            for family, rows in sorted(predictions.items())
        }
        engine = DeterministicBacktestEngine()
        backtests = {
            family: engine.run(
                rows,
                labels,
                patched_config.backtest,
                top_k=(
                    None if family in {"equal_weight", "no_trade"} else patched_config.models.top_k
                ),
            )
            for family, rows in sorted(predictions.items())
        }
        prediction_payload = {
            "artifact_schema_version": "frozen-holdout-predictions-v1",
            "reservation_id": reservation.reservation_id,
            "candidate_hash": freeze.candidate_config_hash,
            "families": predictions,
        }
        backtest_payload = {
            "artifact_schema_version": "frozen-holdout-backtests-v1",
            "reservation_id": reservation.reservation_id,
            "families": backtests,
        }
        metrics_payload = {
            "artifact_schema_version": "frozen-holdout-metrics-v1",
            "reservation_id": reservation.reservation_id,
            "selected_family": selected_family,
            "prediction": prediction_metrics_by_family,
            "portfolio": {
                family: result["metrics"] for family, result in sorted(backtests.items())
            },
            "interpretation": (
                "One-time hypothetical evaluation; no metric is evidence of profitability."
            ),
        }
        model_hash_after = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if model_hash_after != model_hash_before:
            raise ValueError("holdout execution mutated the frozen model")
        prediction_bytes = _write_json(destination / "predictions.json", prediction_payload)
        backtest_bytes = _write_json(destination / "backtests.json", backtest_payload)
        metrics_bytes = _write_json(destination / "metrics.json", metrics_payload)
        pipeline_final_path = finalize_run(
            context,
            universe=list(universe),
            period=execution.period(),
            metrics={
                "prediction": prediction_metrics_by_family,
                "portfolio": {
                    family: value["metrics"] for family, value in sorted(backtests.items())
                },
            },
            known_limitations=[
                *execution.limitations(),
                "This pipeline run is one reserved, one-time frozen-candidate evaluation.",
            ],
            next_questions=execution.next_questions(),
            stage="holdout_evaluation",
        )
        pipeline_final_hash = sha256_file(pipeline_final_path)
        attempt_snapshot = ledger.project().studies[freeze.study_id].attempts
        result = HoldoutResultManifest(
            study_id=freeze.study_id,
            reservation_id=reservation.reservation_id,
            reservation_event_hash=reservation.registry_event_hash,
            candidate_hash=freeze.candidate_config_hash,
            holdout_identity=freeze.holdout_identity,
            frozen_model_manifest_hash=freeze.frozen_model_manifest_hash,
            frozen_model_hash_before=model_hash_before,
            frozen_model_hash_after=model_hash_after,
            selected_family=selected_family,
            execution_definition_hash=definition.definition_id,
            base_config_hash=definition.base_config_hash,
            patched_config_hash=patched_config.content_hash(),
            required_evaluation_contract_hash=freeze.required_evaluation_contract_hash,
            cost_assumptions_hash=definition.cost_assumptions_hash,
            constraint_assumptions_hash=definition.constraint_assumptions_hash,
            attempted_proposal_count=len(attempt_snapshot),
            attempted_proposals_hash=content_sha256(
                [value.model_dump(mode="json") for value in attempt_snapshot]
            ),
            feature_snapshot_hash=content_sha256(features),
            label_snapshot_hash=content_sha256(labels),
            prediction_results_hash=hashlib.sha256(prediction_bytes).hexdigest(),
            backtest_results_hash=hashlib.sha256(backtest_bytes).hexdigest(),
            metrics_hash=hashlib.sha256(metrics_bytes).hexdigest(),
            pipeline_run_final_manifest_hash=pipeline_final_hash,
            input_snapshot_hashes=input_hashes,
            limitations=(
                "One reserved interval and one frozen candidate are evaluated exactly once.",
                "Results are hypothetical research and do not authorize trading.",
            ),
        )
        result_bytes = _write_json(
            destination / "result_manifest.json", result.model_dump(mode="json")
        )
        ledger.complete_holdout_reveal(
            freeze.study_id,
            reservation.reservation_id,
            candidate_hash=freeze.candidate_config_hash,
            holdout_identity=freeze.holdout_identity,
            result_manifest_hash=hashlib.sha256(result_bytes).hexdigest(),
            expected_head_hash=ledger.head_hash(),
            actor_label=actor_label,
        )
        return destination, result
    except Exception as exc:
        _record_pipeline_failure(context, exc)
        with suppress(ValueError):
            ledger.fail_holdout_reveal(
                freeze.study_id,
                reservation.reservation_id,
                failure_stage=type(exc).__name__,
                detail="one-time holdout reveal failed after reservation",
                expected_head_hash=ledger.head_hash(),
                actor_label=actor_label,
            )
        raise


def _record_pipeline_failure(context: RunContext | None, error: Exception) -> None:
    if context is None:
        return
    final_path = context.paths.reports / "run.final.json"
    failed_path = context.paths.reports / "run.failed.json"
    if final_path.exists() or failed_path.exists():
        return
    with suppress(FileExistsError, OSError, TypeError, ValueError):
        fail_run(context, error, stage="holdout_evaluation")


def _select_complete_holdout_rows(
    features: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    outcome_interval: TimeRange,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels_by_key, observed_keys = complete_label_cross_sections(
        features,
        labels,
        row_name="holdout features",
    )
    if not observed_keys:
        raise ValueError("reserved holdout contains no complete label cross-section")
    selected_features = [row for row in features if _row_key(row) in observed_keys]
    selected_labels = [labels_by_key[_row_key(row)] for row in selected_features]
    for label in selected_labels:
        key = _row_key(label)
        for field_name in ("label_end_ts", "label_available_at"):
            value = label.get(field_name)
            if not isinstance(value, str):
                raise ValueError(f"observed holdout label {key} requires {field_name}")
            timestamp = parse_utc(value)
            if not outcome_interval.start <= timestamp <= outcome_interval.end:
                raise ValueError(
                    f"observed holdout label {key} has {field_name} outside the frozen "
                    "outcome interval"
                )
    return selected_features, selected_labels


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["asset_id"]), str(row["asof_ts"]), str(row["horizon"])


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen candidate artifact cannot be read") from exc
    if not isinstance(value, dict):
        raise ValueError("frozen candidate artifact must contain an object")
    return {str(key): nested for key, nested in value.items()}


def _write_json(path: Path, value: object) -> bytes:
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    try:
        write_bytes_exclusive_durable(path, encoded)
    except (FileExistsError, SafeFileError, OSError, ValueError) as exc:
        raise ValueError("holdout artifact cannot be written immutably") from exc
    return encoded
