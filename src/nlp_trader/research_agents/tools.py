from __future__ import annotations

import math
import statistics
import time
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from nlp_trader.research_agents.catalog import CatalogEntry, CatalogSection
from nlp_trader.research_agents.contracts import (
    CalculateRequest,
    CalculateToolCall,
    CalculationReference,
    MetricReference,
    ReadDevelopmentMetricsRequest,
    ReadDevelopmentMetricsToolCall,
    ReadFeatureCatalogRequest,
    ReadFeatureCatalogToolCall,
    ResearchAgentToolCall,
    SearchEvidenceRequest,
    SearchEvidenceToolCall,
    Sha256,
    StrictModel,
    ToolCallAction,
    canonical_json,
    content_sha256,
)
from nlp_trader.research_agents.index import EvidenceSearchPage, search_evidence
from nlp_trader.research_agents.views import DevelopmentMetric, LoadedDevelopmentViewBundle

ToolName = Literal[
    "search_evidence",
    "read_development_metrics",
    "read_feature_catalog",
    "calculate",
]


class SearchEvidenceToolResult(StrictModel):
    payload_type: Literal["search_evidence"] = "search_evidence"
    page: EvidenceSearchPage

    def core_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class DevelopmentMetricsToolResult(StrictModel):
    payload_type: Literal["read_development_metrics"] = "read_development_metrics"
    metrics: tuple[MetricReference, ...]

    def core_payload(self) -> dict[str, Any]:
        return {
            "payload_type": self.payload_type,
            "metrics": [
                value.model_dump(
                    mode="json",
                    exclude={"metric_reference_id", "tool_result_hash"},
                )
                for value in self.metrics
            ],
        }


class FeatureCatalogToolResult(StrictModel):
    payload_type: Literal["read_feature_catalog"] = "read_feature_catalog"
    section: CatalogSection
    entries: tuple[CatalogEntry, ...]

    def core_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class CalculateToolResult(StrictModel):
    payload_type: Literal["calculate"] = "calculate"
    calculation: CalculationReference

    def core_payload(self) -> dict[str, Any]:
        return {
            "payload_type": self.payload_type,
            "calculation": self.calculation.model_dump(
                mode="json", exclude={"calculation_id", "tool_result_hash"}
            ),
        }


ToolResultPayload = Annotated[
    SearchEvidenceToolResult
    | DevelopmentMetricsToolResult
    | FeatureCatalogToolResult
    | CalculateToolResult,
    Field(discriminator="payload_type"),
]


class ToolResultEnvelope(StrictModel):
    artifact_schema_version: Literal["research-agent-tool-result-v1"] = (
        "research-agent-tool-result-v1"
    )
    tool_call_id: Sha256
    tool_name: ToolName
    tool_schema_version: Literal["research-agent-tools-v1"] = "research-agent-tools-v1"
    bundle_id: Sha256
    study_id: Sha256
    request_hash: Sha256
    result_hash: Sha256
    payload: ToolResultPayload
    result_bytes: int = Field(ge=1)
    next_cursor: str | None = Field(default=None, max_length=4096)
    latency_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        if self.tool_name != self.payload.payload_type:
            raise ValueError("tool result payload type does not match tool_name")
        if self.tool_call_id != content_sha256(
            {
                "bundle_id": self.bundle_id,
                "study_id": self.study_id,
                "tool_name": self.tool_name,
                "request_hash": self.request_hash,
            }
        ):
            raise ValueError("tool_call_id does not match canonical request identity")
        expected_hash = _result_hash(
            tool_name=self.tool_name,
            bundle_id=self.bundle_id,
            study_id=self.study_id,
            request_hash=self.request_hash,
            payload=self.payload,
            next_cursor=self.next_cursor,
        )
        if self.result_hash != expected_hash:
            raise ValueError("tool result_hash does not match canonical result content")
        if self.result_bytes != len(self.model_visible_json().encode("utf-8")):
            raise ValueError("tool result_bytes does not match model-visible payload bytes")
        if isinstance(self.payload, DevelopmentMetricsToolResult) and any(
            value.tool_result_hash != self.result_hash for value in self.payload.metrics
        ):
            raise ValueError("metric references do not bind this tool result")
        if (
            isinstance(self.payload, CalculateToolResult)
            and self.payload.calculation.tool_result_hash != self.result_hash
        ):
            raise ValueError("calculation reference does not bind this tool result")
        return self

    def model_visible_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"latency_ms", "result_bytes"})

    def model_visible_json(self) -> str:
        return canonical_json(self.model_visible_payload())

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


class ToolLimits(StrictModel):
    max_evidence_results: int = Field(ge=1, le=1_000)
    max_result_bytes: int = Field(ge=1)


class ResearchToolGateway:
    """Closed pure-tool dispatcher over one already loaded and verified bundle."""

    def __init__(self, bundle: LoadedDevelopmentViewBundle, limits: ToolLimits) -> None:
        self._bundle = bundle
        self._limits = limits

    def execute(self, action: ToolCallAction) -> ToolResultEnvelope:
        if not isinstance(action, ToolCallAction):
            raise TypeError("tool gateway accepts exactly one typed ToolCallAction")
        call = action.tool_call
        started = time.monotonic()
        payload: ToolResultPayload
        next_cursor: str | None
        if isinstance(call, SearchEvidenceToolCall):
            payload, next_cursor = self._search_evidence(call.request)
        elif isinstance(call, ReadDevelopmentMetricsToolCall):
            payload, next_cursor = self._read_development_metrics(call.request)
        elif isinstance(call, ReadFeatureCatalogToolCall):
            payload, next_cursor = self._read_feature_catalog(call.request)
        elif isinstance(call, CalculateToolCall):
            payload, next_cursor = self._calculate(call.request)
        else:  # pragma: no cover - discriminated union is exhaustive
            raise TypeError("unsupported typed research tool call")
        envelope = _make_envelope(
            bundle_id=self._bundle.manifest.bundle_id,
            study_id=self._bundle.manifest.study_id,
            tool_name=call.tool_name,
            request=call,
            payload=payload,
            next_cursor=next_cursor,
            latency_ms=max(0.0, (time.monotonic() - started) * 1000.0),
        )
        if envelope.result_bytes > self._limits.max_result_bytes:
            raise ValueError("tool result exceeds the configured per-step byte limit")
        return envelope

    def _search_evidence(
        self, request: SearchEvidenceRequest
    ) -> tuple[SearchEvidenceToolResult, str | None]:
        if request.result_limit > self._limits.max_evidence_results:
            raise ValueError("evidence result limit exceeds the configured bound")
        universe = set(self._bundle.development_view.universe_asset_ids)
        if not set(request.asset_ids).issubset(universe):
            raise ValueError("evidence request contains an asset outside the bundle universe")
        if (
            request.available_range is not None
            and request.available_range.end > self._bundle.manifest.analysis_cutoff
        ):
            raise ValueError("evidence request time range exceeds the analysis cutoff")
        page = search_evidence(
            self._bundle.evidence_index,
            self._bundle.evidence,
            request,
        )
        return SearchEvidenceToolResult(page=page), page.next_cursor

    def _read_development_metrics(
        self, request: ReadDevelopmentMetricsRequest
    ) -> tuple[DevelopmentMetricsToolResult, None]:
        selected = tuple(
            metric
            for metric in self._bundle.development_view.metrics
            if metric.metric_group == request.metric_group
            and (request.family is None or metric.family == request.family)
            and (request.segment is None or metric.segment == request.segment)
        )
        if not selected:
            raise ValueError("requested development metric group is not present in the bundle")
        request_hash = content_sha256(
            {
                "tool_name": "read_development_metrics",
                "request": request.model_dump(mode="json"),
            }
        )
        core_hash = content_sha256(
            {
                "tool_name": "read_development_metrics",
                "bundle_id": self._bundle.manifest.bundle_id,
                "study_id": self._bundle.manifest.study_id,
                "request_hash": request_hash,
                "payload": {
                    "payload_type": "read_development_metrics",
                    "metrics": [_metric_core(value) for value in selected],
                },
                "next_cursor": None,
            }
        )
        references = tuple(
            MetricReference(tool_result_hash=core_hash, **_metric_core(metric))
            for metric in selected
        )
        return DevelopmentMetricsToolResult(metrics=references), None

    def _read_feature_catalog(
        self, request: ReadFeatureCatalogRequest
    ) -> tuple[FeatureCatalogToolResult, None]:
        return (
            FeatureCatalogToolResult(
                section=request.section,
                entries=self._bundle.feature_catalog.section(request.section),
            ),
            None,
        )

    def _calculate(self, request: CalculateRequest) -> tuple[CalculateToolResult, None]:
        if request.rounding_policy != "decimal_12":
            raise ValueError("calculate supports only the decimal_12 rounding policy")
        output = _calculate_value(request)
        request_hash = content_sha256(
            {"tool_name": "calculate", "request": request.model_dump(mode="json")}
        )
        core = {
            "operation": request.operation,
            "inputs": [value.model_dump(mode="json") for value in request.inputs],
            "output": output,
            "rounding_policy": request.rounding_policy,
        }
        result_hash = content_sha256(
            {
                "tool_name": "calculate",
                "bundle_id": self._bundle.manifest.bundle_id,
                "study_id": self._bundle.manifest.study_id,
                "request_hash": request_hash,
                "payload": {"payload_type": "calculate", "calculation": core},
                "next_cursor": None,
            }
        )
        return (
            CalculateToolResult(
                calculation=CalculationReference(
                    operation=request.operation,
                    inputs=request.inputs,
                    output=output,
                    rounding_policy=request.rounding_policy,
                    tool_result_hash=result_hash,
                )
            ),
            None,
        )


def _metric_core(metric: DevelopmentMetric) -> dict[str, Any]:
    return {
        "metric_group": metric.metric_group,
        "family": metric.family,
        "segment": metric.segment,
        "metric_id": metric.metric_id,
        "value": metric.value,
        "unit": metric.unit,
        "scope": metric.scope,
        "window": metric.window.model_dump(mode="json"),
        "source_artifact_id": metric.source_artifact_id,
        "source_artifact_hash": metric.source_artifact_hash,
    }


def _make_envelope(
    *,
    bundle_id: str,
    study_id: str,
    tool_name: ToolName,
    request: ResearchAgentToolCall,
    payload: ToolResultPayload,
    next_cursor: str | None,
    latency_ms: float,
) -> ToolResultEnvelope:
    request_hash = content_sha256(
        {"tool_name": tool_name, "request": request.request.model_dump(mode="json")}
    )
    result_hash = _result_hash(
        tool_name=tool_name,
        bundle_id=bundle_id,
        study_id=study_id,
        request_hash=request_hash,
        payload=payload,
        next_cursor=next_cursor,
    )
    if isinstance(payload, DevelopmentMetricsToolResult) and any(
        metric.tool_result_hash != result_hash for metric in payload.metrics
    ):
        raise ValueError("development metric construction produced an inconsistent result hash")
    if (
        isinstance(payload, CalculateToolResult)
        and payload.calculation.tool_result_hash != result_hash
    ):
        raise ValueError("calculation construction produced an inconsistent result hash")
    tool_call_id = content_sha256(
        {
            "bundle_id": bundle_id,
            "study_id": study_id,
            "tool_name": tool_name,
            "request_hash": request_hash,
        }
    )
    temporary = {
        "artifact_schema_version": "research-agent-tool-result-v1",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "tool_schema_version": "research-agent-tools-v1",
        "bundle_id": bundle_id,
        "study_id": study_id,
        "request_hash": request_hash,
        "result_hash": result_hash,
        "payload": payload.model_dump(mode="json"),
        "next_cursor": next_cursor,
    }
    result_bytes = len(canonical_json(temporary).encode("utf-8"))
    return ToolResultEnvelope(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        bundle_id=bundle_id,
        study_id=study_id,
        request_hash=request_hash,
        result_hash=result_hash,
        payload=payload,
        next_cursor=next_cursor,
        result_bytes=result_bytes,
        latency_ms=latency_ms,
    )


def _result_hash(
    *,
    tool_name: ToolName,
    bundle_id: str,
    study_id: str,
    request_hash: str,
    payload: ToolResultPayload,
    next_cursor: str | None,
) -> str:
    return content_sha256(
        {
            "tool_name": tool_name,
            "bundle_id": bundle_id,
            "study_id": study_id,
            "request_hash": request_hash,
            "payload": payload.core_payload(),
            "next_cursor": next_cursor,
        }
    )


def _calculate_value(request: CalculateRequest) -> float:
    values = [value.value for value in request.inputs]
    operation = request.operation
    if operation == "sum":
        result = math.fsum(values)
    elif operation == "mean":
        result = statistics.fmean(values)
    elif operation == "difference":
        _require_count(values, 2, operation)
        result = values[0] - values[1]
    elif operation == "ratio":
        _require_count(values, 2, operation)
        if values[1] == 0:
            raise ValueError("ratio denominator cannot be zero")
        result = values[0] / values[1]
    elif operation == "percent_change":
        _require_count(values, 2, operation)
        if values[0] == 0:
            raise ValueError("percent_change baseline cannot be zero")
        result = (values[1] - values[0]) / abs(values[0])
    elif operation == "population_stddev":
        result = statistics.pstdev(values)
    elif operation == "sample_stddev":
        if len(values) < 2:
            raise ValueError("sample_stddev requires at least two values")
        result = statistics.stdev(values)
    else:
        if len(values) < 4 or len(values) % 2:
            raise ValueError(
                "pearson_correlation requires two equal samples of at least two values"
            )
        midpoint = len(values) // 2
        left, right = values[:midpoint], values[midpoint:]
        if statistics.pstdev(left) == 0 or statistics.pstdev(right) == 0:
            raise ValueError("pearson_correlation requires nonconstant samples")
        result = statistics.correlation(left, right)
    rounded = round(float(result), 12)
    if not math.isfinite(rounded):
        raise ValueError("calculation produced a non-finite result")
    return rounded


def _require_count(values: list[float], count: int, operation: str) -> None:
    if len(values) != count:
        raise ValueError(f"{operation} requires exactly {count} inputs")
