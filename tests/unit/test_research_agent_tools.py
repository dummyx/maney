from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from nlp_trader.research_agents.contracts import (
    CalculateRequest,
    CalculateToolCall,
    CalculationInput,
    ReadDevelopmentMetricsRequest,
    ReadDevelopmentMetricsToolCall,
    ReadFeatureCatalogRequest,
    ReadFeatureCatalogToolCall,
    SearchEvidenceRequest,
    SearchEvidenceToolCall,
    StudyDefinition,
    ToolCallAction,
)
from nlp_trader.research_agents.tools import ResearchToolGateway, ToolLimits
from nlp_trader.research_agents.views import load_development_view_bundle


def _gateway(
    tmp_path: Path,
    study: StudyDefinition,
    bundle_factory: Callable[..., tuple[Path, object]],
) -> ResearchToolGateway:
    root, _ = bundle_factory(tmp_path, study)
    return ResearchToolGateway(
        load_development_view_bundle(root),
        ToolLimits(max_evidence_results=10, max_result_bytes=100_000),
    )


def test_tool_gateway_is_typed_bounded_and_model_visible_output_is_deterministic(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
    research_agent_bundle_factory: Callable[..., tuple[Path, object]],
) -> None:
    gateway = _gateway(tmp_path, research_study_definition, research_agent_bundle_factory)
    action = ToolCallAction(
        tool_call=SearchEvidenceToolCall(
            request=SearchEvidenceRequest(
                query="demand weakened",
                purpose="challenge",
                asset_ids=("asset-a",),
                result_limit=5,
            )
        )
    )
    first = gateway.execute(action)
    second = gateway.execute(action)

    assert first.model_visible_json() == second.model_visible_json()
    assert first.payload.page.results
    assert "latency_ms" not in first.model_visible_json()
    assert "/Users/" not in first.model_visible_json()
    assert "final_holdout" not in first.model_visible_json()


def test_metric_catalog_and_calculation_tools_bind_exact_results(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
    research_agent_bundle_factory: Callable[..., tuple[Path, object]],
) -> None:
    gateway = _gateway(tmp_path, research_study_definition, research_agent_bundle_factory)
    metrics = gateway.execute(
        ToolCallAction(
            tool_call=ReadDevelopmentMetricsToolCall(
                request=ReadDevelopmentMetricsRequest(
                    metric_group="prediction",
                    family="combined",
                )
            )
        )
    )
    assert metrics.payload.metrics[0].tool_result_hash == metrics.result_hash

    catalog = gateway.execute(
        ToolCallAction(
            tool_call=ReadFeatureCatalogToolCall(
                request=ReadFeatureCatalogRequest(section="templates")
            )
        )
    )
    assert catalog.payload.entries[0].entry_id == "matched_feature_ablation_v1"

    calculation = gateway.execute(
        ToolCallAction(
            tool_call=CalculateToolCall(
                request=CalculateRequest(
                    operation="difference",
                    inputs=(
                        CalculationInput(source_id="1" * 64, value=3.0),
                        CalculationInput(source_id="2" * 64, value=1.0),
                    ),
                    rounding_policy="decimal_12",
                )
            )
        )
    )
    assert calculation.payload.calculation.output == 2.0
    assert calculation.payload.calculation.tool_result_hash == calculation.result_hash


def test_tool_gateway_rejects_out_of_bundle_assets_and_invalid_arithmetic(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
    research_agent_bundle_factory: Callable[..., tuple[Path, object]],
) -> None:
    gateway = _gateway(tmp_path, research_study_definition, research_agent_bundle_factory)
    with pytest.raises(ValueError, match="outside"):
        gateway.execute(
            ToolCallAction(
                tool_call=SearchEvidenceToolCall(
                    request=SearchEvidenceRequest(
                        query="demand",
                        purpose="support",
                        asset_ids=("asset-z",),
                        result_limit=1,
                    )
                )
            )
        )
    with pytest.raises(ValueError, match="zero"):
        gateway.execute(
            ToolCallAction(
                tool_call=CalculateToolCall(
                    request=CalculateRequest(
                        operation="ratio",
                        inputs=(
                            CalculationInput(source_id="1" * 64, value=1.0),
                            CalculationInput(source_id="2" * 64, value=0.0),
                        ),
                        rounding_policy="decimal_12",
                    )
                )
            )
        )
