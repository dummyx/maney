from __future__ import annotations

import pytest

from nlp_trader.research_agents.audit import (
    ProposalQualityScore,
    compare_proposal_quality,
)


def test_blinded_quality_rubric_compares_against_control_not_acceptance_rate() -> None:
    analyst = ProposalQualityScore(
        blinded_label="sample-x",
        hypothesis_testability=4,
        evidence_grounding=3,
        counterevidence_quality=3,
        falsification_quality=4,
        control_completeness=4,
        point_in_time_safety=4,
    )
    control = ProposalQualityScore(
        blinded_label="sample-y",
        hypothesis_testability=2,
        evidence_grounding=1,
        counterevidence_quality=1,
        falsification_quality=2,
        control_completeness=4,
        point_in_time_safety=4,
    )

    comparison = compare_proposal_quality(analyst, control)

    assert comparison.total_difference == analyst.total - control.total
    assert "acceptance rate is not" in comparison.interpretation
    with pytest.raises(ValueError, match="distinct"):
        compare_proposal_quality(analyst, control.model_copy(update={"blinded_label": "sample-x"}))
