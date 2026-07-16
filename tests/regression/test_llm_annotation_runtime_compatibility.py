from __future__ import annotations

from nlp_trader.nlp import llm_annotations
from nlp_trader.nlp.local_generation import GenerationRequest, GenerationResponse


def test_annotation_generation_contracts_remain_reexported() -> None:
    assert llm_annotations.GenerationRequest is GenerationRequest
    assert llm_annotations.GenerationResponse is GenerationResponse
