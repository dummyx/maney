from __future__ import annotations

from nlp_trader.nlp.local_generation import GenerationRequest, GenerationResponse
from nlp_trader.research_agents.runtime import (
    AgentGenerationRequest,
    ResearchAgentGenerationRuntime,
    build_agent_generation_request,
)


def _request() -> AgentGenerationRequest:
    return build_agent_generation_request(
        study_id="1" * 64,
        attempt_id="2" * 64,
        bundle_id="3" * 64,
        round_index=0,
        prompt="Return one strict action object.",
        action_schema={"type": "object"},
        transcript=[],
    )


def test_agent_generation_is_attempt_scoped_and_replay_never_calls_model() -> None:
    calls: list[GenerationRequest] = []

    def generator(requests: list[GenerationRequest]) -> list[GenerationResponse]:
        calls.extend(requests)
        return [
            GenerationResponse(
                request_id=requests[0].request_id,
                generated_text='{"action_type":"abstention"}',
                input_token_count=10,
                output_token_count=5,
                generation_latency_seconds=0.5,
            )
        ]

    request = _request()
    runtime = ResearchAgentGenerationRuntime(generator)
    record = runtime.generate(request)
    replayed = runtime.replay(request, record)

    assert len(calls) == 1
    assert replayed.generated_text == record.generated_text
    assert record.output_tokens_per_second == 10.0
    assert runtime.records == (record,)


def test_agent_request_identity_binds_schema_and_transcript() -> None:
    first = _request()
    changed = build_agent_generation_request(
        study_id="1" * 64,
        attempt_id="2" * 64,
        bundle_id="3" * 64,
        round_index=0,
        prompt="Return one strict action object.",
        action_schema={"type": "object", "required": ["action_type"]},
        transcript=[],
    )

    assert first.request_id != changed.request_id
