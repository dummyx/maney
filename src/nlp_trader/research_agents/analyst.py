from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import TypeAdapter, ValidationError

from nlp_trader.research_agents.contracts import ResearchAgentAction
from nlp_trader.research_agents.runtime import (
    AgentGenerationRecord,
    AgentGenerationRequest,
    ResearchAgentGenerationRuntime,
)

_ACTION_ADAPTER: TypeAdapter[ResearchAgentAction] = TypeAdapter(ResearchAgentAction)


class AgentActionParseError(ValueError):
    """Raised when one generated output is not exactly one strict typed action."""


class _DuplicateKeyError(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def parse_agent_action(raw: str) -> ResearchAgentAction:
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant is forbidden: {value}")
            ),
        )
    except (_DuplicateKeyError, json.JSONDecodeError, ValueError) as exc:
        raise AgentActionParseError("model output is not one strict JSON object") from exc
    if not isinstance(parsed, dict):
        raise AgentActionParseError("model output must contain one JSON object")
    try:
        action: ResearchAgentAction = _ACTION_ADAPTER.validate_json(raw, strict=True)
        return action
    except ValidationError as exc:
        raise AgentActionParseError("model output violates the research action schema") from exc


@dataclass(frozen=True, slots=True)
class AnalystStepResult:
    generation: AgentGenerationRecord
    action: ResearchAgentAction | None
    parse_error: str | None


class BoundedResearchAnalyst:
    def __init__(self, runtime: ResearchAgentGenerationRuntime) -> None:
        self._runtime = runtime

    def generate_step(self, request: AgentGenerationRequest) -> AnalystStepResult:
        generation = self._runtime.generate(request)
        if generation.input_too_long:
            return AnalystStepResult(
                generation=generation,
                action=None,
                parse_error="generation input exceeded the configured context",
            )
        if generation.output_truncated:
            return AnalystStepResult(
                generation=generation,
                action=None,
                parse_error="generation reached the configured output limit",
            )
        assert generation.generated_text is not None
        try:
            action = parse_agent_action(generation.generated_text)
        except AgentActionParseError as exc:
            return AnalystStepResult(generation=generation, action=None, parse_error=str(exc))
        return AnalystStepResult(generation=generation, action=action, parse_error=None)
