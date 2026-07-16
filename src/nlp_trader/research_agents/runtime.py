from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field, model_validator

from nlp_trader.nlp.local_generation import GenerationRequest, GenerationResponse, RawGenerator
from nlp_trader.research_agents.contracts import (
    Sha256,
    StrictModel,
    canonical_json,
    content_sha256,
)


class AgentGenerationRequest(StrictModel):
    artifact_schema_version: Literal["agent-generation-request-v1"] = "agent-generation-request-v1"
    request_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    study_id: Sha256
    attempt_id: Sha256
    bundle_id: Sha256
    round_index: int = Field(ge=0)
    prompt: str = Field(min_length=1, max_length=500_000, strict=True)
    action_schema_hash: Sha256
    transcript_hash: Sha256

    @model_validator(mode="after")
    def validate_identity(self) -> AgentGenerationRequest:
        expected = content_sha256(self.model_dump(mode="json", exclude={"request_id"}))
        if self.request_id and self.request_id != expected:
            raise ValueError("agent generation request_id does not match canonical content")
        if not self.request_id:
            object.__setattr__(self, "request_id", expected)
        return self


class AgentGenerationRecord(StrictModel):
    artifact_schema_version: Literal["agent-generation-record-v1"] = "agent-generation-record-v1"
    request: AgentGenerationRequest
    generated_text: str | None
    input_too_long: bool
    output_truncated: bool
    input_token_count: int | None = Field(default=None, ge=0)
    output_token_count: int | None = Field(default=None, ge=0)
    generation_latency_seconds: float | None = Field(default=None, ge=0.0)
    output_tokens_per_second: float | None = Field(default=None, ge=0.0)
    context_fit: bool

    @model_validator(mode="after")
    def validate_record(self) -> AgentGenerationRecord:
        if self.context_fit == self.input_too_long:
            raise ValueError("context_fit must be the inverse of input_too_long")
        if self.input_too_long and self.generated_text is not None:
            raise ValueError("input-too-long records cannot retain generated text")
        return self

    def response(self) -> GenerationResponse:
        return GenerationResponse(
            request_id=self.request.request_id,
            generated_text=self.generated_text,
            input_too_long=self.input_too_long,
            output_truncated=self.output_truncated,
            input_token_count=self.input_token_count,
            output_token_count=self.output_token_count,
            generation_latency_seconds=self.generation_latency_seconds,
        )


def build_agent_generation_request(
    *,
    study_id: str,
    attempt_id: str,
    bundle_id: str,
    round_index: int,
    prompt: str,
    action_schema: object,
    transcript: object,
) -> AgentGenerationRequest:
    return AgentGenerationRequest(
        study_id=study_id,
        attempt_id=attempt_id,
        bundle_id=bundle_id,
        round_index=round_index,
        prompt=prompt,
        action_schema_hash=content_sha256(action_schema),
        transcript_hash=content_sha256(transcript),
    )


def _record(request: AgentGenerationRequest, response: GenerationResponse) -> AgentGenerationRecord:
    if response.request_id != request.request_id:
        raise ValueError("generation response does not match its exact agent request")
    throughput: float | None = None
    if (
        response.output_token_count is not None
        and response.generation_latency_seconds is not None
        and response.generation_latency_seconds > 0.0
    ):
        throughput = response.output_token_count / response.generation_latency_seconds
        if not math.isfinite(throughput):
            raise ValueError("generation throughput must be finite")
    return AgentGenerationRecord(
        request=request,
        generated_text=response.generated_text,
        input_too_long=response.input_too_long,
        output_truncated=response.output_truncated,
        input_token_count=response.input_token_count,
        output_token_count=response.output_token_count,
        generation_latency_seconds=response.generation_latency_seconds,
        output_tokens_per_second=throughput,
        context_fit=not response.input_too_long,
    )


@dataclass(slots=True)
class ResearchAgentGenerationRuntime:
    """Attempt-scoped generation transport. It deliberately has no cross-run cache."""

    generator: RawGenerator
    _records: list[AgentGenerationRecord] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._records = []

    @property
    def records(self) -> tuple[AgentGenerationRecord, ...]:
        return tuple(self._records)

    def generate(self, request: AgentGenerationRequest) -> AgentGenerationRecord:
        responses = self.generator(
            [GenerationRequest(request_id=request.request_id, prompt=request.prompt)]
        )
        if len(responses) != 1:
            raise ValueError("agent generator must return exactly one response")
        record = _record(request, responses[0])
        self._records.append(record)
        return record

    @staticmethod
    def replay(
        request: AgentGenerationRequest, stored_record: AgentGenerationRecord
    ) -> GenerationResponse:
        if stored_record.request != request:
            raise ValueError("stored generation record does not match the replay request")
        return stored_record.response()


def canonical_generation_record(record: AgentGenerationRecord) -> str:
    return canonical_json(record.model_dump(mode="json"))
