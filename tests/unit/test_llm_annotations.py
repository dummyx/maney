from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import nlp_trader.nlp.llm_annotations as llm_module
from nlp_trader.nlp.llm_annotations import (
    AssetCandidate,
    CachedLocalLLMAnnotator,
    GenerationRequest,
    GenerationResponse,
    LLMAnnotationConfig,
    build_annotation_request,
)
from nlp_trader.schemas import TextItem


def _item(*, item_id: str = "item-1", body: str = "Issuer A raised guidance.") -> TextItem:
    published = datetime(2024, 1, 2, 13, 0, tzinfo=UTC)
    return TextItem(
        item_id=item_id,
        source="licensed-fixture",
        source_type="news",
        language="en",
        title="Issuer A and Issuer B update",
        body=body,
        published_at=published,
        vendor_received_at=published,
        ingested_at=published,
        available_at=published,
        license_or_terms_ref="redistributable-test-fixture",
    )


def _candidates() -> tuple[AssetCandidate, ...]:
    return (
        AssetCandidate("asset_a", "AAA", "Issuer A"),
        AssetCandidate("asset_b", "BBB", "Issuer B"),
    )


def _config(tmp_path: Path, *, batch_size: int = 2) -> LLMAnnotationConfig:
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    (model_path / "config.json").write_text("{}\n", encoding="utf-8")
    return LLMAnnotationConfig(
        model_path=model_path,
        model_id="test-causal-lm",
        model_revision="test-revision",
        model_license_or_terms_ref="redistributable-test-fixture",
        prompt_version="entity-event-v1",
        schema_version="entity-event-v1",
        cache_dir=tmp_path / "cache",
        batch_size=batch_size,
    )


def _valid_payload() -> dict[str, object]:
    return {
        "annotations": [
            {
                "asset_id": "asset_a",
                "stance_label": "positive",
                "stance_confidence": 0.9,
                "uncertainty": 0.1,
                "primary_event_type": "guidance",
                "event_confidence": 0.8,
                "evidence_span_ids": ["S2"],
                "abstain_reason": None,
            },
            {
                "asset_id": "asset_b",
                "stance_label": "negative",
                "stance_confidence": 0.7,
                "uncertainty": 0.2,
                "primary_event_type": None,
                "event_confidence": 0.0,
                "evidence_span_ids": ["S1"],
                "abstain_reason": None,
            },
        ]
    }


def test_request_contains_only_current_source_evidence_and_canonical_candidates() -> None:
    item = _item(body="Issuer A raised guidance. Issuer B warned on demand.")
    request = build_annotation_request(item, reversed(_candidates()))

    assert [candidate.asset_id for candidate in request.candidates] == ["asset_a", "asset_b"]
    assert [span.span_id for span in request.evidence_spans] == ["S1", "S2", "S3"]
    assert "Issuer A raised guidance." in request.prompt
    assert "Issuer B warned on demand." in request.prompt
    assert item.item_id not in request.prompt
    assert item.available_at.isoformat() not in request.prompt
    assert "published_at" not in request.prompt
    assert "available_at" not in request.prompt


def test_entity_specific_response_is_strictly_parsed_cached_and_replayed(tmp_path: Path) -> None:
    request = build_annotation_request(_item(), _candidates())
    calls = 0

    def generator(values: list[GenerationRequest]) -> list[GenerationResponse]:
        nonlocal calls
        calls += 1
        return [
            GenerationResponse(
                request_id=value.request_id,
                generated_text=json.dumps(_valid_payload()),
            )
            for value in values
        ]

    config = _config(tmp_path)
    first = CachedLocalLLMAnnotator(config, generator=generator).annotate([request])
    second = CachedLocalLLMAnnotator(
        config,
        generator=lambda _values: (_ for _ in ()).throw(AssertionError("cache miss")),
    ).annotate([request])

    assert first == second
    assert calls == 1
    assert [(row.asset_id, row.stance_label) for row in first[0].annotations] == [
        ("asset_a", "positive"),
        ("asset_b", "negative"),
    ]
    assert len(list(config.cache_dir.glob("*.json"))) == 1


def test_identical_canonical_text_is_generated_once_across_item_ids(tmp_path: Path) -> None:
    requests = [
        build_annotation_request(_item(item_id=item_id), _candidates())
        for item_id in ("copy-1", "copy-2")
    ]
    generated_batch_sizes: list[int] = []

    def generator(values: list[GenerationRequest]) -> list[GenerationResponse]:
        generated_batch_sizes.append(len(values))
        return [
            GenerationResponse(
                request_id=value.request_id,
                generated_text=json.dumps(_valid_payload()),
            )
            for value in values
        ]

    responses = CachedLocalLLMAnnotator(
        _config(tmp_path),
        generator=generator,
    ).annotate(requests)

    assert generated_batch_sizes == [1]
    assert [response.item_id for response in responses] == ["copy-1", "copy-2"]


@pytest.mark.parametrize(
    "generated_text",
    [
        "```json\n{}\n```",
        '{"annotations":[],"annotations":[]}',
        json.dumps(
            {
                "annotations": [
                    {
                        **_valid_payload()["annotations"][0],  # type: ignore[index]
                        "asset_id": "unknown_asset",
                    },
                    _valid_payload()["annotations"][1],  # type: ignore[index]
                ]
            }
        ),
        json.dumps(
            {
                "annotations": [
                    {
                        **_valid_payload()["annotations"][0],  # type: ignore[index]
                        "evidence_span_ids": ["S999"],
                    },
                    _valid_payload()["annotations"][1],  # type: ignore[index]
                ]
            }
        ),
    ],
)
def test_invalid_or_ungrounded_output_fails_without_writing_cache(
    tmp_path: Path,
    generated_text: str,
) -> None:
    request = build_annotation_request(_item(), _candidates())
    config = _config(tmp_path)
    engine = CachedLocalLLMAnnotator(
        config,
        generator=lambda values: [
            GenerationResponse(request_id=values[0].request_id, generated_text=generated_text)
        ],
    )

    with pytest.raises(ValueError, match="strict annotation JSON|exactly match|unknown evidence"):
        engine.annotate([request])
    assert not list(config.cache_dir.glob("*.json"))


def test_one_invalid_response_prevents_all_new_batch_cache_writes(tmp_path: Path) -> None:
    requests = [
        build_annotation_request(
            _item(item_id=f"item-{index}", body=f"Issuer A update number {index}."),
            _candidates(),
        )
        for index in range(2)
    ]
    config = _config(tmp_path, batch_size=1)
    call = 0

    def generator(values: list[GenerationRequest]) -> list[GenerationResponse]:
        nonlocal call
        call += 1
        payload = json.dumps(_valid_payload()) if call == 1 else "not-json"
        return [GenerationResponse(request_id=values[0].request_id, generated_text=payload)]

    with pytest.raises(ValueError, match="strict annotation JSON"):
        CachedLocalLLMAnnotator(config, generator=generator).annotate(requests)
    assert not list(config.cache_dir.glob("*.json"))


def test_input_too_long_becomes_explicit_per_candidate_abstention(tmp_path: Path) -> None:
    request = build_annotation_request(_item(), _candidates())
    config = _config(tmp_path)
    engine = CachedLocalLLMAnnotator(
        config,
        generator=lambda values: [
            GenerationResponse(request_id=values[0].request_id, input_too_long=True)
        ],
    )

    response = engine.annotate([request])[0]

    assert [annotation.stance_label for annotation in response.annotations] == [
        "abstain",
        "abstain",
    ]
    assert {annotation.abstain_reason for annotation in response.annotations} == {"input_too_long"}
    cache_record = engine.cache_record_for(request)
    assert cache_record["generation"]["input_too_long"] is True


def test_cache_identity_changes_with_exact_model_bytes_and_schema(tmp_path: Path) -> None:
    request = build_annotation_request(_item(), _candidates())
    config = _config(tmp_path)
    first = CachedLocalLLMAnnotator(config, generator=lambda _values: [])
    first_key = first.cache_key_for(request)

    (config.model_path / "config.json").write_text('{"changed":true}\n', encoding="utf-8")
    changed_model = CachedLocalLLMAnnotator(config, generator=lambda _values: [])
    assert changed_model.cache_key_for(request) != first_key

    changed_schema_config = replace(config, schema_version="entity-event-v2")
    changed_schema = CachedLocalLLMAnnotator(
        changed_schema_config,
        generator=lambda _values: [],
    )
    assert changed_schema.cache_key_for(request) != changed_model.cache_key_for(request)


def test_default_backend_loads_local_causal_lm_lazily_and_generates_strict_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[tuple[str, str, dict[str, object]]] = []

    class FakeTensor:
        def __init__(self, values: list[int] | list[list[int]]) -> None:
            self.values = values

        @property
        def shape(self) -> tuple[int, int]:
            assert self.values and isinstance(self.values[0], list)
            rows = self.values
            return (len(rows), len(rows[0]))

        def __getitem__(self, index: slice) -> FakeTensor:
            assert self.values and isinstance(self.values[0], int)
            values = self.values
            return FakeTensor(values[index])

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def tolist(self) -> list[int]:
            assert not self.values or isinstance(self.values[0], int)
            return list(self.values)

    class FakeBatch(dict[str, FakeTensor]):
        def to(self, _device: str) -> FakeBatch:
            return self

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2
        model_max_length = 1024
        padding_side = "right"

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> FakeTokenizer:
            loads.append(("tokenizer", path, kwargs))
            return cls()

        def __call__(self, value: str | list[str], **_kwargs: object) -> object:
            if isinstance(value, str):
                return {"input_ids": [1] * 12}
            return FakeBatch({"input_ids": FakeTensor([[1] * 12 for _ in value])})

        def decode(self, _values: FakeTensor, **_kwargs: object) -> str:
            return json.dumps(_valid_payload())

    class FakeModel:
        class Config:
            max_position_embeddings = 1024

        config = Config()
        device: str | None = None
        evaluated = False

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> FakeModel:
            loads.append(("model", path, kwargs))
            return cls()

        def to(self, device: str) -> None:
            self.device = device

        def eval(self) -> None:
            self.evaluated = True

        def generate(self, **kwargs: object) -> list[FakeTensor]:
            input_ids = kwargs["input_ids"]
            assert isinstance(input_ids, FakeTensor)
            width = input_ids.shape[1]
            return [FakeTensor([1] * width + [9, 2])]

    class InferenceMode:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeTorch:
        seed: int | None = None

        @classmethod
        def manual_seed(cls, seed: int) -> None:
            cls.seed = seed

        @staticmethod
        def inference_mode() -> InferenceMode:
            return InferenceMode()

    class FakeTransformers:
        AutoTokenizer = FakeTokenizer
        AutoModelForCausalLM = FakeModel

    def fake_import(name: str) -> object:
        if name == "transformers":
            return FakeTransformers
        if name == "torch":
            return FakeTorch
        raise ImportError(name)

    monkeypatch.setattr(llm_module, "import_module", fake_import)
    monkeypatch.setattr(llm_module, "get_torch_device", lambda: "cpu")
    config = _config(tmp_path)
    response = CachedLocalLLMAnnotator(config).annotate(
        [build_annotation_request(_item(), _candidates())]
    )[0]

    assert [annotation.stance_label for annotation in response.annotations] == [
        "positive",
        "negative",
    ]
    assert FakeTorch.seed == config.seed
    assert [kind for kind, _, _ in loads] == ["tokenizer", "model"]
    assert all(
        kwargs == {"local_files_only": True, "trust_remote_code": False} for _, _, kwargs in loads
    )
