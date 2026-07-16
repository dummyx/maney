from __future__ import annotations

from typing import Any, Literal, Self, cast

from pydantic import Field, field_validator, model_validator

from nlp_trader.research_agents.contracts import (
    Identifier,
    NonBlankText,
    StrictModel,
    canonical_json,
    content_sha256,
)

CatalogSection = Literal[
    "features",
    "models",
    "benchmarks",
    "selectors",
    "metrics",
    "controls",
    "templates",
]


class CatalogEntry(StrictModel):
    entry_id: Identifier
    version: Identifier
    definition: NonBlankText


class FeatureCatalog(StrictModel):
    """Outcome-free catalog of identifiers available inside one sealed bundle."""

    artifact_schema_version: Literal["research-feature-catalog-v1"] = "research-feature-catalog-v1"
    catalog_id: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    features: tuple[CatalogEntry, ...]
    models: tuple[CatalogEntry, ...]
    benchmarks: tuple[CatalogEntry, ...]
    selectors: tuple[CatalogEntry, ...]
    metrics: tuple[CatalogEntry, ...]
    controls: tuple[CatalogEntry, ...]
    templates: tuple[CatalogEntry, ...]

    @field_validator(
        "features",
        "models",
        "benchmarks",
        "selectors",
        "metrics",
        "controls",
        "templates",
    )
    @classmethod
    def validate_entries(
        cls, values: tuple[CatalogEntry, ...], info: Any
    ) -> tuple[CatalogEntry, ...]:
        ids = tuple(value.entry_id for value in values)
        if len(ids) != len(set(ids)):
            raise ValueError(f"catalog {info.field_name} entry IDs must be unique")
        if ids != tuple(sorted(ids)):
            raise ValueError(f"catalog {info.field_name} entries must be sorted by entry_id")
        return values

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected = self.computed_catalog_id()
        if self.catalog_id and self.catalog_id != expected:
            raise ValueError("catalog_id does not match canonical catalog content")
        if not self.catalog_id:
            object.__setattr__(self, "catalog_id", expected)
        return self

    def section(self, name: CatalogSection) -> tuple[CatalogEntry, ...]:
        return cast(tuple[CatalogEntry, ...], getattr(self, name))

    def content_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"catalog_id"})

    def computed_catalog_id(self) -> str:
        return content_sha256(self.content_payload())

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))
