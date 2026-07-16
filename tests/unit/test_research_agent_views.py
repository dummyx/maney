from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from nlp_trader.research_agents.artifacts import AgentArtifactError
from nlp_trader.research_agents.contracts import StudyDefinition
from nlp_trader.research_agents.views import load_development_view_bundle


def test_sealed_bundle_round_trip_contains_only_fixed_hash_bound_files(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
    research_agent_bundle_factory: Callable[..., tuple[Path, object]],
) -> None:
    root, manifest = research_agent_bundle_factory(tmp_path, research_study_definition)
    loaded = load_development_view_bundle(root)

    assert loaded.manifest == manifest
    assert loaded.development_view.study_id == research_study_definition.study_id
    assert {path.name for path in root.iterdir()} == {
        "bundle.manifest.json",
        "development_view.json",
        "feature_catalog.json",
        "evidence_snapshot.jsonl",
        "evidence_index.json",
    }
    model_visible = b"".join(
        (root / name).read_bytes()
        for name in (
            "development_view.json",
            "feature_catalog.json",
            "evidence_snapshot.jsonl",
            "evidence_index.json",
        )
    )
    assert b"final_holdout" not in model_visible
    assert b"/Users/" not in model_visible


def test_sealed_bundle_rejects_tamper_and_symlink_root(
    tmp_path: Path,
    research_study_definition: StudyDefinition,
    research_agent_bundle_factory: Callable[..., tuple[Path, object]],
) -> None:
    root, _ = research_agent_bundle_factory(tmp_path, research_study_definition)
    path = root / "development_view.json"
    path.write_bytes(path.read_bytes().replace(b'"value":0.0', b'"value":1.0'))
    with pytest.raises(AgentArtifactError, match="hash or size"):
        load_development_view_bundle(root)

    if hasattr(os, "symlink"):
        link = (tmp_path / "bundle-link").resolve()
        link.symlink_to(root, target_is_directory=True)
        with pytest.raises(AgentArtifactError, match="symlink"):
            load_development_view_bundle(link)
