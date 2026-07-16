from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from pydantic import ValidationError

from nlp_trader.immutable.append import (
    SafeFileError,
    read_bytes_no_follow,
    write_bytes_exclusive_durable,
)
from nlp_trader.research_agents.contracts import AgentArtifactManifest, canonical_json


class AgentArtifactError(ValueError):
    """Raised when an agent artifact root or manifest is unsafe or inconsistent."""


class _DuplicateJsonKeyError(ValueError):
    pass


def ensure_agent_artifact_root(path: str | Path) -> Path:
    """Validate or create one private absolute local artifact root without following a symlink."""

    root = Path(path).expanduser()
    if not root.is_absolute():
        raise AgentArtifactError("research-agent artifact root must be absolute")
    root = root.parent.resolve(strict=False) / root.name
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink():
            raise AgentArtifactError("research-agent artifact root cannot be a symlink")
        metadata = root.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            raise AgentArtifactError("research-agent artifact root must be a directory")
        os.chmod(root, 0o700)
    except AgentArtifactError:
        raise
    except OSError as exc:
        raise AgentArtifactError("research-agent artifact root is unavailable") from exc
    return root


def write_agent_manifest_exclusive(
    path: str | Path,
    manifest: AgentArtifactManifest,
) -> Path:
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        raise AgentArtifactError("agent manifest path must be absolute")
    encoded = (manifest.canonical_json() + "\n").encode("utf-8")
    try:
        write_bytes_exclusive_durable(destination, encoded)
    except FileExistsError as exc:
        raise AgentArtifactError("agent manifest already exists") from exc
    except (SafeFileError, ValueError) as exc:
        raise AgentArtifactError("agent manifest cannot be written safely") from exc
    return destination


def load_agent_manifest(path: str | Path) -> AgentArtifactManifest:
    try:
        encoded = read_bytes_no_follow(Path(path).expanduser().absolute())
    except (FileNotFoundError, SafeFileError, ValueError) as exc:
        raise AgentArtifactError("agent manifest cannot be read safely") from exc
    if encoded is None:  # pragma: no cover - missing_ok is false
        raise AgentArtifactError("agent manifest does not exist")
    try:
        raw = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AgentArtifactError("agent manifest is not UTF-8") from exc
    if not raw.endswith("\n") or raw.count("\n") != 1:
        raise AgentArtifactError("agent manifest must contain one complete canonical JSON line")
    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (_DuplicateJsonKeyError, json.JSONDecodeError, ValueError) as exc:
        raise AgentArtifactError("agent manifest is not strict JSON") from exc
    if not isinstance(parsed, dict) or raw != canonical_json(parsed) + "\n":
        raise AgentArtifactError("agent manifest is not canonical JSON")
    try:
        manifest = AgentArtifactManifest.model_validate_json(raw)
    except ValidationError as exc:
        raise AgentArtifactError("agent manifest does not satisfy its strict contract") from exc
    if raw != manifest.canonical_json() + "\n":
        raise AgentArtifactError("agent manifest does not match canonical typed content")
    return manifest


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
