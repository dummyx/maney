from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nlp_trader.config import ResearchConfig

_PACKAGE_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_parquet_directory(
    path: Path,
) -> tuple[str, int, list[dict[str, Any]]]:
    """Hash a partitioned Parquet input from relative names and exact file hashes."""

    digest = hashlib.sha256()
    byte_count = 0
    files = sorted(path.rglob("*.parquet"))
    manifest: list[dict[str, Any]] = []
    for file_path in files:
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        file_digest = sha256_file(file_path)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_digest))
        size = file_path.stat().st_size
        byte_count += size
        manifest.append(
            {
                "relative_path": relative.decode("utf-8"),
                "sha256": file_digest,
                "bytes": size,
            }
        )
    return digest.hexdigest(), byte_count, manifest


def sha256_directory(path: Path) -> tuple[str, int, list[dict[str, Any]]]:
    """Hash every regular file in a local model directory by name and exact bytes."""

    digest = hashlib.sha256()
    byte_count = 0
    manifest: list[dict[str, Any]] = []
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        file_digest = sha256_file(file_path)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_digest))
        size = file_path.stat().st_size
        byte_count += size
        manifest.append(
            {
                "relative_path": relative.decode("utf-8"),
                "sha256": file_digest,
                "bytes": size,
            }
        )
    return digest.hexdigest(), byte_count, manifest


def input_manifest(config: ResearchConfig) -> list[dict[str, Any]]:
    """Describe configured local inputs without embedding their contents."""

    entries: list[dict[str, Any]] = []
    for role in (
        "assets",
        "market_bars",
        "text_items",
        "fundamentals",
        "earnings_calendar",
        "corporate_actions",
    ):
        path = getattr(config.paths, role)
        if path is None:
            continue
        exists = path.is_file() or (path.is_dir() and any(path.rglob("*.parquet")))
        entry: dict[str, Any] = {"role": role, "path": str(path), "exists": exists}
        if path.is_file():
            stat = path.stat()
            entry.update({"sha256": sha256_file(path), "bytes": stat.st_size})
        elif path.is_dir():
            digest, byte_count, files = sha256_parquet_directory(path)
            entry.update(
                {
                    "sha256": digest,
                    "bytes": byte_count,
                    "file_count": len(files),
                    "input_kind": "partitioned_parquet_directory",
                    "files": files,
                }
            )
        entries.append(entry)
    if config.llm_annotations.enabled:
        model_path = config.paths.llm_model
        if model_path is None:
            raise ValueError("enabled LLM annotations require a configured local model path")
        exists = model_path.is_dir() and any(
            candidate.is_file() for candidate in model_path.rglob("*")
        )
        entry = {
            "role": "llm_model",
            "path": str(model_path),
            "exists": exists,
            "input_kind": "local_model_directory",
            "model_id": config.llm_annotations.model_id,
            "model_revision": config.llm_annotations.model_revision,
            "license_or_terms_ref": config.llm_annotations.model_license_or_terms_ref,
        }
        if exists:
            digest, byte_count, files = sha256_directory(model_path)
            entry.update(
                {
                    "sha256": digest,
                    "bytes": byte_count,
                    "file_count": len(files),
                    "files": files,
                }
            )
        entries.append(entry)
    return entries


def code_version(repo: Path) -> dict[str, Any]:
    """Return Git provenance without requiring a commit to exist."""

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )

    revision = git("rev-parse", "--verify", "HEAD")
    status = git("status", "--porcelain")
    return {
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_available": revision.returncode == 0 or status.returncode == 0,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


@dataclass(frozen=True, slots=True)
class RunPaths:
    interim: Path
    processed: Path
    models: Path
    reports: Path


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    created_at: datetime
    config: ResearchConfig
    paths: RunPaths
    inputs: tuple[dict[str, Any], ...]
    code: dict[str, Any]


def _fingerprint(config: ResearchConfig, inputs: list[dict[str, Any]]) -> str:
    payload = {
        "config_hash": config.content_hash(),
        "inputs": [
            {"role": entry["role"], "sha256": entry.get("sha256"), "exists": entry["exists"]}
            for entry in inputs
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def create_run_context(
    config: ResearchConfig,
    *,
    now: datetime | None = None,
    run_id: str | None = None,
) -> RunContext:
    """Create unique run directories and an immutable configuration snapshot."""

    created_at = (now or datetime.now(UTC)).astimezone(UTC)
    inputs = input_manifest(config)
    base_id = run_id or (
        created_at.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + _fingerprint(config, inputs)
    )
    candidate = base_id
    suffix = 0
    while True:
        paths = RunPaths(
            interim=config.paths.interim_dir / candidate,
            processed=config.paths.processed_dir / candidate,
            models=config.paths.models_dir / candidate,
            reports=config.paths.reports_dir / candidate,
        )
        run_directories = (paths.interim, paths.processed, paths.models, paths.reports)
        if not any(path.exists() for path in run_directories):
            break
        if run_id is not None:
            raise FileExistsError(f"run already exists: {candidate}")
        suffix += 1
        candidate = f"{base_id}-{suffix:02d}"
    for path in (paths.interim, paths.processed, paths.models, paths.reports):
        path.mkdir(parents=True, exist_ok=False)
    context = RunContext(
        run_id=candidate,
        created_at=created_at,
        config=config,
        paths=paths,
        inputs=tuple(inputs),
        code=code_version(_PACKAGE_REPOSITORY_ROOT),
    )
    snapshot = config.model_dump(mode="json", exclude={"path"})
    _write_json_exclusive(paths.reports / "config.snapshot.json", snapshot)
    _write_json_exclusive(
        paths.reports / "run.initial.json",
        {
            "run_id": context.run_id,
            "created_at": created_at.isoformat().replace("+00:00", "Z"),
            "config_hash": config.content_hash(),
            "code_version": context.code,
            "data_manifest": inputs,
            "status": "running",
        },
    )
    return context


def artifact_manifest(paths: RunPaths) -> list[dict[str, Any]]:
    """Hash all materialized run artifacts except the final manifest itself."""

    artifacts: list[dict[str, Any]] = []
    roots = {
        "interim": paths.interim,
        "processed": paths.processed,
        "models": paths.models,
        "reports": paths.reports,
    }
    for area, root in roots.items():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == "run.final.json":
                continue
            artifacts.append(
                {
                    "area": area,
                    "path": str(path.relative_to(root.parent)),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    return artifacts


def finalize_run(
    context: RunContext,
    *,
    universe: list[str],
    period: dict[str, str | None],
    metrics: dict[str, Any],
    known_limitations: list[str],
    next_questions: list[str],
    stage: str = "report",
) -> Path:
    """Write the immutable final research manifest for a completed run."""

    path = context.paths.reports / "run.final.json"
    payload = {
        "run_id": context.run_id,
        "created_at": context.created_at.isoformat().replace("+00:00", "Z"),
        "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "complete",
        "completed_stage": stage,
        "code_version": context.code,
        "config_hash": context.config.content_hash(),
        "data_manifest": list(context.inputs),
        "artifact_manifest": artifact_manifest(context.paths),
        "universe": sorted(set(universe)),
        "period": period,
        "rebalance_frequency": context.config.backtest.rebalance_frequency,
        "feature_set_version": context.config.features.feature_set_version,
        "label_version": context.config.features.label_version,
        "model_version": context.config.features.model_version,
        "cost_model": {
            "commission_bps": context.config.backtest.commission_bps,
            "half_spread_bps": context.config.backtest.half_spread_bps,
            "base_slippage_bps": context.config.backtest.slippage_bps,
            "volatility_slippage_multiplier": (
                context.config.backtest.volatility_slippage_multiplier
            ),
            "participation_slippage_bps": context.config.backtest.participation_slippage_bps,
            "market_impact_multiplier": context.config.backtest.market_impact_multiplier,
            "borrow_bps_per_year": context.config.backtest.borrow_bps_per_year,
        },
        "constraints": context.config.backtest.model_dump(mode="json"),
        "metrics": metrics,
        "known_limitations": known_limitations,
        "next_questions": next_questions,
    }
    _write_json_exclusive(path, payload)
    return path


def fail_run(context: RunContext, error: Exception, *, stage: str) -> Path:
    """Persist a sanitized failure record without mutating successful artifacts."""

    path = context.paths.reports / "run.failed.json"
    _write_json_exclusive(
        path,
        {
            "run_id": context.run_id,
            "failed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "status": "failed",
            "failed_stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "artifact_manifest": artifact_manifest(context.paths),
        },
    )
    return path
