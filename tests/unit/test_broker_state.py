from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from nlp_trader.broker.state import (
    MAX_LOCK_OWNER_BYTES,
    KabuSStateLockError,
    KabuSStatePathError,
    KabuSStatePaths,
    _current_user_state_root,
    advisory_file_lock,
)


def test_explicit_state_root_exposes_one_fixed_set_of_paths(tmp_path: Path) -> None:
    paths = KabuSStatePaths(tmp_path / "state" / ".." / "state")

    assert paths.root == (tmp_path / "state").resolve()
    assert paths.audit_ledger_path == paths.root / "audit.jsonl"
    assert paths.kill_switch_path == paths.root / "KILL_SWITCH"
    assert paths.operation_lock_path == paths.root / "operation.lock"
    assert (
        len(
            {
                paths.audit_ledger_path,
                paths.kill_switch_path,
                paths.operation_lock_path,
            }
        )
        == 3
    )


def test_explicit_state_root_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="absolute"):
        KabuSStatePaths(Path("relative/state"))


def test_current_user_state_root_follows_platform_conventions(tmp_path: Path) -> None:
    home = tmp_path / "home"
    local_app_data = tmp_path / "local-app-data"
    xdg_state = tmp_path / "xdg-state"

    assert (
        _current_user_state_root(
            platform="win32",
            environ={"LOCALAPPDATA": str(local_app_data)},
            home=home,
        )
        == local_app_data / "nlp-trader" / "kabus"
    )
    assert (
        _current_user_state_root(platform="darwin", environ={}, home=home)
        == home / "Library" / "Application Support" / "nlp-trader" / "kabus"
    )
    assert (
        _current_user_state_root(
            platform="linux",
            environ={"XDG_STATE_HOME": str(xdg_state)},
            home=home,
        )
        == xdg_state / "nlp-trader" / "kabus"
    )
    assert (
        _current_user_state_root(platform="freebsd14", environ={}, home=home)
        == home / ".local" / "state" / "nlp-trader" / "kabus"
    )


@pytest.mark.parametrize(
    ("platform", "environ", "message"),
    [
        ("win32", {}, "LOCALAPPDATA is required"),
        ("win32", {"LOCALAPPDATA": "relative"}, "LOCALAPPDATA must be an absolute"),
        ("linux", {"XDG_STATE_HOME": "relative"}, "XDG_STATE_HOME must be an absolute"),
    ],
)
def test_current_user_state_root_rejects_ambiguous_environment_paths(
    platform: str,
    environ: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(KabuSStatePathError, match=message):
        _current_user_state_root(platform=platform, environ=environ, home=Path("/home/alice"))


def test_lock_contention_is_nonblocking_and_error_is_sanitized(tmp_path: Path) -> None:
    lock_path = (tmp_path / "state" / "operation.lock").resolve()

    with (
        advisory_file_lock(lock_path),
        pytest.raises(KabuSStateLockError) as error,
        advisory_file_lock(lock_path),
    ):
        pytest.fail("contended lock context must not run")

    assert str(error.value) == "another broker operation already holds the lock"


def test_lock_releases_after_normal_and_exceptional_context_exit(tmp_path: Path) -> None:
    lock_path = (tmp_path / "operation.lock").resolve()

    with advisory_file_lock(lock_path):
        pass
    with (
        pytest.raises(LookupError, match="caller failure"),
        advisory_file_lock(lock_path),
    ):
        raise LookupError("caller failure")
    with advisory_file_lock(lock_path):
        pass

    assert lock_path.is_file()


def test_lock_file_identity_is_stable_across_release_and_reacquisition(tmp_path: Path) -> None:
    lock_path = (tmp_path / "operation.lock").resolve()

    with advisory_file_lock(lock_path):
        first_identity = (lock_path.stat().st_dev, lock_path.stat().st_ino)
    with advisory_file_lock(lock_path):
        second_identity = (lock_path.stat().st_dev, lock_path.stat().st_ino)

    assert lock_path.is_file()
    assert second_identity == first_identity


def test_process_exit_releases_lock_without_deleting_stable_file(tmp_path: Path) -> None:
    lock_path = (tmp_path / "operation.lock").resolve()
    script = """
import os
import sys
from nlp_trader.broker.state import advisory_file_lock

with advisory_file_lock(sys.argv[1]):
    os._exit(23)
"""

    child = subprocess.run(
        [sys.executable, "-c", script, str(lock_path)],
        check=False,
        timeout=10,
    )

    assert child.returncode == 23
    assert lock_path.is_file()
    with advisory_file_lock(lock_path):
        pass


def test_lock_metadata_is_bounded_and_replaced_after_acquisition(tmp_path: Path) -> None:
    lock_path = (tmp_path / "operation.lock").resolve()
    lock_path.write_text("stale metadata must be replaced", encoding="utf-8")

    with advisory_file_lock(lock_path):
        encoded = lock_path.read_bytes()
        owner = json.loads(encoded)

    assert len(encoded) <= MAX_LOCK_OWNER_BYTES
    assert owner.keys() == {"acquired_at", "pid"}
    assert owner["pid"] == os.getpid()
    assert owner["acquired_at"].endswith("Z")
    assert b"stale" not in encoded


@pytest.mark.skipif(sys.platform == "win32", reason="Windows chmod has different semantics")
def test_state_directory_and_lock_file_permissions_are_private(tmp_path: Path) -> None:
    root = tmp_path / "state"
    root.mkdir(mode=0o777)
    root.chmod(0o777)
    paths = KabuSStatePaths(root)

    paths.ensure_directory()
    with advisory_file_lock(paths.operation_lock_path):
        assert stat.S_IMODE(paths.root.stat().st_mode) == 0o700
        assert stat.S_IMODE(paths.operation_lock_path.stat().st_mode) == 0o600
