from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from nlp_trader.immutable.append import (
    SafeFileError,
    append_bytes_durable,
    read_bytes_no_follow,
    write_bytes_exclusive_durable,
)
from nlp_trader.immutable.locking import (
    AdvisoryFileLockUnavailable,
    advisory_file_lock,
)


def test_neutral_lock_is_nonblocking_stable_and_private(tmp_path: Path) -> None:
    path = (tmp_path / "state" / "authority.lock").resolve()

    with advisory_file_lock(path):
        identity = (path.stat().st_dev, path.stat().st_ino)
        with pytest.raises(AdvisoryFileLockUnavailable), advisory_file_lock(path):
            pytest.fail("contended lock body must not run")

    with advisory_file_lock(path):
        assert (path.stat().st_dev, path.stat().st_ino) == identity

    if os.name == "posix":
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_durable_append_and_exclusive_write_preserve_exact_bytes(tmp_path: Path) -> None:
    path = (tmp_path / "registry" / "events.jsonl").resolve()
    append_bytes_durable(path, b"first\n")
    first = path.read_bytes()
    append_bytes_durable(path, b"second\n")

    assert first == b"first\n"
    assert path.read_bytes() == b"first\nsecond\n"
    assert read_bytes_no_follow(path) == b"first\nsecond\n"
    manifest = (tmp_path / "manifest.json").resolve()
    write_bytes_exclusive_durable(manifest, b"{}\n")
    with pytest.raises(FileExistsError):
        write_bytes_exclusive_durable(manifest, b"changed\n")
    assert manifest.read_bytes() == b"{}\n"


def test_durable_append_loops_on_short_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = (tmp_path / "events.jsonl").resolve()
    original_write = os.write
    calls = 0

    def short_write(descriptor: int, encoded: bytes | memoryview) -> int:
        nonlocal calls
        calls += 1
        view = memoryview(encoded)
        size = max(1, len(view) // 2)
        return original_write(descriptor, view[:size])

    monkeypatch.setattr("nlp_trader.immutable.append.os.write", short_write)
    append_bytes_durable(path, b"complete-record\n")

    assert path.read_bytes() == b"complete-record\n"
    assert calls > 1


def test_neutral_file_operations_reject_relative_and_symlink_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        append_bytes_durable(Path("relative.jsonl"), b"row\n")

    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("platform does not expose no-follow file opens")
    target = (tmp_path / "target.jsonl").resolve()
    target.write_bytes(b"untouched\n")
    link = (tmp_path / "link.jsonl").resolve()
    link.symlink_to(target)

    with pytest.raises(SafeFileError):
        append_bytes_durable(link, b"malicious\n")
    with pytest.raises(SafeFileError):
        read_bytes_no_follow(link)
    assert target.read_bytes() == b"untouched\n"
