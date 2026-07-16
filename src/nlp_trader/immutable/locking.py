from __future__ import annotations

import errno
import json
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

MAX_LOCK_OWNER_BYTES = 256


class AdvisoryFileLockError(RuntimeError):
    """Raised when a local advisory lock cannot be opened, held, or released safely."""

    def __init__(self, message: str, *, operation: str) -> None:
        self.operation = operation
        super().__init__(message)


class AdvisoryFileLockUnavailable(AdvisoryFileLockError):
    """Raised immediately when another process already owns the advisory lock."""


class _LockContentionError(Exception):
    pass


@contextmanager
def advisory_file_lock(path: str | Path) -> Iterator[None]:
    """Hold one nonblocking advisory lock on a stable private regular file."""

    lock_path = Path(path).expanduser()
    if not lock_path.is_absolute():
        raise ValueError("advisory lock path must be absolute")
    lock_path = lock_path.parent.resolve(strict=False) / lock_path.name
    _ensure_private_directory(lock_path.parent)

    descriptor = _open_lock_file(lock_path)
    try:
        try:
            _acquire_nonblocking(descriptor)
        except _LockContentionError:
            raise AdvisoryFileLockUnavailable(
                "another process already holds the advisory lock",
                operation="contention",
            ) from None
        except OSError:
            raise AdvisoryFileLockError(
                "advisory lock could not be acquired",
                operation="acquire",
            ) from None
        _write_owner_metadata(descriptor)
        yield
    finally:
        try:
            os.close(descriptor)
        except OSError:
            raise AdvisoryFileLockError(
                "advisory lock could not be released",
                operation="release",
            ) from None


def _ensure_private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise AdvisoryFileLockError(
            "advisory lock directory is unavailable",
            operation="directory",
        ) from exc


def _open_lock_file(path: Path) -> int:
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if sys.platform == "win32" and hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError:
        raise AdvisoryFileLockError(
            "advisory lock file is unavailable",
            operation="open",
        ) from None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise AdvisoryFileLockError(
                "advisory lock must be a regular file",
                operation="regular_file",
            )
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _acquire_nonblocking(descriptor: int) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if sys.platform == "win32":
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK} or getattr(
                exc, "winerror", None
            ) in {33, 36, 158}:
                raise _LockContentionError from exc
            raise
    else:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise _LockContentionError from exc
            raise


def _write_owner_metadata(descriptor: int) -> None:
    owner = {
        "acquired_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "pid": os.getpid(),
    }
    encoded = (
        json.dumps(owner, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    if len(encoded) > MAX_LOCK_OWNER_BYTES:
        raise AdvisoryFileLockError(
            "advisory lock owner metadata exceeds its bound",
            operation="owner_size",
        )
    try:
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    except OSError:
        raise AdvisoryFileLockError(
            "advisory lock owner metadata could not be written",
            operation="owner_write",
        ) from None
