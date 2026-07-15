from __future__ import annotations

import errno
import json
import os
import stat
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

_APPLICATION_DIRECTORY = "nlp-trader"
_BROKER_DIRECTORY = "kabus"
_AUDIT_FILENAME = "audit.jsonl"
_KILL_SWITCH_FILENAME = "KILL_SWITCH"
_OPERATION_LOCK_FILENAME = "operation.lock"
MAX_LOCK_OWNER_BYTES = 256


class KabuSStatePathError(RuntimeError):
    """Raised when a secure current-user broker state path cannot be selected."""


class KabuSStateLockError(RuntimeError):
    """Raised when exclusive ownership of the broker operation lock is unavailable."""


class _LockContentionError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class KabuSStatePaths:
    """Fixed paths for state shared by every kabuS environment and configuration."""

    root: Path

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser()
        if not root.is_absolute():
            raise ValueError("kabuS state root must be an absolute path")
        object.__setattr__(self, "root", root.resolve(strict=False))

    @classmethod
    def for_current_user(cls) -> KabuSStatePaths:
        """Select the one platform-standard state root for the current OS user."""

        return cls(
            _current_user_state_root(
                platform=sys.platform,
                environ=os.environ,
                home=Path.home(),
            )
        )

    @property
    def audit_ledger_path(self) -> Path:
        return self.root / _AUDIT_FILENAME

    @property
    def kill_switch_path(self) -> Path:
        return self.root / _KILL_SWITCH_FILENAME

    @property
    def operation_lock_path(self) -> Path:
        return self.root / _OPERATION_LOCK_FILENAME

    def ensure_directory(self) -> Path:
        """Create the private state directory and constrain its permissions."""

        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.root, 0o700)
        except OSError as exc:
            raise KabuSStatePathError(
                "cannot create the private current-user kabuS state directory"
            ) from exc
        return self.root


def _current_user_state_root(
    *,
    platform: str,
    environ: Mapping[str, str],
    home: Path,
) -> Path:
    if platform == "win32":
        local_app_data = environ.get("LOCALAPPDATA")
        if not local_app_data:
            raise KabuSStatePathError(
                "LOCALAPPDATA is required to locate current-user kabuS state on Windows"
            )
        base = Path(local_app_data).expanduser()
        if not base.is_absolute():
            raise KabuSStatePathError("LOCALAPPDATA must be an absolute path")
    elif platform == "darwin":
        base = home.expanduser() / "Library" / "Application Support"
    else:
        configured = environ.get("XDG_STATE_HOME")
        if configured:
            base = Path(configured).expanduser()
            if not base.is_absolute():
                raise KabuSStatePathError("XDG_STATE_HOME must be an absolute path")
        else:
            base = home.expanduser() / ".local" / "state"
    return base / _APPLICATION_DIRECTORY / _BROKER_DIRECTORY


@contextmanager
def advisory_file_lock(path: str | Path) -> Iterator[None]:
    """Hold a nonblocking OS advisory lock until this context or process exits."""

    lock_path = Path(path).expanduser()
    if not lock_path.is_absolute():
        raise ValueError("broker operation lock path must be absolute")
    lock_path = lock_path.parent.resolve(strict=False) / lock_path.name
    _ensure_private_directory(lock_path.parent)

    descriptor = _open_lock_file(lock_path)
    try:
        try:
            _acquire_nonblocking(descriptor)
        except _LockContentionError:
            raise KabuSStateLockError("another broker operation already holds the lock") from None
        except OSError:
            raise KabuSStateLockError("broker operation lock could not be acquired") from None
        _write_owner_metadata(descriptor)
        yield
    finally:
        # Closing a locked descriptor releases flock/locking even when unwinding an exception.
        # The OS performs the same release if the process exits without running this block.
        try:
            os.close(descriptor)
        except OSError:
            raise KabuSStateLockError("broker operation lock could not be released") from None


def _ensure_private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise KabuSStateLockError("broker operation lock directory is unavailable") from exc


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
        raise KabuSStateLockError("broker operation lock file is unavailable") from None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise KabuSStateLockError("broker operation lock must be a regular file")
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
        raise KabuSStateLockError("broker operation lock owner metadata exceeds its bound")
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
        raise KabuSStateLockError(
            "broker operation lock owner metadata could not be written"
        ) from None
