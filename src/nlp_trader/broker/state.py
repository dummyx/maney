from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from nlp_trader.immutable.locking import (
    MAX_LOCK_OWNER_BYTES as _MAX_LOCK_OWNER_BYTES,
)
from nlp_trader.immutable.locking import (
    AdvisoryFileLockError,
    AdvisoryFileLockUnavailable,
)
from nlp_trader.immutable.locking import (
    advisory_file_lock as _neutral_advisory_file_lock,
)

MAX_LOCK_OWNER_BYTES = _MAX_LOCK_OWNER_BYTES

_APPLICATION_DIRECTORY = "nlp-trader"
_BROKER_DIRECTORY = "kabus"
_AUDIT_FILENAME = "audit.jsonl"
_KILL_SWITCH_FILENAME = "KILL_SWITCH"
_OPERATION_LOCK_FILENAME = "operation.lock"


class KabuSStatePathError(RuntimeError):
    """Raised when a secure current-user broker state path cannot be selected."""


class KabuSStateLockError(RuntimeError):
    """Raised when exclusive ownership of the broker operation lock is unavailable."""


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

    context = _neutral_advisory_file_lock(path)
    try:
        context.__enter__()
    except ValueError as exc:
        raise ValueError("broker operation lock path must be absolute") from exc
    except AdvisoryFileLockUnavailable:
        raise KabuSStateLockError("another broker operation already holds the lock") from None
    except AdvisoryFileLockError as exc:
        _raise_broker_lock_error(exc)
    try:
        yield
    except BaseException:
        exception_type, exception, traceback = sys.exc_info()
        try:
            suppress = context.__exit__(exception_type, exception, traceback)
        except AdvisoryFileLockError as lock_error:
            _raise_broker_lock_error(lock_error)
        if not suppress:
            raise
    else:
        try:
            context.__exit__(None, None, None)
        except AdvisoryFileLockError as exc:
            _raise_broker_lock_error(exc)


def _raise_broker_lock_error(error: AdvisoryFileLockError) -> None:
    messages = {
        "directory": "broker operation lock directory is unavailable",
        "open": "broker operation lock file is unavailable",
        "regular_file": "broker operation lock must be a regular file",
        "owner_size": "broker operation lock owner metadata exceeds its bound",
        "owner_write": "broker operation lock owner metadata could not be written",
        "release": "broker operation lock could not be released",
    }
    raise KabuSStateLockError(
        messages.get(error.operation, "broker operation lock could not be acquired")
    ) from error
