from __future__ import annotations

import os
import stat
from pathlib import Path


class SafeFileError(RuntimeError):
    """Raised when a durable regular-file operation cannot be completed safely."""

    def __init__(self, message: str, *, operation: str) -> None:
        self.operation = operation
        super().__init__(message)


def append_bytes_durable(path: str | Path, encoded: bytes) -> None:
    """Append all bytes to a private regular file and fsync before returning."""

    if not isinstance(encoded, bytes) or not encoded:
        raise ValueError("durable append requires non-empty bytes")
    destination = _normalized_path(path)
    _ensure_private_directory(destination.parent)
    descriptor, created = _open_append(destination)
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    except OSError as exc:
        raise SafeFileError("durable append failed", operation="append") from exc
    finally:
        os.close(descriptor)
    if created:
        _fsync_directory(destination.parent)


def write_bytes_exclusive_durable(path: str | Path, encoded: bytes) -> None:
    """Create one private regular file exclusively and durably write its exact bytes."""

    if not isinstance(encoded, bytes) or not encoded:
        raise ValueError("durable exclusive write requires non-empty bytes")
    destination = _normalized_path(path)
    _ensure_private_directory(destination.parent)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags = _secure_flags(flags)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError:
        raise
    except OSError as exc:
        raise SafeFileError("exclusive file cannot be opened safely", operation="open") from exc
    try:
        _validate_regular_descriptor(descriptor)
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    except OSError as exc:
        raise SafeFileError("exclusive durable write failed", operation="write") from exc
    finally:
        os.close(descriptor)
    _fsync_directory(destination.parent)


def read_bytes_no_follow(path: str | Path, *, missing_ok: bool = False) -> bytes | None:
    """Read exact bytes from a regular file without following its final symlink."""

    source = _normalized_path(path)
    flags = _secure_flags(os.O_RDONLY)
    try:
        descriptor = os.open(source, flags)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    except OSError as exc:
        raise SafeFileError("regular file cannot be opened safely", operation="open") from exc
    try:
        _validate_regular_descriptor(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise SafeFileError("regular file read failed", operation="read") from exc
    finally:
        os.close(descriptor)


def _normalized_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        raise ValueError("durable file path must be absolute")
    return value.parent.resolve(strict=False) / value.name


def _ensure_private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise SafeFileError("private file directory is unavailable", operation="directory") from exc


def _secure_flags(flags: int) -> int:
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    return flags


def _open_append(path: Path) -> tuple[int, bool]:
    create_flags = _secure_flags(os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        descriptor = os.open(path, create_flags, 0o600)
        created = True
    except FileExistsError:
        append_flags = _secure_flags(os.O_APPEND | os.O_WRONLY)
        try:
            descriptor = os.open(path, append_flags)
        except OSError as exc:
            raise SafeFileError("append file cannot be opened safely", operation="open") from exc
        created = False
    except OSError as exc:
        raise SafeFileError("append file cannot be opened safely", operation="open") from exc
    try:
        _validate_regular_descriptor(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, created


def _validate_regular_descriptor(descriptor: int) -> None:
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise SafeFileError("durable file must be a regular file", operation="regular_file")
    if hasattr(os, "fchmod"):
        os.fchmod(descriptor, 0o600)


def _write_all(descriptor: int, encoded: bytes) -> None:
    view = memoryview(encoded)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if os.name == "nt":
            return
        raise SafeFileError(
            "parent directory cannot be opened for fsync", operation="dir_fsync"
        ) from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if os.name != "nt":
            raise SafeFileError("parent directory fsync failed", operation="dir_fsync") from exc
    finally:
        os.close(descriptor)
