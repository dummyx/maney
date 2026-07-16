"""Neutral durable local-file primitives shared by independent domains."""

from nlp_trader.immutable.append import (
    SafeFileError,
    append_bytes_durable,
    read_bytes_no_follow,
    write_bytes_exclusive_durable,
)
from nlp_trader.immutable.locking import (
    MAX_LOCK_OWNER_BYTES,
    AdvisoryFileLockError,
    AdvisoryFileLockUnavailable,
    advisory_file_lock,
)

__all__ = [
    "MAX_LOCK_OWNER_BYTES",
    "AdvisoryFileLockError",
    "AdvisoryFileLockUnavailable",
    "SafeFileError",
    "advisory_file_lock",
    "append_bytes_durable",
    "read_bytes_no_follow",
    "write_bytes_exclusive_durable",
]
