from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType

from pydantic import ValidationError

from nlp_trader.immutable.append import SafeFileError, append_bytes_durable, read_bytes_no_follow
from nlp_trader.immutable.locking import (
    AdvisoryFileLockError,
    AdvisoryFileLockUnavailable,
    advisory_file_lock,
)
from nlp_trader.research_agents.contracts import (
    GENESIS_HASH,
    ResearchAgentRound,
    canonical_json,
)


class ResearchAgentRoundLedgerError(ValueError):
    """Raised when a research-agent round ledger cannot be verified or extended."""


class ResearchAgentRoundLedgerLockError(RuntimeError):
    """Raised when another process owns one research-agent run ledger."""


class _DuplicateJsonKeyError(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(key)


class ResearchAgentRoundLedger:
    """Locked append-only hash chain for one model/tool execution trace."""

    def __init__(self, path: str | Path) -> None:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            raise ValueError("research-agent round-ledger path must be absolute")
        self.path = candidate.parent.resolve(strict=False) / candidate.name
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def append(self, round_record: ResearchAgentRound) -> ResearchAgentRound:
        if not isinstance(round_record, ResearchAgentRound):
            raise TypeError("round_record must be a ResearchAgentRound")
        with self._exclusive_lock():
            rounds = self._replay_unlocked()
            expected_step = len(rounds) + 1
            expected_previous = rounds[-1].round_id if rounds else GENESIS_HASH
            if round_record.step != expected_step:
                raise ResearchAgentRoundLedgerError(
                    f"research-agent round step must be {expected_step}"
                )
            if round_record.previous_round_hash != expected_previous:
                raise ResearchAgentRoundLedgerError(
                    "research-agent round breaks the previous hash link"
                )
            if rounds:
                first = rounds[0]
                if (
                    round_record.agent_run_id,
                    round_record.study_id,
                    round_record.attempt_id,
                ) != (first.agent_run_id, first.study_id, first.attempt_id):
                    raise ResearchAgentRoundLedgerError(
                        "all rounds must share one run, study, and attempt identity"
                    )
            try:
                append_bytes_durable(
                    self.path,
                    (round_record.canonical_json() + "\n").encode("utf-8"),
                )
            except (SafeFileError, OSError, ValueError) as exc:
                raise ResearchAgentRoundLedgerError(
                    "research-agent round cannot be appended safely"
                ) from exc
            return round_record

    def replay(self) -> tuple[ResearchAgentRound, ...]:
        with self._exclusive_lock():
            return self._replay_unlocked()

    def _replay_unlocked(self) -> tuple[ResearchAgentRound, ...]:
        try:
            encoded = read_bytes_no_follow(self.path, missing_ok=True)
        except (SafeFileError, OSError, ValueError) as exc:
            raise ResearchAgentRoundLedgerError(
                "research-agent round ledger cannot be opened safely"
            ) from exc
        if encoded is None:
            return ()
        try:
            text = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ResearchAgentRoundLedgerError(
                "research-agent round ledger is not valid UTF-8"
            ) from exc
        rounds: list[ResearchAgentRound] = []
        expected_previous = GENESIS_HASH
        seen_ids: set[str] = set()
        for line_number, raw_line in enumerate(text.splitlines(keepends=True), start=1):
            context = f"research-agent round ledger line {line_number}"
            if not raw_line.strip():
                raise ResearchAgentRoundLedgerError(f"{context} is blank")
            if not raw_line.endswith("\n"):
                raise ResearchAgentRoundLedgerError(f"{context} is incomplete")
            parsed = _parse_json_line(raw_line, context=context)
            if raw_line != canonical_json(parsed) + "\n":
                raise ResearchAgentRoundLedgerError(f"{context} is not canonical JSON")
            try:
                round_record = ResearchAgentRound.model_validate_json(raw_line)
            except ValidationError as exc:
                raise ResearchAgentRoundLedgerError(
                    f"{context} violates the round contract"
                ) from exc
            if raw_line != round_record.canonical_json() + "\n":
                raise ResearchAgentRoundLedgerError(f"{context} is not canonical typed round JSON")
            if round_record.step != line_number:
                raise ResearchAgentRoundLedgerError(
                    f"{context} has step {round_record.step}; expected {line_number}"
                )
            if round_record.previous_round_hash != expected_previous:
                raise ResearchAgentRoundLedgerError(f"{context} breaks the previous hash link")
            if round_record.round_id in seen_ids:
                raise ResearchAgentRoundLedgerError(f"{context} repeats round_id")
            if rounds:
                first = rounds[0]
                if (
                    round_record.agent_run_id,
                    round_record.study_id,
                    round_record.attempt_id,
                ) != (first.agent_run_id, first.study_id, first.attempt_id):
                    raise ResearchAgentRoundLedgerError(
                        f"{context} changes run, study, or attempt identity"
                    )
            seen_ids.add(round_record.round_id)
            expected_previous = round_record.round_id
            rounds.append(round_record)
        if text and not text.endswith("\n"):
            raise ResearchAgentRoundLedgerError(
                "research-agent round ledger has an incomplete trailing line"
            )
        return tuple(rounds)

    def _exclusive_lock(self) -> _RoundLockContext:
        return _RoundLockContext(self.lock_path)


class _RoundLockContext:
    def __init__(self, path: Path) -> None:
        self._context = advisory_file_lock(path)

    def __enter__(self) -> None:
        try:
            self._context.__enter__()
        except (AdvisoryFileLockError, AdvisoryFileLockUnavailable) as exc:
            raise ResearchAgentRoundLedgerLockError(
                "research-agent round ledger lock is unavailable"
            ) from exc

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        try:
            return self._context.__exit__(exc_type, exc, traceback)
        except (AdvisoryFileLockError, AdvisoryFileLockUnavailable) as error:
            raise ResearchAgentRoundLedgerLockError(
                "research-agent round ledger lock is unavailable"
            ) from error


def _parse_json_line(raw_line: str, *, context: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw_line,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except _DuplicateJsonKeyError as exc:
        raise ResearchAgentRoundLedgerError(f"{context} repeats JSON key {exc.key!r}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ResearchAgentRoundLedgerError(f"{context} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ResearchAgentRoundLedgerError(f"{context} must contain an object")
    return value


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")
