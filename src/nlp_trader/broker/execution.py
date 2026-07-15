from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from nlp_trader import __version__
from nlp_trader.broker.audit import BrokerAuditLedger
from nlp_trader.broker.config import KabuSConfig
from nlp_trader.broker.contracts import CashOrderIntent
from nlp_trader.broker.kabus import (
    AmbiguousMutationError,
    JsonObject,
    KabuSClientError,
    OrderResult,
    _cash_order_payload,
    _KabuSClient,
)
from nlp_trader.broker.state import KabuSStateLockError, KabuSStatePaths, advisory_file_lock
from nlp_trader.timestamps import format_utc, parse_utc

_JAPAN = ZoneInfo("Asia/Tokyo")
_ACTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SYMBOL = re.compile(r"^[A-Z0-9]{1,16}$")
_OPEN_ORDER_STATES = frozenset({1, 2, 3, 4})
_TERMINAL_ORDER_STATE = 5
_MAX_KILL_SWITCH_MARKER_BYTES = 4096
_CASH_EQUITY_TICK_TABLES: dict[str, tuple[tuple[Decimal | None, Decimal], ...]] = {
    "10000": (
        (Decimal("3000"), Decimal("1")),
        (Decimal("5000"), Decimal("5")),
        (Decimal("30000"), Decimal("10")),
        (Decimal("50000"), Decimal("50")),
        (Decimal("300000"), Decimal("100")),
        (Decimal("500000"), Decimal("500")),
        (Decimal("3000000"), Decimal("1000")),
        (Decimal("5000000"), Decimal("5000")),
        (Decimal("30000000"), Decimal("10000")),
        (Decimal("50000000"), Decimal("50000")),
        (None, Decimal("100000")),
    ),
    "10003": (
        (Decimal("1000"), Decimal("0.1")),
        (Decimal("3000"), Decimal("0.5")),
        (Decimal("10000"), Decimal("1")),
        (Decimal("30000"), Decimal("5")),
        (Decimal("100000"), Decimal("10")),
        (Decimal("300000"), Decimal("50")),
        (Decimal("1000000"), Decimal("100")),
        (Decimal("3000000"), Decimal("500")),
        (Decimal("10000000"), Decimal("1000")),
        (Decimal("30000000"), Decimal("5000")),
        (None, Decimal("10000")),
    ),
    "10004": (
        (Decimal("10000"), Decimal("1")),
        (Decimal("30000"), Decimal("5")),
        (Decimal("100000"), Decimal("10")),
        (Decimal("300000"), Decimal("50")),
        (Decimal("1000000"), Decimal("100")),
        (Decimal("3000000"), Decimal("500")),
        (Decimal("10000000"), Decimal("1000")),
        (Decimal("30000000"), Decimal("5000")),
        (None, Decimal("10000")),
    ),
}


class BrokerSafetyError(RuntimeError):
    """A fail-closed broker safety or state check rejected an operation."""


class _DuplicateJsonKeyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BrokerAccountSnapshot:
    environment: str
    open_order_count: int | None
    cash_position_count: int | None
    account_position_count: int | None
    cash_wallet_jpy: float | None
    stock_soft_limit_jpy: float | None
    total_unrealized_profit_loss_jpy: float | None
    unresolved_client_order_ids: tuple[str, ...]
    kill_switch_engaged: bool
    kill_switch_digest: str | None
    unavailable_fields: tuple[str, ...] = ()

    def public_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "open_order_count": self.open_order_count,
            "cash_position_count": self.cash_position_count,
            "account_position_count": self.account_position_count,
            "cash_wallet_jpy": self.cash_wallet_jpy,
            "stock_soft_limit_jpy": self.stock_soft_limit_jpy,
            "total_unrealized_profit_loss_jpy": self.total_unrealized_profit_loss_jpy,
            "unresolved_client_order_ids": list(self.unresolved_client_order_ids),
            "kill_switch_engaged": self.kill_switch_engaged,
            "kill_switch_digest": self.kill_switch_digest,
            "unavailable_fields": list(self.unavailable_fields),
        }


@dataclass(frozen=True, slots=True)
class BrokerReconciliation:
    account: BrokerAccountSnapshot
    audit_sequence: int
    config_digest: str
    evidence: tuple[dict[str, Any], ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "account": self.account.public_dict(),
            "audit_sequence": self.audit_sequence,
            "config_digest": self.config_digest,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class _BrokerAccountState:
    environment: str
    open_order_count: int
    cash_position_count: int
    account_position_count: int
    cash_wallet_jpy: float
    stock_soft_limit_jpy: float
    total_unrealized_profit_loss_jpy: float
    unresolved_client_order_ids: tuple[str, ...]
    kill_switch_engaged: bool
    kill_switch_digest: str | None
    orders: tuple[JsonObject, ...]
    all_product_orders: tuple[JsonObject, ...]
    cash_positions: tuple[JsonObject, ...]

    def public_snapshot(self) -> BrokerAccountSnapshot:
        return BrokerAccountSnapshot(
            environment=self.environment,
            open_order_count=self.open_order_count,
            cash_position_count=self.cash_position_count,
            account_position_count=self.account_position_count,
            cash_wallet_jpy=self.cash_wallet_jpy,
            stock_soft_limit_jpy=self.stock_soft_limit_jpy,
            total_unrealized_profit_loss_jpy=self.total_unrealized_profit_loss_jpy,
            unresolved_client_order_ids=self.unresolved_client_order_ids,
            kill_switch_engaged=self.kill_switch_engaged,
            kill_switch_digest=self.kill_switch_digest,
        )


@dataclass(frozen=True, slots=True)
class OrderPreflight:
    notional_jpy: float
    reference_price: float
    quote_at: datetime
    trading_unit: int
    current_position_quantity: int
    available_to_sell_quantity: int
    pending_buy_quantity: int
    current_symbol_position_notional_jpy: float
    gross_cash_position_notional_jpy: float
    pending_buy_notional_jpy: float
    cash_buying_power_jpy: float
    broker_symbol_limit_jpy: float | None
    account: _BrokerAccountState

    def audit_dict(self) -> dict[str, Any]:
        """Retain constraint evidence without copying broker account payloads."""

        return {
            "notional_jpy": self.notional_jpy,
            "reference_price": self.reference_price,
            "quote_at": format_utc(self.quote_at),
            "trading_unit": self.trading_unit,
            "current_position_quantity": self.current_position_quantity,
            "available_to_sell_quantity": self.available_to_sell_quantity,
            "pending_buy_quantity": self.pending_buy_quantity,
            "current_symbol_position_notional_jpy": self.current_symbol_position_notional_jpy,
            "gross_cash_position_notional_jpy": self.gross_cash_position_notional_jpy,
            "pending_buy_notional_jpy": self.pending_buy_notional_jpy,
            "cash_buying_power_jpy": self.cash_buying_power_jpy,
            "broker_symbol_limit_jpy": self.broker_symbol_limit_jpy,
            "open_order_count": self.account.open_order_count,
            "cash_check_passed": True,
            "position_check_passed": True,
            "loss_check_passed": True,
            "soft_limit_check_passed": True,
        }


@dataclass(frozen=True, slots=True)
class BrokerSubmission:
    client_order_id: str
    broker_order_id: str
    intent_digest: str
    environment: str
    notional_jpy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "intent_digest": self.intent_digest,
            "environment": self.environment,
            "notional_jpy": self.notional_jpy,
        }


class KabuSExecutor:
    """Single-order, fail-closed coordinator around the low-level kabuS client."""

    def __init__(
        self,
        config: KabuSConfig,
        client: _KabuSClient,
        ledger: BrokerAuditLedger,
        *,
        state_paths: KabuSStatePaths,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        authenticate: Callable[[], None] | None = None,
    ) -> None:
        if client.environment != config.environment:
            raise ValueError("kabuS client environment must match broker config")
        if Path(os.path.abspath(ledger.path)) != Path(
            os.path.abspath(state_paths.audit_ledger_path)
        ):
            raise ValueError("broker audit ledger path must match shared current-user state")
        self.config = config
        self.client = client
        self.ledger = ledger
        self.state_paths = state_paths
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        self._authenticate = authenticate

    def preview_order(self, intent: CashOrderIntent) -> dict[str, Any]:
        """Validate local order fields and return the exact confirmation digest."""

        now = self._now()
        self._validate_static_order(intent, now=now)
        estimated_notional = intent.quantity * intent.limit_price
        request_payload = self._effective_order_payload(intent)
        return {
            "environment": self.config.environment,
            "production_places_real_orders": self.config.environment == "production",
            "confirmation_digest": self._order_approval_digest(intent, request_payload),
            "intent_digest": intent.confirmation_digest(),
            "config_digest": self.config.digest(),
            "intent": json.loads(intent.canonical_json()),
            "effective_request_payload": request_payload,
            "limit_notional_jpy": estimated_notional,
            "network_request_made": False,
        }

    def status(self) -> BrokerAccountSnapshot:
        """Return a sanitized live/validation account summary without changing local state."""

        with self._exclusive_operation():
            self._require_enabled()
            self._authenticate_client()
            snapshot, _, _ = self._incident_account_snapshot()
            return snapshot

    def reconcile(self) -> BrokerReconciliation:
        """Record a bounded account summary; unresolved mutations remain blocked."""

        with self._exclusive_operation():
            return self._reconcile_locked()

    def _reconcile_locked(self) -> BrokerReconciliation:
        self._require_enabled()
        self._authenticate_client()
        snapshot, orders, orders_observed_at = self._incident_account_snapshot()
        reconciliation_completed_at = self._now()
        evidence_observed_at = orders_observed_at or reconciliation_completed_at
        evidence = self._reconciliation_evidence(orders, observed_at=evidence_observed_at)
        event = self._append(
            "broker_reconciliation",
            {
                "observed_at": format_utc(evidence_observed_at),
                "reconciliation_completed_at": format_utc(reconciliation_completed_at),
                "open_order_count": snapshot.open_order_count,
                "cash_position_count": snapshot.cash_position_count,
                "account_position_count": snapshot.account_position_count,
                "loss_limit_breached": (
                    None
                    if snapshot.total_unrealized_profit_loss_jpy is None
                    else snapshot.total_unrealized_profit_loss_jpy
                    <= -self.config.max_total_unrealized_loss_jpy
                ),
                "unresolved_client_order_ids": list(snapshot.unresolved_client_order_ids),
                "kill_switch_engaged": snapshot.kill_switch_engaged,
                "unavailable_fields": list(snapshot.unavailable_fields),
                "config_digest": self.config.digest(),
                "adapter_version": __version__,
                "evidence": evidence,
            },
        )
        return BrokerReconciliation(
            account=snapshot,
            audit_sequence=int(event["sequence"]),
            config_digest=self.config.digest(),
            evidence=tuple(evidence),
        )

    def submit_order(
        self,
        intent: CashOrderIntent,
        *,
        confirmation: str,
        production_confirmation: str | None = None,
    ) -> BrokerSubmission:
        """Preflight and submit exactly one cash order without automatic retry."""

        with self._exclusive_operation():
            return self._submit_order_locked(
                intent,
                confirmation=confirmation,
                production_confirmation=production_confirmation,
            )

    def _submit_order_locked(
        self,
        intent: CashOrderIntent,
        *,
        confirmation: str,
        production_confirmation: str | None,
    ) -> BrokerSubmission:
        started_monotonic = self._monotonic()
        now = self._now()
        self._require_order_submission(production_confirmation)
        request_payload = self._effective_order_payload(intent)
        approval_digest = self._order_approval_digest(intent, request_payload)
        self._require_exact_confirmation(confirmation, approval_digest)
        self._require_kill_switch_clear()
        self._validate_static_order(intent, now=now)

        if self.ledger.seen_client_order_id(intent.client_order_id):
            raise BrokerSafetyError(
                f"client_order_id already exists in the audit ledger: {intent.client_order_id}"
            )
        unresolved = self.ledger.unresolved_order_ids()
        if unresolved:
            raise BrokerSafetyError(
                "new orders are blocked by unresolved broker mutations: " + ", ".join(unresolved)
            )

        self._authenticate_client()
        preflight = self._preflight(intent, now=now)
        ready_now = self._now()
        self._require_preflight_duration(started_monotonic)
        self._validate_static_order(intent, now=ready_now)
        self._validate_expiry(intent, now=ready_now)
        order_date = ready_now.astimezone(_JAPAN).date()
        accepted_today = self.ledger.accepted_notional_for_date(order_date)
        if accepted_today + preflight.notional_jpy > self.config.max_daily_order_notional_jpy:
            raise BrokerSafetyError("adapter daily accepted-order notional limit would be exceeded")

        intent_payload = json.loads(intent.canonical_json())
        common = {
            "client_order_id": intent.client_order_id,
            "intent_digest": intent.confirmation_digest(),
            "approval_digest": approval_digest,
            "config_digest": self.config.digest(),
            "adapter_version": __version__,
            "effective_config": json.loads(self.config.canonical_json()),
            "effective_request_payload": request_payload,
            "order": intent_payload,
            "notional_jpy": preflight.notional_jpy,
            "order_date": order_date.isoformat(),
        }
        self._append(
            "broker_order_preflight",
            {**common, "checks": preflight.audit_dict()},
            event_ts=ready_now,
        )
        self._append("broker_order_attempt", common, event_ts=ready_now)

        try:
            final_now = self._now()
            self._require_preflight_duration(started_monotonic)
            self._validate_static_order(intent, now=final_now)
            self._validate_expiry(intent, now=final_now)
            self._validate_quote_freshness(preflight.quote_at, now=final_now)
            if final_now.astimezone(_JAPAN).date() != order_date:
                raise BrokerSafetyError("Japan order date changed during broker preflight")
            if (
                self.ledger.accepted_notional_for_date(order_date) + preflight.notional_jpy
                > self.config.max_daily_order_notional_jpy
            ):
                raise BrokerSafetyError(
                    "adapter daily accepted-order notional limit would be exceeded"
                )
            transport_now = self._now()
            self._require_preflight_duration(started_monotonic)
            self._validate_static_order(intent, now=transport_now)
            self._validate_expiry(intent, now=transport_now)
            self._validate_quote_freshness(preflight.quote_at, now=transport_now)
            if transport_now.astimezone(_JAPAN).date() != order_date:
                raise BrokerSafetyError("Japan order date changed during broker preflight")
            self._require_kill_switch_clear()
        except BrokerSafetyError:
            self._append(
                "broker_order_rejected",
                {**common, "reason": "final_safety_recheck_failed_before_transport"},
            )
            raise
        try:
            result = self.client.send_cash_order(
                intent,
                account_type=self.config.account_type,
                cash_buy_deliv_type=self.config.cash_buy_deliv_type,
                cash_buy_fund_type=self.config.cash_buy_fund_type,
            )
        except AmbiguousMutationError:
            self._append("broker_order_unknown", common)
            raise
        except KabuSClientError:
            self._append("broker_order_unknown", common)
            raise AmbiguousMutationError("send order") from None

        self._append(
            "broker_order_accepted",
            {**common, "broker_order_id": result.order_id},
        )
        return BrokerSubmission(
            client_order_id=intent.client_order_id,
            broker_order_id=result.order_id,
            intent_digest=intent.confirmation_digest(),
            environment=self.config.environment,
            notional_jpy=preflight.notional_jpy,
        )

    def cancel_order(
        self,
        order_id: str,
        *,
        client_action_id: str,
        confirmation: str,
        production_confirmation: str | None = None,
    ) -> OrderResult:
        """Cancel one confirmed open order; the kill switch does not block cancellations."""

        with self._exclusive_operation():
            return self._cancel_order_locked(
                order_id,
                client_action_id=client_action_id,
                confirmation=confirmation,
                production_confirmation=production_confirmation,
            )

    def preview_cancel(self, order_id: str, *, client_action_id: str) -> dict[str, Any]:
        """Read and bind one current cash-order state to a cancellation approval."""

        with self._exclusive_operation():
            self._require_enabled()
            _validate_action_id(client_action_id)
            self._authenticate_client()
            order = self._cancel_candidate(order_id)
            context = _cancel_order_context(order)
            return {
                "environment": self.config.environment,
                "production_cancellation_is_real": self.config.environment == "production",
                "client_action_id": client_action_id,
                "broker_order": context,
                "config_digest": self.config.digest(),
                "confirmation_digest": self._cancel_approval_digest(
                    client_action_id,
                    context,
                ),
                "network_mutation_made": False,
            }

    def _cancel_order_locked(
        self,
        order_id: str,
        *,
        client_action_id: str,
        confirmation: str,
        production_confirmation: str | None,
    ) -> OrderResult:
        self._require_recovery_operation(production_confirmation)
        _validate_action_id(client_action_id)
        if self.ledger.seen_client_order_id(client_action_id):
            raise BrokerSafetyError(
                f"client_action_id already exists in the audit ledger: {client_action_id}"
            )
        unresolved = set(self.ledger.unresolved_order_ids())
        for event in self.ledger.replay():
            if (
                event.get("client_order_id") in unresolved
                and event.get("event_type") == "broker_cancel_attempt"
                and event.get("broker_order_id") == order_id
            ):
                raise BrokerSafetyError(
                    "the broker order already has an unresolved cancellation attempt"
                )

        self._authenticate_client()
        order = self._cancel_candidate(order_id)
        order_context = _cancel_order_context(order)
        approval_digest = self._cancel_approval_digest(client_action_id, order_context)
        self._require_exact_confirmation(confirmation, approval_digest)

        common = {
            "client_order_id": client_action_id,
            "broker_order_id": order_id,
            "approval_digest": approval_digest,
            "config_digest": self.config.digest(),
            "adapter_version": __version__,
            "effective_config": json.loads(self.config.canonical_json()),
            "broker_order": order_context,
        }
        self._append("broker_cancel_attempt", common)
        try:
            result = self.client.cancel_order(order_id)
        except AmbiguousMutationError:
            self._append("broker_cancel_unknown", common)
            raise
        except KabuSClientError:
            self._append("broker_cancel_unknown", common)
            raise AmbiguousMutationError("cancel order") from None
        if result.order_id != order_id:
            self._append(
                "broker_cancel_unknown",
                {**common, "returned_broker_order_id": result.order_id},
            )
            raise AmbiguousMutationError("cancel order")
        self._append(
            "broker_cancel_accepted",
            {**common, "accepted_broker_order_id": result.order_id},
        )
        return result

    def _cancel_candidate(self, order_id: str) -> JsonObject:
        matching = self.client.get_orders(order_id)
        if len(matching) != 1:
            raise BrokerSafetyError("cancel preflight requires exactly one matching broker order")
        order = matching[0]
        if order.get("ID") != order_id:
            raise BrokerSafetyError("cancel preflight broker order ID does not match the request")
        state = _order_state(order)
        if state == _TERMINAL_ORDER_STATE:
            raise BrokerSafetyError("broker order is already terminal and cannot be cancelled")
        if state == 4:
            raise BrokerSafetyError("broker order already has a cancellation in flight")
        if state != 3:
            raise BrokerSafetyError(
                "broker order is not in the processed state required for cancellation"
            )
        _cancel_order_context(order)
        return order

    def _cancel_approval_digest(
        self,
        client_action_id: str,
        order_context: Mapping[str, Any],
    ) -> str:
        return _canonical_digest(
            {
                "schema_version": "kabus-cancel-approval-v1",
                "adapter_version": __version__,
                "client_action_id": client_action_id,
                "effective_config": json.loads(self.config.canonical_json()),
                "broker_order": dict(order_context),
            }
        )

    def resolve_unknown(
        self,
        client_order_id: str,
        *,
        resolution: Literal["accepted", "not-accepted"],
        confirmation: str,
        broker_order_id: str | None = None,
        production_confirmation: str | None = None,
    ) -> dict[str, Any]:
        """Resolve an ambiguous local mutation only after a recorded reconciliation."""

        with self._exclusive_operation():
            return self._resolve_unknown_locked(
                client_order_id,
                resolution=resolution,
                confirmation=confirmation,
                broker_order_id=broker_order_id,
                production_confirmation=production_confirmation,
            )

    def _resolve_unknown_locked(
        self,
        client_order_id: str,
        *,
        resolution: Literal["accepted", "not-accepted"],
        confirmation: str,
        broker_order_id: str | None,
        production_confirmation: str | None,
    ) -> dict[str, Any]:
        _validate_action_id(client_order_id)
        if resolution not in ("accepted", "not-accepted"):
            raise BrokerSafetyError("unknown resolution must be accepted or not-accepted")
        events = self.ledger.replay()
        related = [event for event in events if event.get("client_order_id") == client_order_id]
        unresolved = set(self.ledger.unresolved_order_ids())
        if client_order_id not in unresolved or not related:
            raise BrokerSafetyError("client order/action ID is not unresolved")
        latest_mutation_sequence = max(
            int(event["sequence"])
            for event in related
            if str(event["event_type"]).endswith(("_attempt", "_unknown"))
        )
        attempt = next(
            event
            for event in related
            if event["event_type"] in {"broker_order_attempt", "broker_cancel_attempt"}
        )
        if attempt.get("environment") != self.config.environment:
            raise BrokerSafetyError("resolve unknown with the attempt's original environment")
        if attempt.get("config_digest") != self.config.digest():
            raise BrokerSafetyError("resolve unknown with the attempt's original broker config")
        self._require_recovery_operation(production_confirmation)
        reconciliation_entry: Mapping[str, Any] | None = None
        reconciliation_sequence: int | None = None
        for event in reversed(events):
            if (
                event.get("event_type") != "broker_reconciliation"
                or int(event["sequence"]) <= latest_mutation_sequence
                or event.get("environment") != attempt.get("environment")
                or event.get("config_digest") != attempt.get("config_digest")
            ):
                continue
            evidence = event.get("evidence")
            if not isinstance(evidence, list):
                continue
            matching_entries = [
                item
                for item in evidence
                if isinstance(item, Mapping)
                and item.get("client_order_id") == client_order_id
                and item.get("attempt_sequence") == int(attempt["sequence"])
                and item.get("attempt_config_digest") == attempt.get("config_digest")
            ]
            if len(matching_entries) > 1:
                raise BrokerSafetyError("broker reconciliation repeats mutation evidence")
            if len(matching_entries) == 1:
                reconciliation_entry = matching_entries[0]
                reconciliation_sequence = int(event["sequence"])
                break
        if (
            reconciliation_entry is None
            or reconciliation_sequence is None
            or reconciliation_entry.get("evidence_valid") is not True
        ):
            raise BrokerSafetyError(
                "run a scoped broker reconciliation with valid order evidence first"
            )

        is_cancel = attempt["event_type"] == "broker_cancel_attempt"
        prefix = "broker_cancel" if is_cancel else "broker_order"

        if resolution == "accepted":
            if not broker_order_id:
                raise BrokerSafetyError("accepted resolution requires broker_order_id")
            expected = (
                f"ACCEPTED:{client_order_id}:{broker_order_id}:"
                f"{int(attempt['sequence'])}:{reconciliation_sequence}:{self.config.digest()}"
            )
            self._require_exact_confirmation(confirmation, expected)
            exact_matches = reconciliation_entry.get("exact_match_order_ids")
            candidate_ids = reconciliation_entry.get("candidate_order_ids")
            if exact_matches != [broker_order_id] or candidate_ids != [broker_order_id]:
                raise BrokerSafetyError(
                    "accepted resolution requires one unique exact reconciliation match"
                )
            self._require_fresh_reconciliation_evidence(reconciliation_entry)
            fresh_verification = self._fresh_resolution_evidence(
                client_order_id,
                attempt,
            )
            if fresh_verification.get("exact_match_order_ids") != [
                broker_order_id
            ] or fresh_verification.get("candidate_order_ids") != [broker_order_id]:
                raise BrokerSafetyError(
                    "accepted resolution requires one unique exact match in a fresh all-order read"
                )
            event = self._append(
                f"{prefix}_accepted",
                {
                    **_resolution_common(attempt),
                    "broker_order_id": broker_order_id,
                    "resolution_method": "broker_reconciliation",
                    "reconciliation_sequence": reconciliation_sequence,
                    "fresh_verification": fresh_verification,
                },
            )
        else:
            if broker_order_id is not None:
                raise BrokerSafetyError("not-accepted resolution forbids broker_order_id")
            expected = (
                f"NOT_ACCEPTED:{client_order_id}:{int(attempt['sequence'])}:"
                f"{reconciliation_sequence}:{self.config.digest()}"
            )
            self._require_exact_confirmation(confirmation, expected)
            exact_matches = reconciliation_entry.get("exact_match_order_ids")
            candidate_ids = reconciliation_entry.get("candidate_order_ids")
            if exact_matches != [] or candidate_ids != []:
                raise BrokerSafetyError(
                    "not-accepted resolution requires no reconciled candidate broker orders"
                )
            if reconciliation_entry.get("negative_observation_complete") is not True:
                raise BrokerSafetyError(
                    "not-accepted resolution requires the complete reconciliation window"
                )
            self._require_fresh_reconciliation_evidence(reconciliation_entry)
            kill_switch_engaged, _ = self._kill_switch_status()
            if not kill_switch_engaged:
                raise BrokerSafetyError(
                    "not-accepted manual resolution requires the kill switch to be engaged"
                )
            fresh_verification = self._fresh_resolution_evidence(
                client_order_id,
                attempt,
            )
            if (
                fresh_verification.get("exact_match_order_ids") != []
                or fresh_verification.get("candidate_order_ids") != []
                or fresh_verification.get("negative_observation_complete") is not True
            ):
                raise BrokerSafetyError(
                    "not-accepted resolution requires a fresh complete read with no candidates"
                )
            event = self._append(
                f"{prefix}_rejected",
                {
                    **_resolution_common(attempt),
                    "resolution_method": "operator_asserted_not_accepted",
                    "reconciliation_sequence": reconciliation_sequence,
                    "fresh_verification": fresh_verification,
                },
            )
        return event

    def engage_kill_switch(self, reason: str) -> dict[str, Any]:
        """Create the risk-off marker while excluding every broker operation."""

        if not isinstance(reason, str) or not reason.strip():
            raise BrokerSafetyError("kill-switch reason must not be empty")
        if len(reason.strip().encode("utf-8")) > 512:
            raise BrokerSafetyError("kill-switch reason exceeds the byte limit")
        with self._exclusive_operation():
            return self._engage_kill_switch_locked(reason.strip())

    def _engage_kill_switch_locked(self, reason: str) -> dict[str, Any]:
        now = self._now()
        marker = {
            "schema_version": "kabus-kill-switch-v1",
            "engaged_at": format_utc(now),
            "environment": self.config.environment,
            "nonce": secrets.token_hex(16),
            "reason": reason,
        }
        marker_digest = _canonical_digest(marker)
        path = self.state_paths.kill_switch_path
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError as exc:
            raise BrokerSafetyError(f"kill switch is already engaged: {path}") from exc
        try:
            encoded = (
                json.dumps(
                    marker,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short kill-switch marker write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return self._append(
            "broker_kill_switch_engaged",
            {**marker, "marker_digest": marker_digest},
            event_ts=now,
        )

    def release_kill_switch(self, *, confirmation: str) -> dict[str, Any]:
        """Audit an explicit release before removing the risk-off marker."""

        with self._exclusive_operation():
            return self._release_kill_switch_locked(confirmation=confirmation)

    def _release_kill_switch_locked(self, *, confirmation: str) -> dict[str, Any]:
        path = self.state_paths.kill_switch_path
        marker, marker_digest = _read_kill_switch_marker(path)
        self._require_exact_confirmation(
            confirmation,
            f"RELEASE_KILL_SWITCH:{marker_digest}",
        )
        unresolved = self.ledger.unresolved_order_ids()
        if unresolved:
            raise BrokerSafetyError(
                "kill switch cannot be released while broker mutations are unresolved"
            )
        event = self._append(
            "broker_kill_switch_release_authorized",
            {
                "marker_digest": marker_digest,
                "engaged_at": marker["engaged_at"],
                "reason": marker["reason"],
            },
        )
        try:
            path.unlink()
        except OSError as exc:
            raise BrokerSafetyError(
                "failed to release the kill switch; it remains engaged"
            ) from exc
        return event

    def _kill_switch_status(self, *, allow_invalid: bool = False) -> tuple[bool, str | None]:
        path = self.state_paths.kill_switch_path
        if not os.path.lexists(path):
            return False, None
        try:
            _, digest = _read_kill_switch_marker(path)
        except BrokerSafetyError:
            if allow_invalid:
                return True, None
            raise
        return True, digest

    def _preflight(self, intent: CashOrderIntent, *, now: datetime) -> OrderPreflight:
        state = self._account_state()
        account = state
        if account.open_order_count >= self.config.max_open_orders:
            raise BrokerSafetyError("maximum open cash-order count has been reached")
        if account.total_unrealized_profit_loss_jpy <= -self.config.max_total_unrealized_loss_jpy:
            raise BrokerSafetyError("configured total unrealized loss limit is breached")

        same_symbol_open = _same_symbol_open_order_count(
            state.all_product_orders,
            symbol=intent.symbol,
        )
        if same_symbol_open >= 5:
            raise BrokerSafetyError("official concurrent per-symbol order limit has been reached")

        symbol = self.client.get_symbol(intent.symbol, intent.reference_exchange)
        board = self.client.get_board(intent.symbol, intent.reference_exchange)
        _require_identity(symbol, intent.symbol, intent.reference_exchange, "symbol")
        _require_identity(board, intent.symbol, intent.reference_exchange, "board")
        trading_unit = _whole_quantity(symbol.get("TradingUnit"), "TradingUnit", positive=True)
        if intent.quantity % trading_unit != 0:
            raise BrokerSafetyError("order quantity is not a multiple of the broker trading unit")
        _validate_cash_equity_tick(intent.limit_price, symbol.get("PriceRangeGroup"))

        if "PerSymbolLimit" not in symbol:
            raise BrokerSafetyError("broker symbol response is missing PerSymbolLimit")
        raw_symbol_limit = symbol["PerSymbolLimit"]
        broker_symbol_limit = (
            None
            if raw_symbol_limit is None
            else _finite_number(raw_symbol_limit, "PerSymbolLimit", positive=True)
        )

        symbol_wallet = self.client.get_cash_wallet(intent.symbol, intent.exchange)
        cash_buying_power = _configured_cash_buying_power(
            symbol_wallet,
            cash_buy_deliv_type=self.config.cash_buy_deliv_type,
        )

        lower_limit = _finite_number(symbol.get("LowerLimit"), "LowerLimit", positive=True)
        upper_limit = _finite_number(symbol.get("UpperLimit"), "UpperLimit", positive=True)
        if lower_limit > upper_limit:
            raise BrokerSafetyError("symbol price limits are inconsistent")
        # kabuS names these from the trader's perspective: BidPrice is Sell1 (the ask)
        # and AskPrice is Buy1 (the bid), per the official BoardSuccess contract.
        quote_field = "BidPrice" if intent.side == "buy" else "AskPrice"
        quote_time_field = "BidTime" if intent.side == "buy" else "AskTime"
        reference_price = _finite_number(board.get(quote_field), quote_field, positive=True)
        quote_time = _broker_timestamp(board.get(quote_time_field), quote_time_field)
        quote_now = self._now()
        self._validate_quote_freshness(quote_time, now=quote_now)
        execution_price = intent.limit_price
        if not lower_limit <= intent.limit_price <= upper_limit:
            raise BrokerSafetyError("limit price is outside the broker price range")
        deviation_bps = abs(intent.limit_price / reference_price - 1.0) * 10_000.0
        if deviation_bps > self.config.max_price_deviation_bps:
            raise BrokerSafetyError("limit price exceeds the configured quote-deviation limit")

        notional = execution_price * intent.quantity
        if not math.isfinite(notional) or notional <= 0:
            raise BrokerSafetyError("order notional is invalid")
        if notional > self.config.max_order_notional_jpy:
            raise BrokerSafetyError("maximum per-order notional would be exceeded")
        if notional > account.stock_soft_limit_jpy:
            raise BrokerSafetyError("order notional exceeds the kabuStation cash soft limit")

        current_quantity, available_to_sell = _cash_position_quantities(
            state.cash_positions,
            symbol=intent.symbol,
            account_type=self.config.account_type,
        )
        pending_buy = _pending_buy_quantity(
            state.orders,
            symbol=intent.symbol,
        )
        gross_position_notional, symbol_position_notional = _cash_position_notionals(
            state.cash_positions,
            symbol=intent.symbol,
        )
        pending_buy_notional, pending_symbol_buy_notional = _pending_buy_notionals(
            state.orders,
            symbol=intent.symbol,
        )
        if intent.side == "buy":
            if notional > cash_buying_power:
                raise BrokerSafetyError("cash buying power is below the estimated order notional")
            projected = current_quantity + pending_buy + intent.quantity
            if projected > self.config.max_position_quantity_per_symbol:
                raise BrokerSafetyError("maximum configured position quantity would be exceeded")
            if (
                symbol_position_notional + pending_symbol_buy_notional + notional
                > self.config.max_position_notional_jpy_per_symbol
            ):
                raise BrokerSafetyError(
                    "maximum configured per-symbol position notional would be exceeded"
                )
            projected_symbol_notional = (
                symbol_position_notional + pending_symbol_buy_notional + notional
            )
            if broker_symbol_limit is not None and projected_symbol_notional > broker_symbol_limit:
                raise BrokerSafetyError("broker per-symbol position limit would be exceeded")
            if (
                gross_position_notional + pending_buy_notional + notional
                > self.config.max_gross_cash_position_notional_jpy
            ):
                raise BrokerSafetyError(
                    "maximum configured gross cash-position notional would be exceeded"
                )
        elif intent.quantity > available_to_sell:
            raise BrokerSafetyError("available cash position is below the requested sell quantity")

        self._validate_expiry(intent, now=now)
        return OrderPreflight(
            notional_jpy=notional,
            reference_price=reference_price,
            quote_at=quote_time,
            trading_unit=trading_unit,
            current_position_quantity=current_quantity,
            available_to_sell_quantity=available_to_sell,
            pending_buy_quantity=pending_buy,
            current_symbol_position_notional_jpy=symbol_position_notional,
            gross_cash_position_notional_jpy=gross_position_notional,
            pending_buy_notional_jpy=pending_buy_notional,
            cash_buying_power_jpy=cash_buying_power,
            broker_symbol_limit_jpy=broker_symbol_limit,
            account=account,
        )

    def _account_state(self) -> _BrokerAccountState:
        kill_switch_engaged, kill_switch_digest = self._kill_switch_status()
        orders = self.client.get_orders()
        all_product_orders = self.client.get_orders(product=0)
        cash_positions = self.client.get_positions()
        account_positions = self.client.get_positions(product=0)
        wallet = self.client.get_cash_wallet()
        soft_limit = self.client.get_api_soft_limit()

        _validate_cash_order_risk_rows(orders)
        _validate_cash_position_risk_rows(cash_positions)
        states = [_order_state(order) for order in orders]
        open_order_count = sum(state in _OPEN_ORDER_STATES for state in states)
        cash_wallet = _configured_cash_buying_power(
            wallet,
            cash_buy_deliv_type=self.config.cash_buy_deliv_type,
        )
        stock_limit_units = _finite_number(soft_limit.get("Stock"), "Stock", positive=True)
        profit_loss = 0.0
        for position in account_positions:
            profit_loss += _finite_number(position.get("ProfitLoss"), "ProfitLoss")
        if not math.isfinite(profit_loss):
            raise BrokerSafetyError("total unrealized profit/loss is not finite")
        return _BrokerAccountState(
            environment=self.config.environment,
            open_order_count=open_order_count,
            cash_position_count=len(cash_positions),
            account_position_count=len(account_positions),
            cash_wallet_jpy=cash_wallet,
            stock_soft_limit_jpy=stock_limit_units * 10_000.0,
            total_unrealized_profit_loss_jpy=profit_loss,
            unresolved_client_order_ids=self.ledger.unresolved_order_ids(),
            kill_switch_engaged=kill_switch_engaged,
            kill_switch_digest=kill_switch_digest,
            orders=tuple(orders),
            all_product_orders=tuple(all_product_orders),
            cash_positions=tuple(cash_positions),
        )

    def _incident_account_snapshot(
        self,
    ) -> tuple[
        BrokerAccountSnapshot,
        tuple[JsonObject, ...] | None,
        datetime | None,
    ]:
        """Best-effort read-only state for incidents; missing fields stay explicit."""

        unavailable: list[str] = []
        orders: tuple[JsonObject, ...] | None = None
        orders_observed_at: datetime | None = None
        open_order_count: int | None = None
        try:
            read_orders = tuple(self.client.get_orders())
            orders_observed_at = self._now()
            open_order_count = sum(
                _order_state(order) in _OPEN_ORDER_STATES for order in read_orders
            )
            orders = read_orders
        except (KabuSClientError, BrokerSafetyError, ValueError):
            unavailable.append("orders")

        cash_position_count: int | None = None
        try:
            cash_position_count = len(self.client.get_positions())
        except (KabuSClientError, BrokerSafetyError, ValueError):
            unavailable.append("cash_positions")

        account_position_count: int | None = None
        total_profit_loss: float | None = None
        try:
            account_positions = self.client.get_positions(product=0)
            account_position_count = len(account_positions)
            total_profit_loss = sum(
                _finite_number(position.get("ProfitLoss"), "ProfitLoss")
                for position in account_positions
            )
            if not math.isfinite(total_profit_loss):
                raise BrokerSafetyError("account unrealized profit/loss is not finite")
        except (KabuSClientError, BrokerSafetyError, ValueError):
            unavailable.append("account_positions_or_profit_loss")
            total_profit_loss = None

        cash_wallet: float | None = None
        try:
            wallet = self.client.get_cash_wallet()
            cash_wallet = _configured_cash_buying_power(
                wallet,
                cash_buy_deliv_type=self.config.cash_buy_deliv_type,
            )
        except (KabuSClientError, BrokerSafetyError, ValueError):
            unavailable.append("cash_wallet")

        stock_soft_limit: float | None = None
        try:
            soft_limit = self.client.get_api_soft_limit()
            stock_soft_limit = (
                _finite_number(soft_limit.get("Stock"), "Stock", positive=True) * 10_000.0
            )
        except (KabuSClientError, BrokerSafetyError, ValueError):
            unavailable.append("stock_soft_limit")

        kill_switch_engaged, kill_switch_digest = self._kill_switch_status(allow_invalid=True)
        if kill_switch_engaged and kill_switch_digest is None:
            unavailable.append("kill_switch_marker")
        snapshot = BrokerAccountSnapshot(
            environment=self.config.environment,
            open_order_count=open_order_count,
            cash_position_count=cash_position_count,
            account_position_count=account_position_count,
            cash_wallet_jpy=cash_wallet,
            stock_soft_limit_jpy=stock_soft_limit,
            total_unrealized_profit_loss_jpy=total_profit_loss,
            unresolved_client_order_ids=self.ledger.unresolved_order_ids(),
            kill_switch_engaged=kill_switch_engaged,
            kill_switch_digest=kill_switch_digest,
            unavailable_fields=tuple(unavailable),
        )
        return snapshot, orders, orders_observed_at

    def _reconciliation_evidence(
        self,
        orders: tuple[JsonObject, ...] | None,
        *,
        observed_at: datetime,
    ) -> list[dict[str, Any]]:
        events = self.ledger.replay()
        unresolved = set(self.ledger.unresolved_order_ids())
        orders_valid = orders is not None
        if orders is not None:
            try:
                _validate_order_evidence_rows(orders)
            except BrokerSafetyError:
                orders_valid = False
        evidence: list[dict[str, Any]] = []
        for client_order_id in sorted(unresolved):
            attempts = [
                event
                for event in events
                if event.get("client_order_id") == client_order_id
                and event.get("event_type") in {"broker_order_attempt", "broker_cancel_attempt"}
            ]
            if not attempts:
                continue
            attempt = max(attempts, key=lambda event: int(event["sequence"]))
            is_cancel = attempt["event_type"] == "broker_cancel_attempt"
            attempt_environment = attempt.get("environment")
            attempt_config_digest = attempt.get("config_digest")
            scoped = (
                attempt_environment == self.config.environment
                and attempt_config_digest == self.config.digest()
            )
            candidate_ids: list[str] = []
            exact_ids: list[str] = []
            evidence_valid = orders_valid and scoped
            after, window_anchor, before = _reconciliation_match_window(
                events,
                attempt=attempt,
                client_order_id=client_order_id,
                match_seconds=self.config.max_reconciliation_match_seconds,
            )
            if evidence_valid and orders is not None:
                try:
                    if is_cancel:
                        requested_id = attempt.get("broker_order_id")
                        matches = [order for order in orders if order.get("ID") == requested_id]
                        if len(matches) != 1 or not isinstance(requested_id, str):
                            evidence_valid = False
                        else:
                            has_candidate, is_exact = _cancel_reconciliation_state(
                                matches[0],
                                after=after,
                                before=before,
                            )
                            candidate_ids = [requested_id] if has_candidate else []
                            exact_ids = [requested_id] if is_exact else []
                    else:
                        request_payload = attempt.get("effective_request_payload")
                        if not isinstance(request_payload, Mapping):
                            evidence_valid = False
                        else:
                            candidate_ids = _candidate_order_ids(
                                request_payload,
                                orders,
                                after=after,
                                before=before,
                            )
                            exact_ids = [
                                str(order["ID"])
                                for order in orders
                                if isinstance(order.get("ID"), str)
                                and _broker_order_matches(
                                    request_payload,
                                    order,
                                    after=after,
                                    before=before,
                                )
                            ]
                except (BrokerSafetyError, TypeError, ValueError):
                    evidence_valid = False
                    candidate_ids = []
                    exact_ids = []
            evidence.append(
                {
                    "client_order_id": client_order_id,
                    "mutation_type": "cancel" if is_cancel else "submit",
                    "attempt_sequence": int(attempt["sequence"]),
                    "attempt_event_ts": str(attempt["event_ts"]),
                    "attempt_environment": attempt_environment,
                    "attempt_config_digest": attempt_config_digest,
                    "orders_available": orders is not None,
                    "orders_valid": orders_valid,
                    "scope_matches": scoped,
                    "evidence_valid": evidence_valid,
                    "observed_at": format_utc(observed_at),
                    "match_window_anchor": format_utc(window_anchor),
                    "match_window_end": format_utc(before),
                    "negative_observation_complete": evidence_valid and observed_at >= before,
                    "candidate_order_ids": sorted(candidate_ids),
                    "exact_match_order_ids": sorted(exact_ids),
                }
            )
        return evidence

    def _require_fresh_reconciliation_evidence(self, evidence: Mapping[str, Any]) -> None:
        observed_raw = evidence.get("observed_at")
        if not isinstance(observed_raw, str):
            raise BrokerSafetyError("broker reconciliation evidence has no observation time")
        try:
            observed_at = parse_utc(observed_raw)
        except (TypeError, ValueError) as exc:
            raise BrokerSafetyError(
                "broker reconciliation evidence has an invalid observation time"
            ) from exc
        now = self._now()
        if observed_at > now + timedelta(seconds=self.config.max_future_intent_skew_seconds):
            raise BrokerSafetyError("broker reconciliation evidence is dated in the future")
        if (now - observed_at).total_seconds() > self.config.max_reconciliation_match_seconds:
            raise BrokerSafetyError("broker reconciliation evidence is stale; reconcile again")

    def _fresh_resolution_evidence(
        self,
        client_order_id: str,
        attempt: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Re-read every cash order and recompute unique evidence immediately."""

        self._authenticate_client()
        orders = tuple(self.client.get_orders())
        observed_at = self._now()
        matches = [
            item
            for item in self._reconciliation_evidence(orders, observed_at=observed_at)
            if item.get("client_order_id") == client_order_id
            and item.get("attempt_sequence") == int(attempt["sequence"])
            and item.get("attempt_config_digest") == attempt.get("config_digest")
        ]
        if len(matches) != 1 or matches[0].get("evidence_valid") is not True:
            raise BrokerSafetyError("fresh broker order evidence is invalid or incomplete")
        return matches[0]

    def _validate_static_order(self, intent: CashOrderIntent, *, now: datetime) -> None:
        if intent.symbol not in self.config.allowed_symbols:
            raise BrokerSafetyError(f"symbol is not allowlisted: {intent.symbol}")
        if intent.exchange not in self.config.allowed_exchanges:
            raise BrokerSafetyError(f"exchange is not allowlisted: {intent.exchange}")
        if intent.quantity > self.config.max_order_quantity:
            raise BrokerSafetyError("maximum per-order quantity would be exceeded")
        if intent.created_at > now + timedelta(seconds=self.config.max_future_intent_skew_seconds):
            raise BrokerSafetyError("order intent timestamp is in the future")
        age = (now - intent.created_at).total_seconds()
        if age > self.config.max_intent_age_seconds:
            raise BrokerSafetyError("order intent is stale")
        notional = intent.limit_price * intent.quantity
        if notional > self.config.max_order_notional_jpy:
            raise BrokerSafetyError("maximum per-order notional would be exceeded")

    def _validate_expiry(self, intent: CashOrderIntent, *, now: datetime) -> None:
        if intent.expire_day:
            expiry = datetime.strptime(str(intent.expire_day), "%Y%m%d").date()
            if expiry < now.astimezone(_JAPAN).date():
                raise BrokerSafetyError("order expiry is before the current Japan date")

    def _validate_quote_freshness(self, quote_at: datetime, *, now: datetime) -> None:
        if quote_at > now + timedelta(seconds=self.config.max_future_intent_skew_seconds):
            raise BrokerSafetyError("broker quote timestamp is in the future")
        if (now - quote_at).total_seconds() > self.config.max_quote_age_seconds:
            raise BrokerSafetyError("broker quote is older than the configured maximum")

    def _effective_order_payload(self, intent: CashOrderIntent) -> JsonObject:
        return _cash_order_payload(
            intent,
            account_type=self.config.account_type,
            cash_buy_deliv_type=self.config.cash_buy_deliv_type,
            cash_buy_fund_type=self.config.cash_buy_fund_type,
        )

    def _order_approval_digest(
        self,
        intent: CashOrderIntent,
        request_payload: Mapping[str, Any],
    ) -> str:
        envelope = {
            "schema_version": "kabus-order-approval-v1",
            "adapter_version": __version__,
            "effective_config": json.loads(self.config.canonical_json()),
            "effective_request_payload": dict(request_payload),
            "intent": json.loads(intent.canonical_json()),
        }
        return _canonical_digest(envelope)

    def _require_preflight_duration(self, started_monotonic: float) -> None:
        elapsed = self._monotonic() - started_monotonic
        if not math.isfinite(elapsed) or elapsed < 0:
            raise BrokerSafetyError("broker monotonic clock returned an invalid duration")
        if elapsed > self.config.max_preflight_duration_seconds:
            raise BrokerSafetyError("broker preflight exceeded its maximum duration")

    def _require_enabled(self) -> None:
        if not self.config.enabled:
            raise BrokerSafetyError("kabuS broker integration is disabled")

    def _authenticate_client(self) -> None:
        if self._authenticate is not None:
            self._authenticate()

    def _require_order_submission(self, production_confirmation: str | None) -> None:
        self._require_enabled()
        if not self.config.order_submission_enabled:
            raise BrokerSafetyError("kabuS order submission is disabled")
        if self.config.environment == "production":
            if self.config.production_acknowledgement != "REAL_ORDERS":
                raise BrokerSafetyError("production broker acknowledgement is missing")
            if production_confirmation != "REAL_ORDERS":
                raise BrokerSafetyError("production operation requires --confirm-production")
        elif production_confirmation is not None:
            raise BrokerSafetyError("production confirmation is invalid in validation mode")

    def _require_recovery_operation(self, production_confirmation: str | None) -> None:
        self._require_enabled()
        if self.config.environment == "production":
            if production_confirmation != "REAL_ORDERS":
                raise BrokerSafetyError("production operation requires --confirm-production")
        elif production_confirmation is not None:
            raise BrokerSafetyError("production confirmation is invalid in validation mode")

    def _require_kill_switch_clear(self) -> None:
        if os.path.lexists(self.state_paths.kill_switch_path):
            raise BrokerSafetyError("kill switch is engaged; new orders are blocked")

    @staticmethod
    def _require_exact_confirmation(actual: str, expected: str) -> None:
        if actual != expected:
            raise BrokerSafetyError("explicit confirmation does not match the required value")

    def _append(
        self,
        event_type: str,
        values: Mapping[str, Any],
        *,
        event_ts: datetime | None = None,
    ) -> dict[str, Any]:
        return self.ledger.append(
            {
                "event_type": event_type,
                "event_ts": format_utc(event_ts or self._now()),
                "environment": self.config.environment,
                **values,
            }
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise BrokerSafetyError("broker clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    @contextmanager
    def _exclusive_operation(self) -> Iterator[None]:
        """Serialize whole broker operations, not merely individual audit appends."""

        try:
            with advisory_file_lock(self.state_paths.operation_lock_path):
                yield
        except KabuSStateLockError as exc:
            raise BrokerSafetyError("another broker operation already holds the lock") from exc


def _validate_action_id(value: str) -> None:
    if not isinstance(value, str) or _ACTION_ID.fullmatch(value) is None:
        raise BrokerSafetyError("client action ID has an invalid format")


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_kill_switch_marker(path: os.PathLike[str]) -> tuple[dict[str, str], str]:
    marker_path = os.fspath(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(marker_path, flags)
    except FileNotFoundError as exc:
        raise BrokerSafetyError("kill switch is not engaged") from exc
    except OSError as exc:
        raise BrokerSafetyError("kill-switch marker cannot be read safely") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise BrokerSafetyError("kill-switch marker must be a regular file")
        chunks: list[bytes] = []
        remaining = _MAX_KILL_SWITCH_MARKER_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
    except OSError as exc:
        raise BrokerSafetyError("kill-switch marker cannot be read safely") from exc
    finally:
        os.close(descriptor)
    if len(encoded) > _MAX_KILL_SWITCH_MARKER_BYTES or not encoded.endswith(b"\n"):
        raise BrokerSafetyError("kill-switch marker is invalid")
    try:
        parsed = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise BrokerSafetyError("kill-switch marker is invalid") from None
    if not isinstance(parsed, dict) or set(parsed) != {
        "schema_version",
        "engaged_at",
        "environment",
        "nonce",
        "reason",
    }:
        raise BrokerSafetyError("kill-switch marker is invalid")
    if parsed.get("schema_version") != "kabus-kill-switch-v1":
        raise BrokerSafetyError("kill-switch marker is invalid")
    if parsed.get("environment") not in {"validation", "production"}:
        raise BrokerSafetyError("kill-switch marker is invalid")
    reason = parsed.get("reason")
    engaged_at = parsed.get("engaged_at")
    nonce = parsed.get("nonce")
    if (
        not isinstance(reason, str)
        or not reason.strip()
        or reason != reason.strip()
        or len(reason.encode("utf-8")) > 512
        or not isinstance(engaged_at, str)
        or not isinstance(nonce, str)
        or re.fullmatch(r"[0-9a-f]{32}", nonce) is None
    ):
        raise BrokerSafetyError("kill-switch marker is invalid")
    try:
        canonical_engaged_at = format_utc(parse_utc(engaged_at))
    except (TypeError, ValueError):
        raise BrokerSafetyError("kill-switch marker is invalid") from None
    if canonical_engaged_at != engaged_at:
        raise BrokerSafetyError("kill-switch marker is invalid")
    marker = {
        "schema_version": "kabus-kill-switch-v1",
        "engaged_at": engaged_at,
        "environment": str(parsed["environment"]),
        "nonce": nonce,
        "reason": reason,
    }
    canonical = json.dumps(marker, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if encoded != (canonical + "\n").encode("utf-8"):
        raise BrokerSafetyError("kill-switch marker is not canonical")
    return marker, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_json_constant(_: str) -> None:
    raise ValueError


def _finite_number(
    value: object,
    field_name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BrokerSafetyError(f"broker response has invalid {field_name}")
    result = float(value)
    if not math.isfinite(result):
        raise BrokerSafetyError(f"broker response has non-finite {field_name}")
    if positive and result <= 0:
        raise BrokerSafetyError(f"broker response requires positive {field_name}")
    if non_negative and result < 0:
        raise BrokerSafetyError(f"broker response requires non-negative {field_name}")
    return result


def _configured_cash_buying_power(
    wallet: Mapping[str, Any],
    *,
    cash_buy_deliv_type: int,
) -> float:
    if cash_buy_deliv_type == 2:
        field_name = "AuKCStockAccountWallet"
    elif cash_buy_deliv_type == 3:
        field_name = "StockAccountWallet"
    else:  # pragma: no cover - KabuSConfig validates the literal
        raise BrokerSafetyError("cash buy delivery type is invalid")
    return _finite_number(wallet.get(field_name), field_name, non_negative=True)


def _validate_cash_equity_tick(limit_price: float, price_range_group: object) -> None:
    if not isinstance(price_range_group, str):
        raise BrokerSafetyError("broker response has invalid PriceRangeGroup")
    table = _CASH_EQUITY_TICK_TABLES.get(price_range_group)
    if table is None:
        raise BrokerSafetyError("broker PriceRangeGroup is not supported for cash equities")
    try:
        price = Decimal(str(limit_price))
    except InvalidOperation as exc:  # pragma: no cover - intent validation rejects this
        raise BrokerSafetyError("limit price is invalid") from exc
    for upper_bound, tick in table:
        if upper_bound is None or price <= upper_bound:
            if price % tick != 0:
                raise BrokerSafetyError(
                    "limit price is not aligned to the broker PriceRangeGroup tick size"
                )
            return
    raise BrokerSafetyError("broker PriceRangeGroup tick table is incomplete")  # pragma: no cover


def _whole_quantity(value: object, field_name: str, *, positive: bool = False) -> int:
    number = _finite_number(value, field_name, positive=positive, non_negative=not positive)
    if not number.is_integer():
        raise BrokerSafetyError(f"broker response has non-integral {field_name}")
    return int(number)


def _broker_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise BrokerSafetyError(f"broker response has invalid {field_name}")
    try:
        return parse_utc(value)
    except (TypeError, ValueError) as exc:
        raise BrokerSafetyError(f"broker response has invalid {field_name}") from exc


def _order_state(order: Mapping[str, Any]) -> int:
    state = order.get("State")
    if type(state) is not int or state not in {*_OPEN_ORDER_STATES, _TERMINAL_ORDER_STATE}:
        raise BrokerSafetyError("broker order response has invalid State")
    order_state = order.get("OrderState")
    if order_state is not None and order_state != state:
        raise BrokerSafetyError("broker State and OrderState disagree")
    return state


def _require_identity(
    payload: Mapping[str, Any],
    symbol: str,
    exchange: int,
    response_name: str,
) -> None:
    if payload.get("Symbol") != symbol or payload.get("Exchange") != exchange:
        raise BrokerSafetyError(f"broker {response_name} identity does not match the order")


def _cash_position_quantities(
    positions: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    account_type: int,
) -> tuple[int, int]:
    total = 0
    available = 0
    for position in positions:
        if position.get("Symbol") != symbol:
            continue
        position_account_type = position.get("AccountType")
        if type(position_account_type) is not int or position_account_type not in (2, 4, 12):
            raise BrokerSafetyError("broker cash position has invalid AccountType")
        side = position.get("Side")
        if side not in ("1", "2"):
            raise BrokerSafetyError("broker cash position has invalid Side")
        if side != "2":
            continue
        leaves = _whole_quantity(position.get("LeavesQty"), "LeavesQty")
        held = _whole_quantity(position.get("HoldQty"), "HoldQty")
        if held > leaves:
            raise BrokerSafetyError("broker cash position HoldQty exceeds LeavesQty")
        total += leaves
        if position_account_type == account_type:
            available += leaves - held
    return total, available


def _same_symbol_open_order_count(
    orders: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
) -> int:
    count = 0
    for order in orders:
        state = _order_state(order)
        order_symbol = order.get("Symbol")
        if not isinstance(order_symbol, str) or _SYMBOL.fullmatch(order_symbol) is None:
            raise BrokerSafetyError("broker all-product order has an invalid Symbol")
        if state in _OPEN_ORDER_STATES and order_symbol == symbol:
            count += 1
    return count


def _validate_cash_order_risk_rows(orders: Sequence[Mapping[str, Any]]) -> None:
    for order in orders:
        state = _order_state(order)
        if state not in _OPEN_ORDER_STATES:
            continue
        symbol = order.get("Symbol")
        if not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None:
            raise BrokerSafetyError("broker open cash order has an invalid Symbol")
        if order.get("Side") not in ("1", "2"):
            raise BrokerSafetyError("broker open cash order has an invalid Side")
        account_type = order.get("AccountType")
        if type(account_type) is not int or account_type not in (2, 4, 12):
            raise BrokerSafetyError("broker open cash order has an invalid AccountType")
        quantity = _whole_quantity(order.get("OrderQty"), "OrderQty", positive=True)
        cumulative = _whole_quantity(order.get("CumQty"), "CumQty")
        if cumulative > quantity:
            raise BrokerSafetyError("broker cash order CumQty exceeds OrderQty")
        if order.get("Side") == "2" and cumulative < quantity:
            _finite_number(order.get("Price"), "Price", positive=True)


def _validate_cash_position_risk_rows(positions: Sequence[Mapping[str, Any]]) -> None:
    for position in positions:
        leaves = _whole_quantity(position.get("LeavesQty"), "LeavesQty")
        held = _whole_quantity(position.get("HoldQty"), "HoldQty")
        if held > leaves:
            raise BrokerSafetyError("broker cash position HoldQty exceeds LeavesQty")
        if leaves == 0:
            continue
        symbol = position.get("Symbol")
        if not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None:
            raise BrokerSafetyError("broker nonzero cash position has an invalid Symbol")
        if position.get("Side") not in ("1", "2"):
            raise BrokerSafetyError("broker nonzero cash position has an invalid Side")
        account_type = position.get("AccountType")
        if type(account_type) is not int or account_type not in (2, 4, 12):
            raise BrokerSafetyError("broker nonzero cash position has an invalid AccountType")
        _finite_number(position.get("CurrentPrice"), "CurrentPrice", positive=True)


def _pending_buy_quantity(
    orders: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
) -> int:
    pending = 0
    for order in orders:
        if (
            _order_state(order) not in _OPEN_ORDER_STATES
            or order.get("Symbol") != symbol
            or order.get("Side") != "2"
        ):
            continue
        order_account_type = order.get("AccountType")
        if type(order_account_type) is not int or order_account_type not in (2, 4, 12):
            raise BrokerSafetyError("broker cash order has invalid AccountType")
        order_quantity = _whole_quantity(order.get("OrderQty"), "OrderQty")
        cumulative = _whole_quantity(order.get("CumQty"), "CumQty")
        if cumulative > order_quantity:
            raise BrokerSafetyError("broker cash order CumQty exceeds OrderQty")
        pending += order_quantity - cumulative
    return pending


def _reconciliation_match_window(
    events: Sequence[Mapping[str, Any]],
    *,
    attempt: Mapping[str, Any],
    client_order_id: str,
    match_seconds: int,
) -> tuple[datetime, datetime, datetime]:
    attempt_at = parse_utc(str(attempt["event_ts"]))
    attempt_sequence = int(attempt["sequence"])
    unknown_event_type = (
        "broker_cancel_unknown"
        if attempt.get("event_type") == "broker_cancel_attempt"
        else "broker_order_unknown"
    )
    anchor = attempt_at
    for event in events:
        if (
            event.get("client_order_id") != client_order_id
            or event.get("event_type") != unknown_event_type
            or int(event["sequence"]) <= attempt_sequence
        ):
            continue
        event_at = parse_utc(str(event["event_ts"]))
        if event_at > anchor:
            anchor = event_at
    return attempt_at, anchor, anchor + timedelta(seconds=match_seconds)


def _cash_position_notionals(
    positions: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
) -> tuple[float, float]:
    gross = 0.0
    matching_symbol = 0.0
    for position in positions:
        leaves = _whole_quantity(position.get("LeavesQty"), "LeavesQty")
        if leaves == 0:
            continue
        current_price = _finite_number(
            position.get("CurrentPrice"),
            "CurrentPrice",
            positive=True,
        )
        value = leaves * current_price
        if not math.isfinite(value):
            raise BrokerSafetyError("cash position notional is not finite")
        gross += value
        if position.get("Symbol") == symbol:
            matching_symbol += value
    if not math.isfinite(gross) or not math.isfinite(matching_symbol):
        raise BrokerSafetyError("cash position notional total is not finite")
    return gross, matching_symbol


def _pending_buy_notionals(
    orders: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
) -> tuple[float, float]:
    total = 0.0
    matching_symbol = 0.0
    for order in orders:
        if _order_state(order) not in _OPEN_ORDER_STATES or order.get("Side") != "2":
            continue
        order_quantity = _whole_quantity(order.get("OrderQty"), "OrderQty")
        cumulative = _whole_quantity(order.get("CumQty"), "CumQty")
        if cumulative > order_quantity:
            raise BrokerSafetyError("broker cash order CumQty exceeds OrderQty")
        remaining = order_quantity - cumulative
        if remaining == 0:
            continue
        price = _finite_number(order.get("Price"), "Price", positive=True)
        value = remaining * price
        if not math.isfinite(value):
            raise BrokerSafetyError("pending cash-buy notional is not finite")
        total += value
        if order.get("Symbol") == symbol:
            matching_symbol += value
    if not math.isfinite(total) or not math.isfinite(matching_symbol):
        raise BrokerSafetyError("pending cash-buy notional total is not finite")
    return total, matching_symbol


def _order_has_cancel_record(
    order: Mapping[str, Any],
    *,
    after: datetime,
    before: datetime,
) -> bool:
    _, is_exact = _cancel_reconciliation_state(order, after=after, before=before)
    return is_exact


def _cancel_reconciliation_state(
    order: Mapping[str, Any],
    *,
    after: datetime,
    before: datetime,
) -> tuple[bool, bool]:
    top_level_state = _order_state(order)
    details = order.get("Details")
    if not isinstance(details, list):
        raise BrokerSafetyError("broker order Details are required for cancel reconciliation")
    has_cancel_activity = False
    has_processed_cancel = False
    for detail in details:
        if not isinstance(detail, Mapping):
            raise BrokerSafetyError("broker cancel detail must be an object")
        if detail.get("RecType") != 6:
            continue
        state = detail.get("State")
        if type(state) is not int or state not in (1, 2, 3, 4, 5):
            raise BrokerSafetyError("broker cancel detail has an invalid State")
        transact_time = _broker_timestamp(detail.get("TransactTime"), "TransactTime")
        if after - timedelta(seconds=1) <= transact_time <= before:
            has_cancel_activity = True
            has_processed_cancel = has_processed_cancel or state == 3
    # Only an unchanged, processed order with no cancellation detail supports a
    # negative conclusion. Every other state remains a possible cancellation.
    return (
        top_level_state != 3 or has_cancel_activity,
        top_level_state == _TERMINAL_ORDER_STATE and has_processed_cancel,
    )


def _validate_order_evidence_rows(orders: Sequence[Mapping[str, Any]]) -> None:
    seen_ids: set[str] = set()
    for order in orders:
        order_id = order.get("ID")
        if not isinstance(order_id, str) or _ACTION_ID.fullmatch(order_id) is None:
            raise BrokerSafetyError("broker order evidence has an invalid ID")
        normalized_id = order_id.casefold()
        if normalized_id in seen_ids:
            raise BrokerSafetyError("broker order evidence repeats an ID")
        seen_ids.add(normalized_id)
        _broker_timestamp(order.get("RecvTime"), "RecvTime")
        _order_state(order)
        symbol = order.get("Symbol")
        if not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None:
            raise BrokerSafetyError("broker order evidence has an invalid Symbol")
        if order.get("Side") not in ("1", "2"):
            raise BrokerSafetyError("broker order evidence has an invalid Side")
        account_type = order.get("AccountType")
        if type(account_type) is not int or account_type not in (2, 4, 12):
            raise BrokerSafetyError("broker order evidence has an invalid AccountType")
        exchange = order.get("Exchange")
        if type(exchange) is not int or exchange not in (1, 3, 5, 6, 9, 27):
            raise BrokerSafetyError("broker order evidence has an invalid Exchange")


def _cancel_order_context(order: Mapping[str, Any]) -> dict[str, Any]:
    order_id = order.get("ID")
    if not isinstance(order_id, str) or _ACTION_ID.fullmatch(order_id) is None:
        raise BrokerSafetyError("broker cash order has an invalid ID")
    symbol = order.get("Symbol")
    if not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None:
        raise BrokerSafetyError("broker cash order has an invalid Symbol")
    exchange = order.get("Exchange")
    if type(exchange) is not int or exchange not in (1, 3, 5, 6, 9, 27):
        raise BrokerSafetyError("broker cash order has an invalid Exchange")
    side = order.get("Side")
    if side not in ("1", "2"):
        raise BrokerSafetyError("broker cash order has an invalid Side")
    if order.get("CashMargin") != 1:
        raise BrokerSafetyError("broker cancellation target is not a cash order")
    account_type = order.get("AccountType")
    if type(account_type) is not int or account_type not in (2, 4, 12):
        raise BrokerSafetyError("broker cash order has an invalid AccountType")
    deliv_type = order.get("DelivType")
    if type(deliv_type) is not int or deliv_type not in (0, 2, 3):
        raise BrokerSafetyError("broker cash order has an invalid DelivType")
    quantity = _whole_quantity(order.get("OrderQty"), "OrderQty", positive=True)
    cumulative = _whole_quantity(order.get("CumQty"), "CumQty")
    if cumulative > quantity:
        raise BrokerSafetyError("broker cash order CumQty exceeds OrderQty")
    price = _finite_number(order.get("Price"), "Price", non_negative=True)
    ord_type = order.get("OrdType")
    if type(ord_type) is not int or ord_type not in (1, 2, 3, 4, 5, 6):
        raise BrokerSafetyError("broker cash order has an invalid OrdType")
    expire_day = _whole_quantity(order.get("ExpireDay"), "ExpireDay")
    received_at = _broker_timestamp(order.get("RecvTime"), "RecvTime")
    return {
        "ID": order_id,
        "State": _order_state(order),
        "RecvTime": format_utc(received_at),
        "Symbol": symbol,
        "Exchange": exchange,
        "Price": price,
        "OrdType": ord_type,
        "OrderQty": quantity,
        "CumQty": cumulative,
        "Side": side,
        "CashMargin": 1,
        "AccountType": account_type,
        "DelivType": deliv_type,
        "ExpireDay": expire_day,
    }


def _broker_order_matches(
    request_payload: Mapping[str, Any],
    broker: Mapping[str, Any],
    *,
    after: datetime,
    before: datetime,
) -> bool:
    try:
        broker_quantity = _whole_quantity(broker.get("OrderQty"), "OrderQty", positive=True)
        broker_price = _finite_number(broker.get("Price"), "Price", non_negative=True)
        broker_expire_day = _whole_quantity(broker.get("ExpireDay"), "ExpireDay")
        received_at = _broker_timestamp(broker.get("RecvTime"), "RecvTime")
        _order_state(broker)
    except BrokerSafetyError:
        return False
    return (
        after - timedelta(seconds=1) <= received_at <= before
        and broker.get("Symbol") == request_payload.get("Symbol")
        and broker.get("Exchange") == request_payload.get("Exchange")
        and broker.get("Side") == request_payload.get("Side")
        and broker.get("CashMargin") == request_payload.get("CashMargin") == 1
        and broker.get("AccountType") == request_payload.get("AccountType")
        and broker.get("DelivType") == request_payload.get("DelivType")
        and broker.get("OrdType") == 1
        and broker_quantity == request_payload.get("Qty")
        and broker_price == request_payload.get("Price")
        and broker_expire_day == request_payload.get("ExpireDay")
    )


def _candidate_order_ids(
    request_payload: Mapping[str, Any],
    orders: Sequence[Mapping[str, Any]],
    *,
    after: datetime,
    before: datetime,
) -> list[str]:
    candidates: list[str] = []
    for order in orders:
        order_id = order.get("ID")
        if not isinstance(order_id, str):  # pragma: no cover - evidence rows validated first
            raise BrokerSafetyError("broker order evidence has an invalid ID")
        received_at = _broker_timestamp(order.get("RecvTime"), "RecvTime")
        if (
            after - timedelta(seconds=1) <= received_at <= before
            and order.get("Symbol") == request_payload.get("Symbol")
            and order.get("Side") == request_payload.get("Side")
            and order.get("AccountType") == request_payload.get("AccountType")
        ):
            candidates.append(order_id)
    return candidates


def _resolution_common(attempt: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "client_order_id",
        "intent_digest",
        "approval_digest",
        "config_digest",
        "adapter_version",
        "effective_config",
        "effective_request_payload",
        "broker_order",
        "order",
        "notional_jpy",
        "order_date",
    )
    return {name: attempt[name] for name in names if name in attempt}
