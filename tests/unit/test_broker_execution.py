from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from nlp_trader.broker.audit import BrokerAuditLedger
from nlp_trader.broker.config import KabuSConfig
from nlp_trader.broker.contracts import SCHEMA_VERSION, CashOrderIntent
from nlp_trader.broker.execution import BrokerSafetyError, BrokerSubmission, KabuSExecutor
from nlp_trader.broker.kabus import (
    AmbiguousMutationError,
    KabuSAPIRejection,
    KabuSTransportError,
    OrderResult,
)
from nlp_trader.broker.state import KabuSStatePaths, advisory_file_lock

NOW = datetime(2026, 7, 15, 3, 0, tzinfo=UTC)


class FakeKabuSClient:
    """Small protocol-compatible fake using documented kabuS response fields."""

    def __init__(self, environment: str = "validation") -> None:
        self.environment = environment
        self.orders: list[dict[str, Any]] = []
        self.account_orders: list[dict[str, Any]] | None = None
        self.positions: list[dict[str, Any]] = []
        self.account_positions: list[dict[str, Any]] | None = None
        self.wallet: dict[str, Any] = {
            "StockAccountWallet": 1_000_000.0,
            "AuKCStockAccountWallet": 1_000_000.0,
        }
        self.symbol_wallet: dict[str, Any] = dict(self.wallet)
        self.soft_limit: dict[str, Any] = {"Stock": 100.0}
        self.symbol: dict[str, Any] = {
            "Symbol": "9433",
            "Exchange": 1,
            "TradingUnit": 100.0,
            "LowerLimit": 500.0,
            "UpperLimit": 2_000.0,
            "PriceRangeGroup": "10000",
            "PerSymbolLimit": 10_000_000.0,
        }
        self.board: dict[str, Any] = {
            "Symbol": "9433",
            "Exchange": 1,
            "BidPrice": 1_000.0,
            "AskPrice": 1_010.0,
            "BidTime": (NOW - timedelta(seconds=1)).isoformat(),
            "AskTime": (NOW - timedelta(seconds=1)).isoformat(),
        }
        self.orders_by_id: dict[str, list[dict[str, Any]]] = {}
        self.send_results: list[OrderResult | Exception] = [OrderResult(0, "ORDER-001")]
        self.cancel_results: list[OrderResult | Exception] = [OrderResult(0, "ORDER-OPEN")]
        self.before_send: Callable[[], None] | None = None
        self.before_cancel: Callable[[], None] | None = None
        self.calls: list[tuple[str, object]] = []

    def get_orders(
        self,
        order_id: str | None = None,
        *,
        product: int = 1,
    ) -> list[dict[str, Any]]:
        self.calls.append(("get_orders", (order_id, product)))
        if order_id is not None:
            return list(self.orders_by_id.get(order_id, ()))
        if product == 0 and self.account_orders is not None:
            return list(self.account_orders)
        return list(self.orders)

    def get_positions(self, *, product: int = 1) -> list[dict[str, Any]]:
        self.calls.append(("get_positions", product))
        if product == 0 and self.account_positions is not None:
            return list(self.account_positions)
        return list(self.positions)

    def get_cash_wallet(
        self,
        symbol: str | None = None,
        reference_exchange: int | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("get_cash_wallet", (symbol, reference_exchange)))
        return dict(self.symbol_wallet if symbol is not None else self.wallet)

    def get_api_soft_limit(self) -> dict[str, Any]:
        self.calls.append(("get_api_soft_limit", None))
        return dict(self.soft_limit)

    def get_symbol(self, symbol: str, reference_exchange: int) -> dict[str, Any]:
        self.calls.append(("get_symbol", (symbol, reference_exchange)))
        return dict(self.symbol)

    def get_board(self, symbol: str, reference_exchange: int) -> dict[str, Any]:
        self.calls.append(("get_board", (symbol, reference_exchange)))
        return dict(self.board)

    def send_cash_order(
        self,
        intent: CashOrderIntent,
        *,
        account_type: int,
        cash_buy_deliv_type: int,
        cash_buy_fund_type: str,
    ) -> OrderResult:
        if self.before_send is not None:
            self.before_send()
        self.calls.append(
            (
                "send_cash_order",
                (intent, account_type, cash_buy_deliv_type, cash_buy_fund_type),
            )
        )
        result = self.send_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def cancel_order(self, order_id: str) -> OrderResult:
        if self.before_cancel is not None:
            self.before_cancel()
        self.calls.append(("cancel_order", order_id))
        result = self.cancel_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _config(tmp_path: Path, **changes: object) -> KabuSConfig:
    payload: dict[str, object] = {
        "schema_version": "kabus-broker-v1",
        "provider": "kabus",
        "environment": "validation",
        "enabled": True,
        "order_submission_enabled": True,
        "production_acknowledgement": None,
        "single_user_private_use": True,
        "account_type": 4,
        "cash_buy_deliv_type": 2,
        "cash_buy_fund_type": "02",
        "allowed_symbols": ["9433"],
        "allowed_exchanges": [27],
        "allow_market_orders": False,
        "max_order_quantity": 1_000,
        "max_order_notional_jpy": 500_000.0,
        "max_daily_order_notional_jpy": 1_000_000.0,
        "max_position_quantity_per_symbol": 2_000,
        "max_position_notional_jpy_per_symbol": 2_000_000.0,
        "max_gross_cash_position_notional_jpy": 5_000_000.0,
        "max_total_unrealized_loss_jpy": 100_000.0,
        "max_open_orders": 10,
        "max_intent_age_seconds": 60,
        "max_future_intent_skew_seconds": 5,
        "max_preflight_duration_seconds": 15,
        "max_reconciliation_match_seconds": 30,
        "max_quote_age_seconds": 30,
        "max_price_deviation_bps": 500.0,
        "timeout_seconds": 5.0,
    }
    payload.update(changes)
    return KabuSConfig.model_validate(payload)


def _intent(**changes: object) -> CashOrderIntent:
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "client_order_id": "client-001",
        "strategy_id": "strategy-001",
        "created_at": NOW - timedelta(seconds=1),
        "symbol": "9433",
        "exchange": 27,
        "reference_exchange": 1,
        "side": "buy",
        "quantity": 100,
        "order_type": "limit",
        "limit_price": 1_000.0,
        "expire_day": 20260716,
    }
    payload.update(changes)
    return CashOrderIntent.model_validate(payload)


def _executor(
    tmp_path: Path,
    *,
    config_changes: dict[str, object] | None = None,
    authenticate: Callable[[], None] | None = None,
    clock: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> tuple[KabuSExecutor, FakeKabuSClient, BrokerAuditLedger]:
    config = _config(tmp_path, **(config_changes or {}))
    client = FakeKabuSClient(config.environment)
    state_paths = KabuSStatePaths((tmp_path / "broker-state").resolve())
    ledger = BrokerAuditLedger(state_paths.audit_ledger_path)
    executor = KabuSExecutor(
        config,
        client,  # type: ignore[arg-type]
        ledger,
        state_paths=state_paths,
        clock=clock or (lambda: NOW),
        monotonic=monotonic or (lambda: 0.0),
        authenticate=authenticate,
    )
    return executor, client, ledger


def _submit(
    executor: KabuSExecutor,
    intent: CashOrderIntent,
    *,
    production_confirmation: str | None = None,
) -> BrokerSubmission:
    return executor.submit_order(
        intent,
        confirmation=str(executor.preview_order(intent)["confirmation_digest"]),
        production_confirmation=production_confirmation,
    )


def _cancel_confirmation(
    executor: KabuSExecutor,
    order_id: str,
    client_action_id: str,
) -> str:
    return str(
        executor.preview_cancel(
            order_id,
            client_action_id=client_action_id,
        )["confirmation_digest"]
    )


def _open_order(
    *,
    symbol: str = "9433",
    side: str = "2",
    quantity: float = 100.0,
    cumulative: float = 0.0,
    account_type: int = 4,
) -> dict[str, Any]:
    return {
        "ID": "OPEN-001",
        "State": 1,
        "OrderState": 1,
        "Symbol": symbol,
        "Exchange": 27,
        "Side": side,
        "AccountType": account_type,
        "OrderQty": quantity,
        "CumQty": cumulative,
        "Price": 1_000.0,
        "CashMargin": 1,
        "OrdType": 1,
        "DelivType": 2 if side == "2" else 0,
        "RecvTime": NOW.isoformat(),
        "ExpireDay": 20260716,
        "Details": [],
    }


def _account_open_order(
    index: int,
    *,
    symbol: object = "9433",
    state: object = 1,
) -> dict[str, Any]:
    return {
        "ID": f"MARGIN-{index:03d}",
        "State": state,
        "OrderState": state,
        "Symbol": symbol,
        "CashMargin": 2,
        "RecvTime": NOW.isoformat(),
    }


def _long_position(
    *,
    leaves: float = 100.0,
    held: float = 0.0,
    profit_loss: float = 0.0,
    account_type: int = 4,
) -> dict[str, Any]:
    return {
        "Symbol": "9433",
        "Exchange": 1,
        "Side": "2",
        "AccountType": account_type,
        "LeavesQty": leaves,
        "HoldQty": held,
        "ProfitLoss": profit_loss,
        "CurrentPrice": 1_000.0,
    }


def _matching_broker_order(
    order_id: str = "ORDER-UNKNOWN",
    *,
    state: int = 3,
) -> dict[str, Any]:
    return {
        "ID": order_id,
        "State": state,
        "OrderState": state,
        "Symbol": "9433",
        "Exchange": 27,
        "Side": "2",
        "AccountType": 4,
        "OrderQty": 100.0,
        "CumQty": 0.0,
        "Price": 1_000.0,
        "CashMargin": 1,
        "OrdType": 1,
        "DelivType": 2,
        "RecvTime": NOW.isoformat(),
        "ExpireDay": 20260716,
        "Details": [],
    }


def _record_unknown_cancel(
    executor: KabuSExecutor,
    client: FakeKabuSClient,
    *,
    order_id: str = "ORDER-OPEN",
    client_action_id: str = "cancel-001",
) -> dict[str, Any]:
    order = _matching_broker_order(order_id, state=3)
    client.orders_by_id[order_id] = [order]
    confirmation = _cancel_confirmation(executor, order_id, client_action_id)
    client.cancel_results = [AmbiguousMutationError("cancel order")]
    with pytest.raises(AmbiguousMutationError):
        executor.cancel_order(
            order_id,
            client_action_id=client_action_id,
            confirmation=confirmation,
        )
    return order


def _call_names(client: FakeKabuSClient) -> list[str]:
    return [name for name, _ in client.calls]


def _operation_lock_path(ledger: BrokerAuditLedger) -> Path:
    return KabuSStatePaths(ledger.path.parent.resolve()).operation_lock_path


def test_preview_returns_confirmation_digest_without_network_or_audit(tmp_path: Path) -> None:
    executor, client, ledger = _executor(tmp_path)
    intent = _intent()

    preview = executor.preview_order(intent)

    assert preview["environment"] == "validation"
    assert preview["production_places_real_orders"] is False
    assert preview["intent_digest"] == intent.confirmation_digest()
    assert preview["config_digest"] == executor.config.digest()
    assert preview["confirmation_digest"] != intent.confirmation_digest()
    assert preview["intent"] == {
        "schema_version": SCHEMA_VERSION,
        "client_order_id": "client-001",
        "strategy_id": "strategy-001",
        "created_at": "2026-07-15T02:59:59Z",
        "symbol": "9433",
        "exchange": 27,
        "reference_exchange": 1,
        "side": "buy",
        "quantity": 100,
        "order_type": "limit",
        "limit_price": 1_000.0,
        "expire_day": 20260716,
    }
    assert preview["effective_request_payload"]["AccountType"] == 4
    assert preview["effective_request_payload"]["FundType"] == "02"
    assert preview["limit_notional_jpy"] == 100_000.0
    assert preview["network_request_made"] is False
    assert client.calls == []
    assert ledger.replay() == []


def test_submission_enablement_confirmation_and_environment_gates(tmp_path: Path) -> None:
    disabled, disabled_client, disabled_ledger = _executor(
        tmp_path / "disabled",
        config_changes={"order_submission_enabled": False},
    )
    intent = _intent()
    with pytest.raises(BrokerSafetyError, match="order submission is disabled"):
        _submit(disabled, intent)
    assert disabled_client.calls == []
    assert disabled_ledger.replay() == []

    validation, validation_client, validation_ledger = _executor(tmp_path / "validation")
    with pytest.raises(BrokerSafetyError, match="explicit confirmation"):
        validation.submit_order(intent, confirmation="not-the-digest")
    with pytest.raises(BrokerSafetyError, match="invalid in validation"):
        _submit(validation, intent, production_confirmation="REAL_ORDERS")
    assert validation_client.calls == []
    assert validation_ledger.replay() == []

    production, production_client, production_ledger = _executor(
        tmp_path / "production",
        config_changes={
            "environment": "production",
            "production_acknowledgement": "REAL_ORDERS",
        },
    )
    with pytest.raises(BrokerSafetyError, match="confirm-production"):
        _submit(production, intent)
    assert production_client.calls == []
    assert production_ledger.replay() == []

    submission = _submit(production, intent, production_confirmation="REAL_ORDERS")
    assert submission.environment == "production"
    assert _call_names(production_client).count("send_cash_order") == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"created_at": NOW - timedelta(seconds=61)}, "stale"),
        ({"created_at": NOW + timedelta(seconds=6)}, "future"),
        ({"symbol": "8306"}, "symbol is not allowlisted"),
        ({"quantity": 1_001}, "maximum per-order quantity"),
    ],
)
def test_preview_rejects_static_safety_violations_without_network(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    executor, client, ledger = _executor(tmp_path)

    with pytest.raises(BrokerSafetyError, match=message):
        executor.preview_order(_intent(**changes))

    assert client.calls == []
    assert ledger.replay() == []


def test_full_buy_preflight_records_attempt_before_exact_single_send(tmp_path: Path) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.orders = [_open_order(quantity=100.0, cumulative=20.0)]
    client.positions = [_long_position(leaves=100.0)]
    intent = _intent()
    operation_lock = _operation_lock_path(ledger)

    def assert_durable_attempt() -> None:
        assert operation_lock.is_file()
        events = ledger.replay()
        assert [event["event_type"] for event in events] == [
            "broker_order_preflight",
            "broker_order_attempt",
        ]
        assert events[-1]["client_order_id"] == intent.client_order_id
        assert events[-1]["intent_digest"] == intent.confirmation_digest()

    client.before_send = assert_durable_attempt
    submission = _submit(executor, intent)

    assert submission.to_dict() == {
        "client_order_id": "client-001",
        "broker_order_id": "ORDER-001",
        "intent_digest": intent.confirmation_digest(),
        "environment": "validation",
        "notional_jpy": 100_000.0,
    }
    assert _call_names(client) == [
        "get_orders",
        "get_orders",
        "get_positions",
        "get_positions",
        "get_cash_wallet",
        "get_api_soft_limit",
        "get_symbol",
        "get_board",
        "get_cash_wallet",
        "send_cash_order",
    ]
    sent = client.calls[-1][1]
    assert sent == (intent, 4, 2, "02")
    assert operation_lock.exists()
    events = ledger.replay()
    assert [event["event_type"] for event in events] == [
        "broker_order_preflight",
        "broker_order_attempt",
        "broker_order_accepted",
    ]
    assert events[0]["checks"] == {
        "notional_jpy": 100_000.0,
        "reference_price": 1_000.0,
        "quote_at": "2026-07-15T02:59:59Z",
        "trading_unit": 100,
        "current_position_quantity": 100,
        "available_to_sell_quantity": 100,
        "pending_buy_quantity": 80,
        "current_symbol_position_notional_jpy": 100_000.0,
        "gross_cash_position_notional_jpy": 100_000.0,
        "pending_buy_notional_jpy": 80_000.0,
        "cash_buying_power_jpy": 1_000_000.0,
        "broker_symbol_limit_jpy": 10_000_000.0,
        "open_order_count": 1,
        "cash_check_passed": True,
        "position_check_passed": True,
        "loss_check_passed": True,
        "soft_limit_check_passed": True,
    }


def test_submission_reads_cash_and_all_product_order_snapshots(tmp_path: Path) -> None:
    executor, client, _ = _executor(tmp_path)

    _submit(executor, _intent())

    assert client.calls[:2] == [
        ("get_orders", (None, 1)),
        ("get_orders", (None, 0)),
    ]


def test_five_cross_product_open_orders_for_symbol_block_cash_submission(
    tmp_path: Path,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.orders = []
    client.account_orders = [_account_open_order(index) for index in range(5)]

    with pytest.raises(BrokerSafetyError, match="concurrent per-symbol order limit"):
        _submit(executor, _intent())

    assert client.calls[:2] == [
        ("get_orders", (None, 1)),
        ("get_orders", (None, 0)),
    ]
    assert "send_cash_order" not in _call_names(client)
    assert ledger.replay() == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("State", "3", "State"),
        ("State", 6, "State"),
        ("Symbol", None, "Symbol"),
        ("Symbol", "9433 ", "Symbol"),
    ],
)
def test_malformed_all_product_order_state_or_symbol_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    malformed = _account_open_order(1)
    malformed[field] = value
    client.account_orders = [malformed]

    with pytest.raises(BrokerSafetyError, match=message):
        _submit(executor, _intent())

    assert "send_cash_order" not in _call_names(client)
    assert ledger.replay() == []


@pytest.mark.parametrize(
    ("field", "remove", "invalid_value"),
    [
        ("Symbol", True, None),
        ("Symbol", False, "9433 "),
        ("Side", True, None),
        ("Side", False, "3"),
        ("AccountType", True, None),
        ("AccountType", False, 3),
    ],
)
def test_any_malformed_open_cash_order_identity_fails_preflight(
    tmp_path: Path,
    field: str,
    remove: bool,
    invalid_value: object,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    malformed = _open_order()
    if remove:
        del malformed[field]
    else:
        malformed[field] = invalid_value
    client.orders = [malformed]

    with pytest.raises(BrokerSafetyError, match=field):
        _submit(executor, _intent())

    assert "send_cash_order" not in _call_names(client)
    assert ledger.replay() == []


@pytest.mark.parametrize(
    ("field", "remove", "invalid_value"),
    [
        ("Symbol", True, None),
        ("Symbol", False, "9433 "),
        ("Side", True, None),
        ("Side", False, "3"),
        ("AccountType", True, None),
        ("AccountType", False, 3),
    ],
)
def test_any_nonzero_cash_position_with_malformed_identity_fails_preflight(
    tmp_path: Path,
    field: str,
    remove: bool,
    invalid_value: object,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    malformed = _long_position(leaves=100.0)
    if remove:
        del malformed[field]
    else:
        malformed[field] = invalid_value
    client.positions = [malformed]

    with pytest.raises(BrokerSafetyError, match=field):
        _submit(executor, _intent())

    assert "send_cash_order" not in _call_names(client)
    assert ledger.replay() == []


@pytest.mark.parametrize(
    "operation",
    ["status", "submit", "cancel", "reconcile", "engage-kill-switch"],
)
def test_existing_operation_lock_blocks_before_network_or_audit_and_is_untouched(
    tmp_path: Path,
    operation: str,
) -> None:
    authentication_calls: list[bool] = []
    executor, client, ledger = _executor(
        tmp_path,
        authenticate=lambda: authentication_calls.append(True),
    )
    client.orders_by_id["ORDER-OPEN"] = [_matching_broker_order("ORDER-OPEN")]
    operation_lock = _operation_lock_path(ledger)
    with (
        advisory_file_lock(operation_lock),
        pytest.raises(BrokerSafetyError, match="another broker operation"),
    ):
        if operation == "status":
            executor.status()
        elif operation == "submit":
            _submit(executor, _intent())
        elif operation == "cancel":
            executor.cancel_order(
                "ORDER-OPEN",
                client_action_id="cancel-001",
                confirmation="not-reached",
            )
        elif operation == "reconcile":
            executor.reconcile()
        else:
            executor.engage_kill_switch("serialized risk stop")

    assert authentication_calls == []
    assert client.calls == []
    assert ledger.replay() == []
    assert operation_lock.exists()
    assert not executor.state_paths.kill_switch_path.exists()


@pytest.mark.parametrize("operation", ["status", "submit", "cancel", "reconcile"])
def test_authentication_callback_runs_once_inside_whole_operation_lock(
    tmp_path: Path,
    operation: str,
) -> None:
    operation_lock = (tmp_path / "broker-state" / "operation.lock").resolve()
    lock_observations: list[bool] = []

    def authenticate() -> None:
        lock_observations.append(operation_lock.is_file())

    executor, client, _ = _executor(tmp_path, authenticate=authenticate)
    client.orders_by_id["ORDER-OPEN"] = [_matching_broker_order("ORDER-OPEN")]

    if operation == "status":
        executor.status()
    elif operation == "submit":
        _submit(executor, _intent())
    elif operation == "cancel":
        confirmation = str(
            executor.preview_cancel(
                "ORDER-OPEN",
                client_action_id="cancel-001",
            )["confirmation_digest"]
        )
        lock_observations.clear()
        client.calls.clear()
        executor.cancel_order(
            "ORDER-OPEN",
            client_action_id="cancel-001",
            confirmation=confirmation,
        )
    else:
        executor.reconcile()

    assert lock_observations == [True]
    assert operation_lock.exists()


def test_kabus_reversed_board_names_use_bid_for_buy_and_ask_for_sell(tmp_path: Path) -> None:
    """kabuS documents BidPrice=Sell1 and AskPrice=Buy1, unlike common quote naming."""

    executor, client, _ = _executor(tmp_path, config_changes={"max_price_deviation_bps": 1.0})
    client.positions = [_long_position(leaves=100.0)]
    client.send_results = [OrderResult(0, "BUY-001"), OrderResult(0, "SELL-001")]

    buy = _intent(limit_price=1_000.0)
    sell = _intent(
        client_order_id="sell-limit-001",
        side="sell",
        limit_price=1_010.0,
    )

    assert _submit(executor, buy).notional_jpy == 100_000.0
    assert _submit(executor, sell).notional_jpy == 101_000.0


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("trading-unit", "multiple of the broker trading unit"),
        ("price-range", "outside the broker price range"),
        ("quote-deviation", "quote-deviation limit"),
        ("wallet", "cash buying power"),
        ("soft-limit", "cash soft limit"),
        ("open-orders", "maximum open cash-order count"),
        ("daily-notional", "daily accepted-order notional limit"),
        ("position", "maximum configured position quantity"),
        ("loss", "unrealized loss limit"),
    ],
)
def test_buy_preflight_enforces_account_and_market_constraints(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    config_changes: dict[str, object] = {}
    intent = _intent()
    if case == "open-orders":
        config_changes["max_open_orders"] = 1
    if case == "position":
        config_changes.update(max_order_quantity=200, max_position_quantity_per_symbol=200)
    executor, client, ledger = _executor(tmp_path, config_changes=config_changes)

    if case == "trading-unit":
        intent = _intent(quantity=150)
    elif case == "price-range":
        intent = _intent(limit_price=2_500.0)
    elif case == "quote-deviation":
        intent = _intent(limit_price=1_100.0)
    elif case == "wallet":
        client.symbol_wallet["AuKCStockAccountWallet"] = 99_999.0
    elif case == "soft-limit":
        client.soft_limit["Stock"] = 9.0
    elif case == "open-orders":
        client.orders = [_open_order()]
    elif case == "daily-notional":
        ledger.append(
            {
                "event_type": "broker_order_accepted",
                "event_ts": "2026-07-15T02:00:00Z",
                "environment": "validation",
                "client_order_id": "prior-order",
                "notional_jpy": 950_001.0,
                "order_date": "2026-07-15",
            }
        )
    elif case == "position":
        client.positions = [_long_position(leaves=200.0)]
    elif case == "loss":
        client.positions = [_long_position(profit_loss=-100_000.0)]

    with pytest.raises(BrokerSafetyError, match=message):
        _submit(executor, intent)

    assert "send_cash_order" not in _call_names(client)


@pytest.mark.parametrize(
    ("cash_buy_deliv_type", "expected_wallet_field", "should_accept"),
    [
        (2, "AuKCStockAccountWallet", False),
        (3, "StockAccountWallet", True),
    ],
)
def test_buy_preflight_uses_symbol_specific_wallet_for_configured_delivery_type(
    tmp_path: Path,
    cash_buy_deliv_type: int,
    expected_wallet_field: str,
    should_accept: bool,
) -> None:
    executor, client, _ = _executor(
        tmp_path,
        config_changes={"cash_buy_deliv_type": cash_buy_deliv_type},
    )
    client.wallet.update(
        StockAccountWallet=1_000_000.0,
        AuKCStockAccountWallet=1_000_000.0,
    )
    client.symbol_wallet.update(
        StockAccountWallet=1_000_000.0,
        AuKCStockAccountWallet=99_999.0,
    )

    if should_accept:
        submission = _submit(executor, _intent())
        assert submission.broker_order_id == "ORDER-001"
    else:
        with pytest.raises(BrokerSafetyError, match="cash buying power"):
            _submit(executor, _intent())
        assert "send_cash_order" not in _call_names(client)

    assert ("get_cash_wallet", ("9433", 27)) in client.calls
    selected = client.symbol_wallet[expected_wallet_field]
    assert selected == (1_000_000.0 if should_accept else 99_999.0)


@pytest.mark.parametrize(
    ("cash_buy_deliv_type", "wallet_field"),
    [(2, "AuKCStockAccountWallet"), (3, "StockAccountWallet")],
)
@pytest.mark.parametrize("missing", [False, True])
def test_applicable_symbol_wallet_field_must_be_present_and_numeric(
    tmp_path: Path,
    cash_buy_deliv_type: int,
    wallet_field: str,
    missing: bool,
) -> None:
    executor, client, _ = _executor(
        tmp_path,
        config_changes={"cash_buy_deliv_type": cash_buy_deliv_type},
    )
    if missing:
        del client.symbol_wallet[wallet_field]
    else:
        client.symbol_wallet[wallet_field] = None

    with pytest.raises(BrokerSafetyError, match=wallet_field):
        _submit(executor, _intent())

    assert "send_cash_order" not in _call_names(client)


@pytest.mark.parametrize(
    ("price_range_group", "limit_price", "valid"),
    [
        ("10000", 1_000.5, False),
        ("10003", 999.9, True),
        ("10003", 1_000.5, True),
        ("10003", 1_000.1, False),
        ("10004", 1_000.5, False),
    ],
)
def test_limit_price_must_align_to_official_price_range_group_tick(
    tmp_path: Path,
    price_range_group: str,
    limit_price: float,
    valid: bool,
) -> None:
    executor, client, _ = _executor(tmp_path)
    client.symbol["PriceRangeGroup"] = price_range_group
    client.board["BidPrice"] = limit_price

    if valid:
        assert _submit(executor, _intent(limit_price=limit_price)).broker_order_id == "ORDER-001"
    else:
        with pytest.raises(BrokerSafetyError, match="tick"):
            _submit(executor, _intent(limit_price=limit_price))
        assert "send_cash_order" not in _call_names(client)


@pytest.mark.parametrize("price_range_group", [None, "99999"])
def test_unknown_or_missing_price_range_group_fails_closed(
    tmp_path: Path,
    price_range_group: str | None,
) -> None:
    executor, client, _ = _executor(tmp_path)
    if price_range_group is None:
        del client.symbol["PriceRangeGroup"]
    else:
        client.symbol["PriceRangeGroup"] = price_range_group

    with pytest.raises(BrokerSafetyError, match="PriceRangeGroup"):
        _submit(executor, _intent())

    assert "send_cash_order" not in _call_names(client)


def test_broker_per_symbol_limit_is_enforced_in_addition_to_local_limit(
    tmp_path: Path,
) -> None:
    executor, client, _ = _executor(tmp_path)
    client.symbol["PerSymbolLimit"] = 99_999.0

    with pytest.raises(BrokerSafetyError, match="broker.*per-symbol|PerSymbolLimit"):
        _submit(executor, _intent())

    assert "send_cash_order" not in _call_names(client)


def test_null_broker_per_symbol_limit_allows_officially_exempt_symbol(tmp_path: Path) -> None:
    executor, client, _ = _executor(tmp_path)
    client.symbol["PerSymbolLimit"] = None

    assert _submit(executor, _intent()).broker_order_id == "ORDER-001"


def test_missing_broker_per_symbol_limit_fails_closed(tmp_path: Path) -> None:
    executor, client, _ = _executor(tmp_path)
    del client.symbol["PerSymbolLimit"]

    with pytest.raises(BrokerSafetyError, match="PerSymbolLimit"):
        _submit(executor, _intent())

    assert "send_cash_order" not in _call_names(client)


@pytest.mark.parametrize("cross_account_exposure", ["holding", "pending-buy"])
def test_buy_position_quantity_counts_cross_account_cash_exposure(
    tmp_path: Path,
    cross_account_exposure: str,
) -> None:
    executor, client, ledger = _executor(
        tmp_path,
        config_changes={
            "max_order_quantity": 100,
            "max_position_quantity_per_symbol": 250,
        },
    )
    if cross_account_exposure == "holding":
        client.positions = [_long_position(leaves=200.0, account_type=2)]
    else:
        client.orders = [_open_order(quantity=200.0, account_type=2)]

    with pytest.raises(BrokerSafetyError, match="position quantity"):
        _submit(executor, _intent())

    assert "send_cash_order" not in _call_names(client)
    assert ledger.replay() == []


def test_sell_availability_does_not_borrow_holdings_from_another_account_type(
    tmp_path: Path,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.positions = [_long_position(leaves=100.0, account_type=2)]

    with pytest.raises(BrokerSafetyError, match="available cash position"):
        _submit(executor, _intent(side="sell", limit_price=1_010.0))

    assert "send_cash_order" not in _call_names(client)
    assert ledger.replay() == []


def test_sell_requires_available_cash_holdings_and_does_not_require_buying_power(
    tmp_path: Path,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.symbol_wallet["AuKCStockAccountWallet"] = 0.0
    client.positions = [_long_position(leaves=200.0, held=100.0)]

    with pytest.raises(BrokerSafetyError, match="available cash position"):
        _submit(executor, _intent(side="sell", limit_price=1_010.0, quantity=200))
    assert "send_cash_order" not in _call_names(client)
    assert ledger.replay() == []

    sell = _intent(
        client_order_id="sell-001",
        side="sell",
        limit_price=1_010.0,
        quantity=100,
    )
    submission = _submit(executor, sell)
    assert submission.notional_jpy == 101_000.0
    assert submission.broker_order_id == "ORDER-001"


def test_accepted_client_order_id_is_idempotently_blocked_before_preflight(tmp_path: Path) -> None:
    executor, client, ledger = _executor(tmp_path)
    intent = _intent()
    _submit(executor, intent)
    calls_after_acceptance = list(client.calls)

    with pytest.raises(BrokerSafetyError, match="already exists"):
        _submit(executor, intent)

    assert client.calls == calls_after_acceptance
    assert [event["event_type"] for event in ledger.replay()].count("broker_order_accepted") == 1


def test_ambiguous_attempt_blocks_all_subsequent_orders_without_retry(tmp_path: Path) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.send_results = [AmbiguousMutationError("send order")]

    with pytest.raises(AmbiguousMutationError, match="ambiguous"):
        _submit(executor, _intent())
    calls_after_unknown = list(client.calls)

    with pytest.raises(BrokerSafetyError, match="unresolved broker mutations: client-001"):
        _submit(executor, _intent(client_order_id="client-002"))

    assert client.calls == calls_after_unknown
    assert ledger.unresolved_order_ids() == ("client-001",)
    assert [event["event_type"] for event in ledger.replay()] == [
        "broker_order_preflight",
        "broker_order_attempt",
        "broker_order_unknown",
    ]


def test_unexpected_client_rejection_is_conservatively_ambiguous(tmp_path: Path) -> None:
    executor, client, ledger = _executor(tmp_path)
    rejection = KabuSAPIRejection("send order", status_code=400, api_code=1001001)
    client.send_results = [rejection]

    with pytest.raises(AmbiguousMutationError, match="ambiguous"):
        _submit(executor, _intent())
    assert ledger.unresolved_order_ids() == ("client-001",)
    assert [event["event_type"] for event in ledger.replay()] == [
        "broker_order_preflight",
        "broker_order_attempt",
        "broker_order_unknown",
    ]


def test_kill_switch_blocks_submissions_but_permits_confirmed_cancellation(
    tmp_path: Path,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.orders_by_id["ORDER-OPEN"] = [_matching_broker_order("ORDER-OPEN")]
    executor.engage_kill_switch("operator risk stop")
    calls_before_submit = list(client.calls)

    with pytest.raises(BrokerSafetyError, match="kill switch is engaged"):
        _submit(executor, _intent())
    assert client.calls == calls_before_submit

    def assert_durable_cancel_attempt() -> None:
        assert ledger.replay()[-1]["event_type"] == "broker_cancel_attempt"

    client.before_cancel = assert_durable_cancel_attempt
    confirmation = _cancel_confirmation(executor, "ORDER-OPEN", "cancel-001")

    result = executor.cancel_order(
        "ORDER-OPEN",
        client_action_id="cancel-001",
        confirmation=confirmation,
    )
    assert result == OrderResult(0, "ORDER-OPEN")
    assert _call_names(client) == ["get_orders", "get_orders", "cancel_order"]
    assert ledger.replay()[-1]["event_type"] == "broker_cancel_accepted"


def test_cancel_requires_exact_confirmation_and_is_idempotently_audited(tmp_path: Path) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.orders_by_id["ORDER-OPEN"] = [_matching_broker_order("ORDER-OPEN")]

    with pytest.raises(BrokerSafetyError, match="explicit confirmation"):
        executor.cancel_order(
            "ORDER-OPEN",
            client_action_id="cancel-001",
            confirmation="CANCEL-WITHOUT-IDENTITY",
        )
    assert _call_names(client) == ["get_orders"]
    assert ledger.replay() == []

    confirmation = _cancel_confirmation(executor, "ORDER-OPEN", "cancel-001")
    result = executor.cancel_order(
        "ORDER-OPEN",
        client_action_id="cancel-001",
        confirmation=confirmation,
    )
    assert result.result_code == 0
    calls_after_cancel = list(client.calls)

    with pytest.raises(BrokerSafetyError, match="already exists"):
        executor.cancel_order(
            "ORDER-OPEN",
            client_action_id="cancel-001",
            confirmation=confirmation,
        )
    assert client.calls == calls_after_cancel
    assert [event["event_type"] for event in ledger.replay()] == [
        "broker_cancel_attempt",
        "broker_cancel_accepted",
    ]


def test_cancel_response_for_different_order_is_ambiguous_not_accepted(tmp_path: Path) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.orders_by_id["ORDER-OPEN"] = [_matching_broker_order("ORDER-OPEN")]
    client.cancel_results = [OrderResult(0, "DIFFERENT-ORDER")]
    confirmation = _cancel_confirmation(executor, "ORDER-OPEN", "cancel-001")

    with pytest.raises(AmbiguousMutationError, match="ambiguous"):
        executor.cancel_order(
            "ORDER-OPEN",
            client_action_id="cancel-001",
            confirmation=confirmation,
        )

    events = ledger.replay()
    assert [event["event_type"] for event in events] == [
        "broker_cancel_attempt",
        "broker_cancel_unknown",
    ]
    assert events[-1]["broker_order_id"] == "ORDER-OPEN"
    assert events[-1]["returned_broker_order_id"] == "DIFFERENT-ORDER"
    assert ledger.unresolved_order_ids() == ("cancel-001",)


def test_unknown_order_acceptance_requires_reconcile_and_exact_broker_match(
    tmp_path: Path,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.send_results = [AmbiguousMutationError("send order")]
    intent = _intent()
    with pytest.raises(AmbiguousMutationError):
        _submit(executor, intent)

    with pytest.raises(BrokerSafetyError, match="scoped broker reconciliation"):
        executor.resolve_unknown(
            intent.client_order_id,
            resolution="accepted",
            confirmation="not-yet-reconciled",
            broker_order_id="ORDER-UNKNOWN",
        )

    matching_order = _matching_broker_order()
    client.orders = [matching_order]
    client.orders_by_id["ORDER-UNKNOWN"] = [matching_order]
    reconciliation = executor.reconcile()
    assert reconciliation.account.unresolved_client_order_ids == ("client-001",)
    evidence = reconciliation.evidence[0]
    confirmation = (
        f"ACCEPTED:client-001:ORDER-UNKNOWN:{evidence['attempt_sequence']}:"
        f"{reconciliation.audit_sequence}:{reconciliation.config_digest}"
    )
    operation_lock = _operation_lock_path(ledger)
    lock_observations: list[bool] = []
    resolving_executor = KabuSExecutor(
        executor.config,
        client,  # type: ignore[arg-type]
        ledger,
        state_paths=executor.state_paths,
        clock=lambda: NOW,
        monotonic=lambda: 0.0,
        authenticate=lambda: lock_observations.append(operation_lock.is_file()),
    )
    event = resolving_executor.resolve_unknown(
        intent.client_order_id,
        resolution="accepted",
        confirmation=confirmation,
        broker_order_id="ORDER-UNKNOWN",
    )

    assert event["event_type"] == "broker_order_accepted"
    assert event["broker_order_id"] == "ORDER-UNKNOWN"
    assert event["resolution_method"] == "broker_reconciliation"
    assert lock_observations == [True]
    assert operation_lock.exists()
    assert ledger.unresolved_order_ids() == ()


def test_unknown_not_accepted_resolution_requires_reconcile_and_engaged_kill_switch(
    tmp_path: Path,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.send_results = [AmbiguousMutationError("send order")]
    intent = _intent()
    with pytest.raises(AmbiguousMutationError):
        _submit(executor, intent)
    early_reconciliation = executor.reconcile()
    early_evidence = early_reconciliation.evidence[0]
    early_confirmation = (
        f"NOT_ACCEPTED:client-001:{early_evidence['attempt_sequence']}:"
        f"{early_reconciliation.audit_sequence}:{early_reconciliation.config_digest}"
    )
    with pytest.raises(BrokerSafetyError, match="complete reconciliation window"):
        executor.resolve_unknown(
            intent.client_order_id,
            resolution="not-accepted",
            confirmation=early_confirmation,
        )
    resolving_executor = KabuSExecutor(
        executor.config,
        client,  # type: ignore[arg-type]
        ledger,
        state_paths=executor.state_paths,
        clock=lambda: NOW + timedelta(seconds=31),
        monotonic=lambda: 0.0,
    )
    reconciliation = resolving_executor.reconcile()
    evidence = reconciliation.evidence[0]
    confirmation = (
        f"NOT_ACCEPTED:client-001:{evidence['attempt_sequence']}:"
        f"{reconciliation.audit_sequence}:{reconciliation.config_digest}"
    )

    with pytest.raises(BrokerSafetyError, match="kill switch to be engaged"):
        resolving_executor.resolve_unknown(
            intent.client_order_id,
            resolution="not-accepted",
            confirmation=confirmation,
        )
    resolving_executor.engage_kill_switch("unknown order risk containment")
    event = resolving_executor.resolve_unknown(
        intent.client_order_id,
        resolution="not-accepted",
        confirmation=confirmation,
    )

    assert event["event_type"] == "broker_order_rejected"
    assert event["resolution_method"] == "operator_asserted_not_accepted"
    assert resolving_executor.state_paths.kill_switch_path.exists()
    assert ledger.unresolved_order_ids() == ()


def test_reconciliation_uses_order_read_time_not_later_slow_account_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = [NOW]
    executor, client, ledger = _executor(tmp_path, clock=lambda: current[0])
    client.send_results = [AmbiguousMutationError("send order")]
    with pytest.raises(AmbiguousMutationError):
        _submit(executor, _intent())

    original_get_positions = client.get_positions

    def slow_positions(*, product: int = 1) -> list[dict[str, Any]]:
        current[0] = NOW + timedelta(seconds=31)
        return original_get_positions(product=product)

    monkeypatch.setattr(client, "get_positions", slow_positions)
    reconciliation = executor.reconcile()
    evidence = reconciliation.evidence[0]
    event = ledger.replay()[-1]

    assert evidence["observed_at"] == "2026-07-15T03:00:00Z"
    assert evidence["match_window_end"] == "2026-07-15T03:00:30Z"
    assert evidence["negative_observation_complete"] is False
    assert event["observed_at"] == "2026-07-15T03:00:00Z"
    assert event["reconciliation_completed_at"] == "2026-07-15T03:00:31Z"


def test_late_unknown_event_extends_window_and_blocks_early_negative_resolution(
    tmp_path: Path,
) -> None:
    current = [NOW]
    executor, client, ledger = _executor(tmp_path, clock=lambda: current[0])
    client.send_results = [AmbiguousMutationError("send order")]
    client.before_send = lambda: current.__setitem__(0, NOW + timedelta(seconds=6))
    with pytest.raises(AmbiguousMutationError):
        _submit(executor, _intent())
    unknown = ledger.replay()[-1]
    assert unknown["event_type"] == "broker_order_unknown"
    assert unknown["event_ts"] == "2026-07-15T03:00:06Z"

    current[0] = NOW + timedelta(seconds=31)
    reconciliation = executor.reconcile()
    evidence = reconciliation.evidence[0]
    assert evidence["match_window_anchor"] == "2026-07-15T03:00:06Z"
    assert evidence["match_window_end"] == "2026-07-15T03:00:36Z"
    assert evidence["negative_observation_complete"] is False
    executor.engage_kill_switch("transport outcome is still inside its evidence window")
    confirmation = (
        f"NOT_ACCEPTED:client-001:{evidence['attempt_sequence']}:"
        f"{reconciliation.audit_sequence}:{reconciliation.config_digest}"
    )

    with pytest.raises(BrokerSafetyError, match="complete reconciliation window"):
        executor.resolve_unknown(
            "client-001",
            resolution="not-accepted",
            confirmation=confirmation,
        )

    assert ledger.unresolved_order_ids() == ("client-001",)


def test_late_unknown_event_keeps_later_broker_receive_time_in_match_window(
    tmp_path: Path,
) -> None:
    current = [NOW]
    executor, client, _ = _executor(tmp_path, clock=lambda: current[0])
    client.send_results = [AmbiguousMutationError("send order")]
    client.before_send = lambda: current.__setitem__(0, NOW + timedelta(seconds=6))
    with pytest.raises(AmbiguousMutationError):
        _submit(executor, _intent())
    matched = _matching_broker_order("ORDER-LATE")
    matched["RecvTime"] = (NOW + timedelta(seconds=33)).isoformat()
    client.orders = [matched]
    current[0] = NOW + timedelta(seconds=34)

    reconciliation = executor.reconcile()
    evidence = reconciliation.evidence[0]

    assert evidence["match_window_anchor"] == "2026-07-15T03:00:06Z"
    assert evidence["match_window_end"] == "2026-07-15T03:00:36Z"
    assert evidence["candidate_order_ids"] == ["ORDER-LATE"]
    assert evidence["exact_match_order_ids"] == ["ORDER-LATE"]


def test_unknown_cancel_not_accepted_requires_unchanged_processed_order(
    tmp_path: Path,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    unchanged = _record_unknown_cancel(executor, client)
    client.orders = [unchanged]

    resolving_executor = KabuSExecutor(
        executor.config,
        client,  # type: ignore[arg-type]
        ledger,
        state_paths=executor.state_paths,
        clock=lambda: NOW + timedelta(seconds=31),
        monotonic=lambda: 0.0,
    )
    reconciliation = resolving_executor.reconcile()
    evidence = reconciliation.evidence[0]
    assert evidence["candidate_order_ids"] == []
    assert evidence["exact_match_order_ids"] == []
    assert evidence["negative_observation_complete"] is True
    marker = resolving_executor.engage_kill_switch("cancel was not observed")
    confirmation = (
        f"NOT_ACCEPTED:cancel-001:{evidence['attempt_sequence']}:"
        f"{reconciliation.audit_sequence}:{reconciliation.config_digest}"
    )

    event = resolving_executor.resolve_unknown(
        "cancel-001",
        resolution="not-accepted",
        confirmation=confirmation,
    )

    assert event["event_type"] == "broker_cancel_rejected"
    assert event["fresh_verification"]["candidate_order_ids"] == []
    assert event["fresh_verification"]["negative_observation_complete"] is True
    assert resolving_executor.state_paths.kill_switch_path.exists()
    assert marker["marker_digest"]
    assert ledger.unresolved_order_ids() == ()


@pytest.mark.parametrize(
    ("state", "details"),
    [
        (1, []),
        (2, []),
        (4, []),
        (5, []),
        (
            3,
            [
                {
                    "RecType": 6,
                    "State": 1,
                    "TransactTime": (NOW + timedelta(seconds=1)).isoformat(),
                }
            ],
        ),
    ],
)
def test_unknown_cancel_not_accepted_rejects_changed_or_cancel_active_order(
    tmp_path: Path,
    state: int,
    details: list[dict[str, Any]],
) -> None:
    executor, client, ledger = _executor(tmp_path)
    changed = _record_unknown_cancel(executor, client)
    changed["State"] = state
    changed["OrderState"] = state
    changed["Details"] = details
    client.orders = [changed]

    resolving_executor = KabuSExecutor(
        executor.config,
        client,  # type: ignore[arg-type]
        ledger,
        state_paths=executor.state_paths,
        clock=lambda: NOW + timedelta(seconds=31),
        monotonic=lambda: 0.0,
    )
    reconciliation = resolving_executor.reconcile()
    evidence = reconciliation.evidence[0]
    resolving_executor.engage_kill_switch("changed cancel evidence")
    confirmation = (
        f"NOT_ACCEPTED:cancel-001:{evidence['attempt_sequence']}:"
        f"{reconciliation.audit_sequence}:{reconciliation.config_digest}"
    )

    with pytest.raises(BrokerSafetyError, match="no reconciled candidate"):
        resolving_executor.resolve_unknown(
            "cancel-001",
            resolution="not-accepted",
            confirmation=confirmation,
        )

    assert evidence["candidate_order_ids"] == ["ORDER-OPEN"]
    assert ledger.unresolved_order_ids() == ("cancel-001",)


def test_audit_records_only_sanitized_evidence_not_broker_secrets(tmp_path: Path) -> None:
    executor, client, ledger = _executor(tmp_path)
    secret = "do-not-retain-this-broker-secret"
    client.orders = [{"State": 5, "OrderState": 5, "Symbol": "OTHER", "Token": secret}]
    client.positions = [
        {
            "Symbol": "OTHER",
            "AccountType": 4,
            "Side": "2",
            "LeavesQty": 0.0,
            "HoldQty": 0.0,
            "CurrentPrice": 1_000.0,
            "ProfitLoss": 0.0,
            "APIPassword": secret,
        }
    ]
    client.wallet["Authorization"] = secret
    client.soft_limit["Secret"] = secret
    client.symbol["ApiKey"] = secret
    client.board["X-API-KEY"] = secret

    _submit(executor, _intent())

    raw_audit = ledger.path.read_text(encoding="utf-8").casefold()
    assert secret not in raw_audit
    assert "password" not in raw_audit
    assert "authorization" not in raw_audit
    assert "api-key" not in raw_audit
    assert '"token"' not in raw_audit


def test_order_approval_binds_environment_config_and_effective_payload(tmp_path: Path) -> None:
    intent = _intent()
    validation, _, _ = _executor(tmp_path / "validation")
    different_account, _, _ = _executor(
        tmp_path / "account",
        config_changes={"account_type": 2},
    )
    production, _, _ = _executor(
        tmp_path / "production",
        config_changes={
            "environment": "production",
            "production_acknowledgement": "REAL_ORDERS",
        },
    )

    validation_preview = validation.preview_order(intent)
    account_preview = different_account.preview_order(intent)
    production_preview = production.preview_order(intent)

    assert account_preview["effective_request_payload"]["AccountType"] == 2
    assert production_preview["production_places_real_orders"] is True
    assert (
        len(
            {
                validation_preview["confirmation_digest"],
                account_preview["confirmation_digest"],
                production_preview["confirmation_digest"],
            }
        )
        == 3
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("account-loss", "unrealized loss limit"),
        ("symbol-notional", "per-symbol position notional"),
        ("gross-notional", "gross cash-position notional"),
    ],
)
def test_account_wide_loss_and_position_notional_limits_fail_closed(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    changes: dict[str, object] = {}
    if case == "symbol-notional":
        changes.update(
            max_order_notional_jpy=100_000.0,
            max_position_notional_jpy_per_symbol=150_000.0,
        )
    elif case == "gross-notional":
        changes.update(
            max_order_notional_jpy=100_000.0,
            max_position_notional_jpy_per_symbol=500_000.0,
            max_gross_cash_position_notional_jpy=500_000.0,
        )
    executor, client, _ = _executor(tmp_path, config_changes=changes)

    if case == "account-loss":
        client.account_positions = [{"ProfitLoss": -100_000.0}]
    elif case == "symbol-notional":
        client.positions = [_long_position(leaves=100.0)]
    else:
        other = _long_position(leaves=450.0)
        other["Symbol"] = "OTHER"
        client.positions = [other]

    with pytest.raises(BrokerSafetyError, match=message):
        _submit(executor, _intent())

    assert "send_cash_order" not in _call_names(client)


@pytest.mark.parametrize(
    ("quote_delta", "message"),
    [
        (timedelta(seconds=-31), "older than"),
        (timedelta(seconds=6), "in the future"),
    ],
)
def test_quote_timestamp_must_be_recent_and_not_future(
    tmp_path: Path,
    quote_delta: timedelta,
    message: str,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.board["BidTime"] = (NOW + quote_delta).isoformat()

    with pytest.raises(BrokerSafetyError, match=message):
        _submit(executor, _intent())

    assert "send_cash_order" not in _call_names(client)
    assert ledger.replay() == []


def test_quote_freshness_is_rechecked_after_durable_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = [NOW]
    executor, client, ledger = _executor(
        tmp_path,
        config_changes={"max_quote_age_seconds": 2},
        clock=lambda: current[0],
    )
    original_append = ledger.append

    def advance_after_attempt(event: dict[str, Any]) -> dict[str, Any]:
        recorded = original_append(event)
        if event.get("event_type") == "broker_order_attempt":
            current[0] = NOW + timedelta(seconds=3)
        return recorded

    monkeypatch.setattr(ledger, "append", advance_after_attempt)

    with pytest.raises(BrokerSafetyError, match="quote is older"):
        _submit(executor, _intent())

    assert "send_cash_order" not in _call_names(client)
    assert [event["event_type"] for event in ledger.replay()] == [
        "broker_order_preflight",
        "broker_order_attempt",
        "broker_order_rejected",
    ]


def test_preflight_deadline_is_enforced_before_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    elapsed = [0.0]
    executor, client, ledger = _executor(tmp_path, monotonic=lambda: elapsed[0])
    original_get_board = client.get_board

    def slow_board(symbol: str, reference_exchange: int) -> dict[str, Any]:
        response = original_get_board(symbol, reference_exchange)
        elapsed[0] = 16.0
        return response

    monkeypatch.setattr(client, "get_board", slow_board)

    with pytest.raises(BrokerSafetyError, match="preflight exceeded"):
        _submit(executor, _intent())

    assert "send_cash_order" not in _call_names(client)
    assert ledger.replay() == []


def test_japan_order_date_is_rechecked_after_durable_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_midnight = datetime(2026, 7, 15, 14, 59, 59, tzinfo=UTC)
    current = [before_midnight]
    executor, client, ledger = _executor(tmp_path, clock=lambda: current[0])
    client.board["BidTime"] = (before_midnight - timedelta(seconds=1)).isoformat()
    original_append = ledger.append

    def cross_midnight_after_attempt(event: dict[str, Any]) -> dict[str, Any]:
        recorded = original_append(event)
        if event.get("event_type") == "broker_order_attempt":
            current[0] = before_midnight + timedelta(seconds=2)
        return recorded

    monkeypatch.setattr(ledger, "append", cross_midnight_after_attempt)
    intent = _intent(created_at=before_midnight - timedelta(seconds=1))

    with pytest.raises(BrokerSafetyError, match="Japan order date changed"):
        _submit(executor, intent)

    assert "send_cash_order" not in _call_names(client)
    assert ledger.replay()[-1]["event_type"] == "broker_order_rejected"


def test_cancel_remains_available_when_new_order_submission_is_disabled(
    tmp_path: Path,
) -> None:
    executor, client, ledger = _executor(
        tmp_path,
        config_changes={"order_submission_enabled": False},
    )
    client.orders_by_id["ORDER-OPEN"] = [_matching_broker_order("ORDER-OPEN")]
    confirmation = _cancel_confirmation(executor, "ORDER-OPEN", "cancel-001")

    result = executor.cancel_order(
        "ORDER-OPEN",
        client_action_id="cancel-001",
        confirmation=confirmation,
    )

    assert result.order_id == "ORDER-OPEN"
    assert ledger.replay()[-1]["event_type"] == "broker_cancel_accepted"


def test_cancel_can_inspect_maintenance_only_direct_tse_order(tmp_path: Path) -> None:
    executor, client, _ = _executor(tmp_path)
    order = _matching_broker_order("ORDER-TSE")
    order["Exchange"] = 1
    client.orders_by_id["ORDER-TSE"] = [order]

    preview = executor.preview_cancel("ORDER-TSE", client_action_id="cancel-tse-001")

    assert preview["broker_order"]["Exchange"] == 1
    assert preview["broker_order"]["OrdType"] == 1


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (1, "processed state|State=3|cancellable"),
        (2, "processed state|State=3|cancellable"),
        (4, "cancellation in flight"),
        (5, "terminal"),
    ],
)
def test_cancel_preflight_accepts_only_processed_state_three(
    tmp_path: Path,
    state: int,
    message: str,
) -> None:
    executor, client, _ = _executor(tmp_path)
    client.orders_by_id["ORDER-OPEN"] = [_matching_broker_order("ORDER-OPEN", state=state)]

    with pytest.raises(BrokerSafetyError, match=message):
        executor.preview_cancel("ORDER-OPEN", client_action_id="cancel-001")

    assert "cancel_order" not in _call_names(client)


def test_cancel_preflight_accepts_state_three_and_binds_order_type(tmp_path: Path) -> None:
    executor, client, _ = _executor(tmp_path)
    client.orders_by_id["ORDER-OPEN"] = [_matching_broker_order("ORDER-OPEN", state=3)]

    preview = executor.preview_cancel("ORDER-OPEN", client_action_id="cancel-001")

    assert preview["broker_order"]["State"] == 3
    assert preview["broker_order"]["OrdType"] == 1


def test_cancel_confirmation_is_invalidated_by_broker_context_change(tmp_path: Path) -> None:
    executor, client, ledger = _executor(tmp_path)
    original = _matching_broker_order("ORDER-OPEN")
    client.orders_by_id["ORDER-OPEN"] = [original]
    confirmation = _cancel_confirmation(executor, "ORDER-OPEN", "cancel-001")
    changed = dict(original)
    changed["Price"] = 999.0
    client.orders_by_id["ORDER-OPEN"] = [changed]

    with pytest.raises(BrokerSafetyError, match="explicit confirmation"):
        executor.cancel_order(
            "ORDER-OPEN",
            client_action_id="cancel-001",
            confirmation=confirmation,
        )

    assert "cancel_order" not in _call_names(client)
    assert ledger.replay() == []


def test_cancel_confirmation_is_invalidated_by_order_type_change(tmp_path: Path) -> None:
    executor, client, ledger = _executor(tmp_path)
    original = _matching_broker_order("ORDER-OPEN")
    client.orders_by_id["ORDER-OPEN"] = [original]
    confirmation = _cancel_confirmation(executor, "ORDER-OPEN", "cancel-001")
    changed = dict(original)
    changed["OrdType"] = 2
    client.orders_by_id["ORDER-OPEN"] = [changed]

    with pytest.raises(BrokerSafetyError, match="explicit confirmation"):
        executor.cancel_order(
            "ORDER-OPEN",
            client_action_id="cancel-001",
            confirmation=confirmation,
        )

    assert "cancel_order" not in _call_names(client)
    assert ledger.replay() == []


def test_repeat_cancel_is_blocked_while_same_broker_order_is_unresolved(tmp_path: Path) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.orders_by_id["ORDER-OPEN"] = [_matching_broker_order("ORDER-OPEN")]
    client.cancel_results = [AmbiguousMutationError("cancel order")]
    confirmation = _cancel_confirmation(executor, "ORDER-OPEN", "cancel-001")
    with pytest.raises(AmbiguousMutationError):
        executor.cancel_order(
            "ORDER-OPEN",
            client_action_id="cancel-001",
            confirmation=confirmation,
        )
    calls_after_unknown = list(client.calls)

    with pytest.raises(BrokerSafetyError, match="unresolved cancellation attempt"):
        executor.cancel_order(
            "ORDER-OPEN",
            client_action_id="cancel-002",
            confirmation="not-reached",
        )

    assert client.calls == calls_after_unknown
    assert ledger.unresolved_order_ids() == ("cancel-001",)


def test_reconciliation_tolerates_partial_read_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, client, ledger = _executor(tmp_path)

    def unavailable_wallet() -> dict[str, Any]:
        raise KabuSTransportError("sanitized read failure")

    monkeypatch.setattr(client, "get_cash_wallet", unavailable_wallet)

    reconciliation = executor.reconcile()

    assert reconciliation.account.cash_wallet_jpy is None
    assert reconciliation.account.unavailable_fields == ("cash_wallet",)
    assert ledger.replay()[-1]["unavailable_fields"] == ["cash_wallet"]


def test_reconciliation_loss_limit_is_unknown_when_profit_loss_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    original_get_positions = client.get_positions

    def positions_without_account_profit_loss(*, product: int = 1) -> list[dict[str, Any]]:
        if product == 0:
            raise KabuSTransportError("sanitized account-position failure")
        return original_get_positions(product=product)

    monkeypatch.setattr(client, "get_positions", positions_without_account_profit_loss)

    reconciliation = executor.reconcile()
    event = ledger.replay()[-1]

    assert reconciliation.account.total_unrealized_profit_loss_jpy is None
    assert "account_positions_or_profit_loss" in reconciliation.account.unavailable_fields
    assert event["loss_limit_breached"] is None


@pytest.mark.parametrize(
    ("profit_loss", "expected_breached"),
    [(0.0, False), (-99_999.0, False), (-100_000.0, True)],
)
def test_reconciliation_loss_limit_is_boolean_when_profit_loss_is_observed(
    tmp_path: Path,
    profit_loss: float,
    expected_breached: bool,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.account_positions = [{"ProfitLoss": profit_loss}]

    reconciliation = executor.reconcile()
    event = ledger.replay()[-1]

    assert reconciliation.account.total_unrealized_profit_loss_jpy == profit_loss
    assert event["loss_limit_breached"] is expected_breached
    assert type(event["loss_limit_breached"]) is bool


def test_accepted_unknown_resolution_requires_one_unique_exact_match(tmp_path: Path) -> None:
    executor, client, _ = _executor(tmp_path)
    client.send_results = [AmbiguousMutationError("send order")]
    with pytest.raises(AmbiguousMutationError):
        _submit(executor, _intent())
    first = _matching_broker_order("ORDER-A")
    second = _matching_broker_order("ORDER-B")
    client.orders = [first, second]
    reconciliation = executor.reconcile()
    evidence = reconciliation.evidence[0]
    confirmation = (
        f"ACCEPTED:client-001:ORDER-A:{evidence['attempt_sequence']}:"
        f"{reconciliation.audit_sequence}:{reconciliation.config_digest}"
    )

    with pytest.raises(BrokerSafetyError, match="one unique exact reconciliation match"):
        executor.resolve_unknown(
            "client-001",
            resolution="accepted",
            confirmation=confirmation,
            broker_order_id="ORDER-A",
        )


def test_accepted_resolution_rejects_second_candidate_in_fresh_all_orders_read(
    tmp_path: Path,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.send_results = [AmbiguousMutationError("send order")]
    with pytest.raises(AmbiguousMutationError):
        _submit(executor, _intent())
    first = _matching_broker_order("ORDER-A")
    client.orders = [first]
    reconciliation = executor.reconcile()
    evidence = reconciliation.evidence[0]
    confirmation = (
        f"ACCEPTED:client-001:ORDER-A:{evidence['attempt_sequence']}:"
        f"{reconciliation.audit_sequence}:{reconciliation.config_digest}"
    )
    client.orders = [first, _matching_broker_order("ORDER-B")]

    with pytest.raises(BrokerSafetyError, match="fresh all-order read"):
        executor.resolve_unknown(
            "client-001",
            resolution="accepted",
            confirmation=confirmation,
            broker_order_id="ORDER-A",
        )

    assert client.calls[-1] == ("get_orders", (None, 1))
    assert ledger.unresolved_order_ids() == ("client-001",)


def test_resolution_rejects_stale_prior_reconciliation_before_fresh_read(tmp_path: Path) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.send_results = [AmbiguousMutationError("send order")]
    with pytest.raises(AmbiguousMutationError):
        _submit(executor, _intent())
    client.orders = [_matching_broker_order("ORDER-UNKNOWN")]
    reconciliation = executor.reconcile()
    evidence = reconciliation.evidence[0]
    confirmation = (
        f"ACCEPTED:client-001:ORDER-UNKNOWN:{evidence['attempt_sequence']}:"
        f"{reconciliation.audit_sequence}:{reconciliation.config_digest}"
    )
    calls_before_resolution = list(client.calls)
    stale_executor = KabuSExecutor(
        executor.config,
        client,  # type: ignore[arg-type]
        ledger,
        state_paths=executor.state_paths,
        clock=lambda: NOW + timedelta(seconds=31),
        monotonic=lambda: 0.0,
    )

    with pytest.raises(BrokerSafetyError, match="evidence is stale"):
        stale_executor.resolve_unknown(
            "client-001",
            resolution="accepted",
            confirmation=confirmation,
            broker_order_id="ORDER-UNKNOWN",
        )

    assert client.calls == calls_before_resolution
    assert ledger.unresolved_order_ids() == ("client-001",)


@pytest.mark.parametrize("malformed", ["invalid-id", "invalid-time", "duplicate-id"])
def test_reconciliation_marks_malformed_or_duplicate_all_order_evidence_invalid(
    tmp_path: Path,
    malformed: str,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.send_results = [AmbiguousMutationError("send order")]
    with pytest.raises(AmbiguousMutationError):
        _submit(executor, _intent())
    order = _matching_broker_order("ORDER-UNKNOWN")
    if malformed == "invalid-id":
        order["ID"] = "INVALID ID"
        client.orders = [order]
    elif malformed == "invalid-time":
        order["RecvTime"] = "not-a-timestamp"
        client.orders = [order]
    else:
        client.orders = [order, dict(order)]

    reconciliation = executor.reconcile()
    evidence = reconciliation.evidence[0]

    assert evidence["orders_available"] is True
    assert evidence["orders_valid"] is False
    assert evidence["evidence_valid"] is False
    assert evidence["negative_observation_complete"] is False
    assert ledger.unresolved_order_ids() == ("client-001",)


@pytest.mark.parametrize("malformed", ["invalid-time", "duplicate-id"])
def test_resolution_rejects_malformed_or_duplicate_fresh_all_order_evidence(
    tmp_path: Path,
    malformed: str,
) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.send_results = [AmbiguousMutationError("send order")]
    with pytest.raises(AmbiguousMutationError):
        _submit(executor, _intent())
    order = _matching_broker_order("ORDER-UNKNOWN")
    client.orders = [order]
    reconciliation = executor.reconcile()
    evidence = reconciliation.evidence[0]
    confirmation = (
        f"ACCEPTED:client-001:ORDER-UNKNOWN:{evidence['attempt_sequence']}:"
        f"{reconciliation.audit_sequence}:{reconciliation.config_digest}"
    )
    if malformed == "invalid-time":
        malformed_order = dict(order)
        malformed_order["RecvTime"] = "not-a-timestamp"
        client.orders = [malformed_order]
    else:
        client.orders = [order, dict(order)]

    with pytest.raises(BrokerSafetyError, match="fresh broker order evidence is invalid"):
        executor.resolve_unknown(
            "client-001",
            resolution="accepted",
            confirmation=confirmation,
            broker_order_id="ORDER-UNKNOWN",
        )

    assert client.calls[-1] == ("get_orders", (None, 1))
    assert ledger.unresolved_order_ids() == ("client-001",)


def test_submit_reconciliation_requires_normal_limit_order_type(tmp_path: Path) -> None:
    executor, client, _ = _executor(tmp_path)
    client.send_results = [AmbiguousMutationError("send order")]
    with pytest.raises(AmbiguousMutationError):
        _submit(executor, _intent())

    mismatched = _matching_broker_order("ORDER-UNKNOWN")
    mismatched["OrdType"] = 2
    client.orders = [mismatched]
    mismatched_reconciliation = executor.reconcile()

    assert mismatched_reconciliation.evidence[0]["candidate_order_ids"] == ["ORDER-UNKNOWN"]
    assert mismatched_reconciliation.evidence[0]["exact_match_order_ids"] == []

    exact = dict(mismatched)
    exact["OrdType"] = 1
    client.orders = [exact]
    exact_reconciliation = executor.reconcile()

    assert exact_reconciliation.evidence[0]["exact_match_order_ids"] == ["ORDER-UNKNOWN"]


def test_unknown_resolution_rejects_a_different_effective_config(tmp_path: Path) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.send_results = [AmbiguousMutationError("send order")]
    with pytest.raises(AmbiguousMutationError):
        _submit(executor, _intent())
    client.orders = [_matching_broker_order("ORDER-UNKNOWN")]
    executor.reconcile()
    changed_config = _config(tmp_path, max_order_quantity=900)
    changed_executor = KabuSExecutor(
        changed_config,
        client,  # type: ignore[arg-type]
        ledger,
        state_paths=executor.state_paths,
        clock=lambda: NOW,
        monotonic=lambda: 0.0,
    )

    with pytest.raises(BrokerSafetyError, match="original broker config"):
        changed_executor.resolve_unknown(
            "client-001",
            resolution="accepted",
            confirmation="not-reached",
            broker_order_id="ORDER-UNKNOWN",
        )


def test_kill_switch_release_confirmation_cannot_be_reused(tmp_path: Path) -> None:
    executor, _, _ = _executor(tmp_path)
    first = executor.engage_kill_switch("same reason")
    first_confirmation = f"RELEASE_KILL_SWITCH:{first['marker_digest']}"
    executor.release_kill_switch(confirmation=first_confirmation)
    second = executor.engage_kill_switch("same reason")

    assert second["marker_digest"] != first["marker_digest"]
    with pytest.raises(BrokerSafetyError, match="explicit confirmation"):
        executor.release_kill_switch(confirmation=first_confirmation)
    executor.release_kill_switch(confirmation=f"RELEASE_KILL_SWITCH:{second['marker_digest']}")


def test_unicode_kill_switch_reason_digest_and_audit_round_trip(tmp_path: Path) -> None:
    executor, _, ledger = _executor(tmp_path)
    reason = "日本語の緊急停止理由：価格急変を確認"

    engaged = executor.engage_kill_switch(reason)
    marker_digest = str(engaged["marker_digest"])
    snapshot = executor.status()

    assert snapshot.kill_switch_engaged is True
    assert snapshot.kill_switch_digest == marker_digest

    released = executor.release_kill_switch(
        confirmation=f"RELEASE_KILL_SWITCH:{marker_digest}",
    )
    events = ledger.replay()

    assert released["event_type"] == "broker_kill_switch_release_authorized"
    assert released["marker_digest"] == marker_digest
    assert [event["reason"] for event in events] == [reason, reason]
    assert executor.state_paths.kill_switch_path.exists() is False


def test_kill_switch_cannot_be_released_with_an_unresolved_mutation(tmp_path: Path) -> None:
    executor, client, ledger = _executor(tmp_path)
    client.send_results = [AmbiguousMutationError("send order")]
    with pytest.raises(AmbiguousMutationError):
        _submit(executor, _intent())
    marker = executor.engage_kill_switch("contain unknown order")

    with pytest.raises(BrokerSafetyError, match="mutations are unresolved"):
        executor.release_kill_switch(confirmation=f"RELEASE_KILL_SWITCH:{marker['marker_digest']}")

    assert executor.state_paths.kill_switch_path.exists()
    assert ledger.unresolved_order_ids() == ("client-001",)


def test_broken_kill_switch_symlink_still_blocks_new_orders(tmp_path: Path) -> None:
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("platform does not expose no-follow file opens")
    executor, client, ledger = _executor(tmp_path)
    executor.state_paths.root.mkdir(parents=True, exist_ok=True)
    executor.state_paths.kill_switch_path.symlink_to(tmp_path / "missing-marker-target")

    with pytest.raises(BrokerSafetyError, match="kill switch is engaged"):
        _submit(executor, _intent())

    snapshot = executor.status()
    assert snapshot.kill_switch_engaged is True
    assert snapshot.kill_switch_digest is None
    assert "kill_switch_marker" in snapshot.unavailable_fields
    assert "send_cash_order" not in _call_names(client)
    assert ledger.replay() == []
