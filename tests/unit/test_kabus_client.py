from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from http.client import BadStatusLine, HTTPConnection, IncompleteRead
from typing import Any
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request

import pytest
from pydantic import ValidationError

import nlp_trader.broker.kabus as kabus_module
from nlp_trader.broker.contracts import (
    MAX_CASH_ORDER_INTENT_BYTES,
    SCHEMA_VERSION,
    CashOrderIntent,
    CashOrderIntentJsonError,
)
from nlp_trader.broker.kabus import (
    AmbiguousMutationError,
    KabuSAPIRejection,
    KabuSAuthenticationRequired,
    KabuSProtocolError,
    KabuSTransportError,
    TransportFailure,
    TransportResponse,
    UrllibTransport,
    _KabuSClient,
)


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None
    timeout_seconds: float
    max_response_bytes: int


class FakeTransport:
    def __init__(self, *results: TransportResponse | Exception) -> None:
        self.results = list(results)
        self.requests: list[RecordedRequest] = []

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> TransportResponse:
        self.requests.append(
            RecordedRequest(
                method=method,
                url=url,
                headers=dict(headers),
                body=body,
                timeout_seconds=timeout_seconds,
                max_response_bytes=max_response_bytes,
            )
        )
        if not self.results:
            raise AssertionError("fake transport received an unexpected request")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _response(payload: object, *, status_code: int = 200) -> TransportResponse:
    return TransportResponse(
        status_code=status_code,
        body=json.dumps(payload, separators=(",", ":")).encode(),
    )


def _raw_response(payload: bytes, *, status_code: int = 200) -> TransportResponse:
    return TransportResponse(status_code=status_code, body=payload)


def _token_response(token: str = "token-value") -> TransportResponse:
    return _response({"ResultCode": 0, "Token": token})


def _intent_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "client_order_id": "client-001",
        "strategy_id": "strategy-001",
        "created_at": "2026-07-15T12:34:56.123456+09:00",
        "symbol": "9433",
        "exchange": 27,
        "reference_exchange": 1,
        "side": "buy",
        "quantity": 100,
        "order_type": "limit",
        "limit_price": 123.5,
        "expire_day": 20260716,
    }
    payload.update(changes)
    return payload


def _intent(**changes: object) -> CashOrderIntent:
    return CashOrderIntent.from_json(json.dumps(_intent_payload(**changes)))


def test_cash_order_intent_is_strict_frozen_canonical_and_digestible() -> None:
    intent = _intent()

    assert intent.created_at == datetime(2026, 7, 15, 3, 34, 56, 123456, tzinfo=UTC)
    canonical = intent.canonical_json()
    assert canonical == json.dumps(
        {
            **_intent_payload(),
            "created_at": "2026-07-15T03:34:56.123456Z",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert intent.confirmation_digest() == hashlib.sha256(canonical.encode()).hexdigest()
    assert _intent(created_at="2026-07-15T03:34:56.123456Z").confirmation_digest() == (
        intent.confirmation_digest()
    )

    with pytest.raises(ValidationError, match="frozen"):
        intent.quantity = 200  # type: ignore[misc]


def test_cash_order_intent_rejects_duplicate_and_oversized_json() -> None:
    duplicate = json.dumps(_intent_payload()).replace(
        '"schema_version": "kabus-cash-order-v1",',
        '"schema_version": "kabus-cash-order-v1", "schema_version": "kabus-cash-order-v1",',
        1,
    )

    with pytest.raises(CashOrderIntentJsonError, match="repeats"):
        CashOrderIntent.from_json(duplicate)
    with pytest.raises(CashOrderIntentJsonError, match="byte limit"):
        CashOrderIntent.from_json(" " * (MAX_CASH_ORDER_INTENT_BYTES + 1))


def test_cash_order_intent_requires_an_explicit_expiry_date() -> None:
    payload = _intent_payload()
    del payload["expire_day"]

    with pytest.raises(ValidationError, match="Field required") as raised:
        CashOrderIntent.from_json(json.dumps(payload))
    assert raised.value.errors()[0]["loc"] == ("expire_day",)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"created_at": "2026-07-15T03:34:56"}, "include a timezone"),
        ({"quantity": True}, "must be an integer"),
        ({"quantity": "100"}, "must be an integer"),
        ({"exchange": True, "reference_exchange": True}, "exchange must be an integer"),
        ({"expire_day": 20260230}, "valid explicit YYYYMMDD"),
        ({"expire_day": 0}, "valid explicit YYYYMMDD"),
        ({"order_type": "limit", "limit_price": None}, "valid number"),
        ({"order_type": "market", "limit_price": 100.0}, "Input should be 'limit'"),
        ({"exchange": 1, "reference_exchange": 1}, "exchange"),
        ({"exchange": 27, "reference_exchange": 3}, "reference_exchange"),
        ({"exchange": 3, "reference_exchange": 1}, "reference_exchange"),
        ({"unexpected": True}, "Extra inputs are not permitted"),
    ],
)
def test_cash_order_intent_rejects_invalid_semantics(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _intent(**changes)


def test_authentication_uses_exact_validation_endpoint_and_retains_only_token() -> None:
    fake = FakeTransport(_token_response("memory-token"), _response({"StockAccountWallet": None}))
    client = _KabuSClient("validation", transport=fake)

    client.authenticate("password-that-must-not-be-retained")
    wallet = client.get_cash_wallet()

    assert wallet == {"StockAccountWallet": None}
    token_request, wallet_request = fake.requests
    assert token_request.method == "POST"
    assert token_request.url == "http://127.0.0.1:18081/kabusapi/token"
    assert token_request.headers == {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    assert json.loads(token_request.body or b"") == {
        "APIPassword": "password-that-must-not-be-retained"
    }
    assert wallet_request.headers["X-API-KEY"] == "memory-token"
    assert "memory-token" not in wallet_request.url
    assert wallet_request.body is None
    assert "memory-token" not in repr(client)
    assert "password-that-must-not-be-retained" not in repr(client)
    assert not hasattr(client, "_api_password")

    client.clear_token()
    assert client.authenticated is False
    with pytest.raises(KabuSAuthenticationRequired):
        client.get_positions()


@pytest.mark.parametrize(
    "malicious_token",
    [
        "token\r\nX-Injected: value",
        "token\nX-Injected: value",
        "token\x00suffix",
        "token\x1fsuffix",
        "token\x7fsuffix",
        "tökén",
        "x" * 4097,
    ],
)
def test_authentication_rejects_unsafe_token_before_it_can_become_a_header(
    malicious_token: str,
) -> None:
    fake = FakeTransport(
        _token_response(malicious_token),
        _response({"StockAccountWallet": 1_000_000.0}),
    )
    client = _KabuSClient("validation", transport=fake)

    with pytest.raises(KabuSProtocolError, match="invalid Token"):
        client.authenticate("password")

    assert client.authenticated is False
    assert len(fake.requests) == 1
    assert all("X-API-KEY" not in request.headers for request in fake.requests)
    with pytest.raises(KabuSAuthenticationRequired):
        client.get_cash_wallet()
    assert len(fake.requests) == 1


def test_symbol_specific_cash_wallet_uses_exact_official_endpoint() -> None:
    fake = FakeTransport(
        _token_response(),
        _response(
            {
                "StockAccountWallet": 1_000_000.0,
                "AuKCStockAccountWallet": 900_000.0,
            }
        ),
    )
    client = _KabuSClient("validation", transport=fake)
    client.authenticate("password")

    wallet = client.get_cash_wallet("9433", 27)

    assert wallet["AuKCStockAccountWallet"] == 900_000.0
    assert fake.requests[-1].url == "http://127.0.0.1:18081/kabusapi/wallet/cash/9433@27"

    with pytest.raises(ValueError, match="cash wallet exchange"):
        client.get_cash_wallet("9433", 2)  # type: ignore[arg-type]


def test_cash_read_endpoints_are_exact_and_tolerate_optional_null_fields() -> None:
    fake = FakeTransport(
        _token_response(),
        _response([{"ID": "ORDER1", "State": 5, "Details": None}]),
        _response([{"Symbol": "9433", "LeavesQty": 100.0, "HoldQty": None}]),
        _response(
            {
                "StockAccountWallet": None,
                "AuKCStockAccountWallet": 100000.0,
                "AuJbnStockAccountWallet": 0.0,
            }
        ),
        _response({"Symbol": "9433", "CurrentPrice": None}),
        _response({"Symbol": "9433", "TradingUnit": 100.0}),
        _response({"Stock": 200.0, "KabuSVersion": "5.43.0.0"}),
    )
    client = _KabuSClient("production", transport=fake)
    client.authenticate("password")

    assert client.get_orders("ORDER1")[0]["Details"] is None
    assert client.get_positions()[0]["HoldQty"] is None
    assert client.get_cash_wallet()["StockAccountWallet"] is None
    assert client.get_board("9433", 1)["CurrentPrice"] is None
    assert client.get_symbol("9433", 1)["TradingUnit"] == 100.0
    assert client.get_api_soft_limit()["Stock"] == 200.0

    urls = [request.url for request in fake.requests[1:]]
    assert urls == [
        "http://127.0.0.1:18080/kabusapi/orders?product=1&details=true&id=ORDER1",
        "http://127.0.0.1:18080/kabusapi/positions?product=1&addinfo=true",
        "http://127.0.0.1:18080/kabusapi/wallet/cash",
        "http://127.0.0.1:18080/kabusapi/board/9433@1",
        "http://127.0.0.1:18080/kabusapi/symbol/9433@1",
        "http://127.0.0.1:18080/kabusapi/apisoftlimit",
    ]

    with pytest.raises(ValueError, match="reference_exchange"):
        client.get_board("9433", 9)  # type: ignore[arg-type]


def test_all_product_orders_use_exact_product_zero_query() -> None:
    fake = FakeTransport(
        _token_response(),
        _response([{"ID": "MARGIN-001", "State": 1, "Symbol": "9433"}]),
    )
    client = _KabuSClient("validation", transport=fake)
    client.authenticate("password")

    orders = client.get_orders(product=0)

    assert orders[0]["ID"] == "MARGIN-001"
    assert fake.requests[-1].url == (
        "http://127.0.0.1:18081/kabusapi/orders?product=0&details=true"
    )
    for invalid_product in (True, 5):
        with pytest.raises(ValueError, match="order product"):
            client.get_orders(product=invalid_product)  # type: ignore[arg-type]


def test_send_and_cancel_use_exact_official_cash_payloads() -> None:
    fake = FakeTransport(
        _token_response(),
        _response({"Result": 0, "OrderId": "ORDER-BUY"}),
        _response({"Result": 0, "OrderId": "ORDER-SELL"}),
        _response({"Result": 0, "OrderId": "ORDER-SELL"}),
    )
    client = _KabuSClient("validation", transport=fake)
    client.authenticate("password")

    buy = client.send_cash_order(
        _intent(),
        account_type=4,
        cash_buy_deliv_type=2,
        cash_buy_fund_type="02",
    )
    sell = client.send_cash_order(
        _intent(
            client_order_id="client-002",
            exchange=3,
            reference_exchange=3,
            side="sell",
            order_type="limit",
            limit_price=900.0,
            expire_day=20260716,
        ),
        account_type=2,
        cash_buy_deliv_type=3,
        cash_buy_fund_type="AA",
    )
    cancelled = client.cancel_order("ORDER-SELL")

    assert buy.order_id == "ORDER-BUY"
    assert sell.order_id == "ORDER-SELL"
    assert cancelled.order_id == "ORDER-SELL"
    buy_request, sell_request, cancel_request = fake.requests[1:]
    assert buy_request.method == "POST"
    assert buy_request.url == "http://127.0.0.1:18081/kabusapi/sendorder"
    assert json.loads(buy_request.body or b"") == {
        "Symbol": "9433",
        "Exchange": 27,
        "SecurityType": 1,
        "Side": "2",
        "CashMargin": 1,
        "DelivType": 2,
        "FundType": "02",
        "AccountType": 4,
        "Qty": 100,
        "FrontOrderType": 20,
        "Price": 123.5,
        "ExpireDay": 20260716,
    }
    assert json.loads(sell_request.body or b"") == {
        "Symbol": "9433",
        "Exchange": 3,
        "SecurityType": 1,
        "Side": "1",
        "CashMargin": 1,
        "DelivType": 0,
        "FundType": "  ",
        "AccountType": 2,
        "Qty": 100,
        "FrontOrderType": 20,
        "Price": 900.0,
        "ExpireDay": 20260716,
    }
    for forbidden in ("client_order_id", "strategy_id", "APIPassword", "Password"):
        assert forbidden not in (buy_request.body or b"").decode()
    assert cancel_request.method == "PUT"
    assert cancel_request.url == "http://127.0.0.1:18081/kabusapi/cancelorder"
    assert json.loads(cancel_request.body or b"") == {"OrderId": "ORDER-SELL"}


def test_clear_api_rejection_is_structured_and_sanitized() -> None:
    fake = FakeTransport(
        _token_response(),
        _response(
            {"Code": 4001009, "Message": "server echoed private-text"},
            status_code=401,
        ),
    )
    client = _KabuSClient("validation", transport=fake)
    client.authenticate("password")

    with pytest.raises(KabuSAPIRejection) as raised:
        client.get_cash_wallet()

    assert raised.value.status_code == 401
    assert raised.value.api_code == 4001009
    assert "private-text" not in str(raised.value)


@pytest.mark.parametrize(
    "response",
    [
        _raw_response(b'{"Result":0,"Result":0,"OrderId":"ORDER1"}'),
        _raw_response(b'{"Result":0}'),
        _raw_response(b"not-json"),
        _raw_response(b'{"Code":4001005,"Code":4001005}', status_code=400),
    ],
)
def test_malformed_mutation_responses_are_ambiguous(response: TransportResponse) -> None:
    fake = FakeTransport(_token_response(), response)
    client = _KabuSClient("validation", transport=fake)
    client.authenticate("password")

    with pytest.raises(AmbiguousMutationError, match="do not retry"):
        client.send_cash_order(
            _intent(),
            account_type=4,
            cash_buy_deliv_type=2,
            cash_buy_fund_type="02",
        )

    assert len(fake.requests) == 2


def test_transport_failure_makes_send_ambiguous_and_is_never_retried() -> None:
    fake = FakeTransport(_token_response(), TransportFailure("private transport detail"))
    client = _KabuSClient("validation", transport=fake)
    client.authenticate("password")

    with pytest.raises(AmbiguousMutationError) as raised:
        client.send_cash_order(
            _intent(),
            account_type=4,
            cash_buy_deliv_type=2,
            cash_buy_fund_type="02",
        )

    assert "private transport detail" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert len(fake.requests) == 2


@pytest.mark.parametrize("mutation", ["send", "cancel"])
def test_application_level_mutation_failure_is_ambiguous(mutation: str) -> None:
    fake = FakeTransport(_token_response(), _response({"Result": 21, "OrderId": ""}))
    client = _KabuSClient("validation", transport=fake)
    client.authenticate("password")

    with pytest.raises(AmbiguousMutationError, match="do not retry"):
        if mutation == "send":
            client.send_cash_order(
                _intent(),
                account_type=4,
                cash_buy_deliv_type=2,
                cash_buy_fund_type="02",
            )
        else:
            client.cancel_order("ORDER-001")

    assert len(fake.requests) == 2


@pytest.mark.parametrize("mutation", ["send", "cancel"])
@pytest.mark.parametrize("status_code", [201, 302, 400, 500, 503])
def test_non_success_mutation_status_is_always_ambiguous(
    mutation: str,
    status_code: int,
) -> None:
    fake = FakeTransport(
        _token_response(),
        _response({"Code": 4001001, "Message": "structured"}, status_code=status_code),
    )
    client = _KabuSClient("validation", transport=fake)
    client.authenticate("password")

    with pytest.raises(AmbiguousMutationError, match="do not retry"):
        if mutation == "send":
            client.send_cash_order(
                _intent(),
                account_type=4,
                cash_buy_deliv_type=2,
                cash_buy_fund_type="02",
            )
        else:
            client.cancel_order("ORDER-001")

    assert len(fake.requests) == 2


def test_duplicate_and_oversized_read_responses_fail_closed() -> None:
    duplicate = FakeTransport(_token_response(), _raw_response(b'[{"ID":"A","ID":"B"}]'))
    duplicate_client = _KabuSClient("validation", transport=duplicate)
    duplicate_client.authenticate("password")
    with pytest.raises(KabuSProtocolError, match="invalid JSON"):
        duplicate_client.get_orders()

    oversized = FakeTransport(_token_response("x"), _raw_response(b" " * 65))
    oversized_client = _KabuSClient(
        "validation",
        transport=oversized,
        max_response_bytes=64,
    )
    oversized_client.authenticate("password")
    with pytest.raises(KabuSTransportError, match="byte limit"):
        oversized_client.get_cash_wallet()


def test_urllib_transport_installs_no_proxy_and_no_redirect_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers: list[object] = []

    class NeverOpen:
        def open(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("the opener must not be used in this test")

    def fake_build_opener(*configured_handlers: object) -> Any:
        handlers.extend(configured_handlers)
        return NeverOpen()

    monkeypatch.setattr(kabus_module, "build_opener", fake_build_opener)

    UrllibTransport()

    proxy_handlers = [handler for handler in handlers if isinstance(handler, ProxyHandler)]
    redirect_handlers = [
        handler for handler in handlers if isinstance(handler, HTTPRedirectHandler)
    ]
    assert len(proxy_handlers) == 1
    assert proxy_handlers[0].proxies == {}
    assert len(redirect_handlers) == 1
    assert (
        redirect_handlers[0].redirect_request(
            Request("http://127.0.0.1:18080/kabusapi/token"),
            None,
            302,
            "redirect",
            {},
            "http://example.invalid/credential-sink",
        )
        is None
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:18080/kabusapi/token",
        "http://127.0.0.2:18080/kabusapi/token",
        "http://127.0.0.1:18082/kabusapi/token",
        "https://127.0.0.1:18080/kabusapi/token",
        "http://127.0.0.1:18080/not-kabusapi/token",
        "http://127.0.0.1:18080/kabusapi/token#redirect",
        "http://user@127.0.0.1:18080/kabusapi/token",
    ],
)
def test_urllib_transport_rejects_non_kabus_origins_before_opening(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
) -> None:
    class RecordingOpener:
        opened = False

        def open(self, *_args: object, **_kwargs: object) -> None:
            self.opened = True
            raise AssertionError("an invalid origin must never be opened")

    opener = RecordingOpener()
    monkeypatch.setattr(kabus_module, "build_opener", lambda *_handlers: opener)
    transport = UrllibTransport()

    with pytest.raises(TransportFailure, match="loopback boundary"):
        transport.request(
            method="POST",
            url=url,
            headers={"X-API-KEY": "must-not-leave-loopback"},
            body=b'{"APIPassword":"must-not-leave-loopback"}',
            timeout_seconds=1.0,
            max_response_bytes=1024,
        )

    assert opener.opened is False


def test_urllib_transport_blocks_valid_kabus_request_off_windows_before_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingOpener:
        opened = False

        def open(self, *_args: object, **_kwargs: object) -> None:
            self.opened = True
            raise AssertionError("a non-Windows host must never open the request")

    opener = RecordingOpener()
    monkeypatch.setattr(kabus_module, "build_opener", lambda *_handlers: opener)
    monkeypatch.setattr(kabus_module.platform, "system", lambda: "Darwin")
    transport = UrllibTransport()

    with pytest.raises(TransportFailure, match="same Windows PC"):
        transport.request(
            method="POST",
            url="http://127.0.0.1:18081/kabusapi/token",
            headers={"Content-Type": "application/json"},
            body=b'{"APIPassword":"must-not-leave-this-process"}',
            timeout_seconds=1.0,
            max_response_bytes=1024,
        )

    assert opener.opened is False


def test_urllib_transport_converts_bad_status_line_during_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingOpener:
        def open(self, *_args: object, **_kwargs: object) -> None:
            raise BadStatusLine("private malformed status line")

    monkeypatch.setattr(kabus_module, "build_opener", lambda *_handlers: FailingOpener())
    monkeypatch.setattr(kabus_module.platform, "system", lambda: "Windows")
    transport = UrllibTransport()

    with pytest.raises(TransportFailure, match="loopback transport failed") as raised:
        transport.request(
            method="GET",
            url="http://127.0.0.1:18081/kabusapi/orders",
            headers={"X-API-KEY": "safe-token"},
            body=None,
            timeout_seconds=1.0,
            max_response_bytes=1024,
        )

    assert "private malformed status line" not in str(raised.value)


def test_urllib_transport_converts_incomplete_success_body_and_closes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IncompleteResponse:
        status = 200

        def __init__(self) -> None:
            self.closed = False

        def read(self, _size: int) -> bytes:
            raise IncompleteRead(b"private partial response", 100)

        def close(self) -> None:
            self.closed = True

    response = IncompleteResponse()

    class IncompleteOpener:
        def open(self, *_args: object, **_kwargs: object) -> IncompleteResponse:
            return response

    monkeypatch.setattr(kabus_module, "build_opener", lambda *_handlers: IncompleteOpener())
    monkeypatch.setattr(kabus_module.platform, "system", lambda: "Windows")
    transport = UrllibTransport()

    with pytest.raises(TransportFailure, match="loopback response failed") as raised:
        transport.request(
            method="GET",
            url="http://127.0.0.1:18081/kabusapi/orders",
            headers={"X-API-KEY": "safe-token"},
            body=None,
            timeout_seconds=1.0,
            max_response_bytes=1024,
        )

    assert "private partial response" not in str(raised.value)
    assert response.closed is True


def test_urllib_transport_converts_incomplete_error_body_and_closes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IncompleteBody:
        def __init__(self) -> None:
            self.closed = False

        def read(self, _size: int) -> bytes:
            raise IncompleteRead(b"private partial error", 100)

        def close(self) -> None:
            self.closed = True

    body = IncompleteBody()
    error = HTTPError(
        "http://127.0.0.1:18081/kabusapi/token",
        500,
        "private response reason",
        {},
        body,
    )

    class FailingOpener:
        def open(self, *_args: object, **_kwargs: object) -> None:
            raise error

    monkeypatch.setattr(kabus_module, "build_opener", lambda *_handlers: FailingOpener())
    monkeypatch.setattr(kabus_module.platform, "system", lambda: "Windows")
    transport = UrllibTransport()

    with pytest.raises(TransportFailure, match="error response failed") as raised:
        transport.request(
            method="POST",
            url="http://127.0.0.1:18081/kabusapi/token",
            headers={"Content-Type": "application/json"},
            body=b'{"APIPassword":"must-not-appear"}',
            timeout_seconds=1.0,
            max_response_bytes=1024,
        )

    assert "private partial error" not in str(raised.value)
    assert body.closed is True


def test_incomplete_mutation_response_becomes_ambiguous_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StaticResponse:
        status = 200

        def __init__(self, payload: bytes | None) -> None:
            self.payload = payload
            self.closed = False

        def read(self, _size: int) -> bytes:
            if self.payload is None:
                raise IncompleteRead(b"private mutation fragment", 100)
            return self.payload

        def close(self) -> None:
            self.closed = True

    token_response = StaticResponse(b'{"ResultCode":0,"Token":"safe-token"}')
    mutation_response = StaticResponse(None)

    class SequenceOpener:
        def __init__(self) -> None:
            self.responses = [token_response, mutation_response]
            self.open_count = 0

        def open(self, *_args: object, **_kwargs: object) -> StaticResponse:
            self.open_count += 1
            return self.responses.pop(0)

    opener = SequenceOpener()
    monkeypatch.setattr(kabus_module, "build_opener", lambda *_handlers: opener)
    monkeypatch.setattr(kabus_module.platform, "system", lambda: "Windows")
    client = _KabuSClient("validation", transport=UrllibTransport())
    client.authenticate("password")

    with pytest.raises(AmbiguousMutationError, match="do not retry") as raised:
        client.send_cash_order(
            _intent(),
            account_type=4,
            cash_buy_deliv_type=2,
            cash_buy_fund_type="02",
        )

    assert "private mutation fragment" not in str(raised.value)
    assert opener.open_count == 2
    assert token_response.closed is True
    assert mutation_response.closed is True


def test_urllib_transport_converts_http_error_body_read_failure_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingBody:
        def __init__(self) -> None:
            self.closed = False

        def read(self, _size: int) -> bytes:
            raise OSError("private body-read detail")

        def close(self) -> None:
            self.closed = True

    body = FailingBody()
    error = HTTPError(
        "http://127.0.0.1:18081/kabusapi/token",
        500,
        "private response reason",
        {},
        body,
    )

    class FailingOpener:
        def open(self, *_args: object, **_kwargs: object) -> None:
            raise error

    monkeypatch.setattr(kabus_module, "build_opener", lambda *_handlers: FailingOpener())
    monkeypatch.setattr(kabus_module.platform, "system", lambda: "Windows")
    transport = UrllibTransport()

    with pytest.raises(TransportFailure, match="error response failed") as raised:
        transport.request(
            method="POST",
            url="http://127.0.0.1:18081/kabusapi/token",
            headers={"Content-Type": "application/json"},
            body=b'{"APIPassword":"must-not-appear"}',
            timeout_seconds=1.0,
            max_response_bytes=1024,
        )

    assert "private body-read detail" not in str(raised.value)
    assert body.closed is True


def test_loopback_connection_rejects_unverified_peer_before_http_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSocket:
        def __init__(self, peer: tuple[str, int]) -> None:
            self.peer = peer
            self.closed = False

        def getpeername(self) -> tuple[str, int]:
            return self.peer

        def close(self) -> None:
            self.closed = True

    fake_socket = FakeSocket(("192.0.2.10", 18080))

    def fake_connect(connection: HTTPConnection) -> None:
        connection.sock = fake_socket  # type: ignore[assignment]

    monkeypatch.setattr(HTTPConnection, "connect", fake_connect)
    connection = kabus_module._LoopbackHTTPConnection("127.0.0.1", 18080)

    with pytest.raises(OSError, match="outside the loopback boundary"):
        connection.connect()

    assert fake_socket.closed is True
    assert connection.sock is None


def test_loopback_connection_accepts_exact_numeric_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSocket:
        def getpeername(self) -> tuple[str, int]:
            return ("127.0.0.1", 18081)

    fake_socket = FakeSocket()

    def fake_connect(connection: HTTPConnection) -> None:
        connection.sock = fake_socket  # type: ignore[assignment]

    monkeypatch.setattr(HTTPConnection, "connect", fake_connect)
    connection = kabus_module._LoopbackHTTPConnection("127.0.0.1", 18081)

    connection.connect()

    assert connection.sock is fake_socket


def test_loopback_connection_rejects_hostname_before_connecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_connect_called = False

    def fake_connect(_connection: HTTPConnection) -> None:
        nonlocal base_connect_called
        base_connect_called = True

    monkeypatch.setattr(HTTPConnection, "connect", fake_connect)
    connection = kabus_module._LoopbackHTTPConnection("localhost", 18080)

    with pytest.raises(OSError, match="outside the loopback boundary"):
        connection.connect()

    assert base_connect_called is False
