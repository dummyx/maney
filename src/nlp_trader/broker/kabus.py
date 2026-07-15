from __future__ import annotations

import json
import math
import platform
import re
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException
from ipaddress import IPv4Address, ip_address
from typing import Any, Literal, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    OpenerDirector,
    ProxyHandler,
    Request,
    build_opener,
)

from nlp_trader.broker.contracts import CashOrderIntent

DEFAULT_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_CONFIGURABLE_RESPONSE_BYTES = 8 * 1024 * 1024

_BASE_URLS = {
    "validation": "http://127.0.0.1:18081/kabusapi",
    "production": "http://127.0.0.1:18080/kabusapi",
}
_LOOPBACK_HOST = "127.0.0.1"
_LOOPBACK_ADDRESS = IPv4Address(_LOOPBACK_HOST)
_KABUS_PORTS = frozenset({18080, 18081})
_SYMBOL = re.compile(r"^[A-Z0-9]{1,16}$")
_ORDER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_API_TOKEN = re.compile(r"^[\x21-\x7e]{1,256}$")

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class KabuSClientError(RuntimeError):
    """Base class for sanitized kabuS client failures."""


class TransportFailure(RuntimeError):
    """Transport-level failure without a trustworthy HTTP response."""


class KabuSTransportError(KabuSClientError):
    """A read-only request failed before a trustworthy response was available."""


class KabuSProtocolError(KabuSClientError):
    """A read-only response violated the bounded kabuS response contract."""


class KabuSAuthenticationRequired(KabuSClientError):
    """An authenticated endpoint was called before obtaining an API token."""


class KabuSAPIRejection(KabuSClientError):
    """The API returned a structured, definitive rejection."""

    def __init__(self, operation: str, *, status_code: int, api_code: int) -> None:
        self.operation = operation
        self.status_code = status_code
        self.api_code = api_code
        super().__init__(f"kabuS rejected {operation} (HTTP {status_code}, API code {api_code})")


class AmbiguousMutationError(KabuSClientError):
    """A mutation may have reached kabuS, so it must not be retried blindly."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(
            f"kabuS {operation} outcome is ambiguous; do not retry without reconciliation"
        )


class _DuplicateJsonKeyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TransportResponse:
    status_code: int
    body: bytes


@dataclass(frozen=True, slots=True)
class OrderResult:
    result_code: int
    order_id: str


class Transport(Protocol):
    """Single-attempt HTTP transport used by the internal kabuS client."""

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> TransportResponse: ...


class UrllibTransport:
    """Minimal single-attempt loopback transport implemented with urllib."""

    def __init__(self) -> None:
        # An empty ProxyHandler prevents urllib from consulting proxy environment
        # variables. The redirect handler rejects all redirects, and the HTTP
        # handler verifies the connected peer before urllib writes credentials.
        self._opener: OpenerDirector = build_opener(
            ProxyHandler({}),
            _NoRedirectHandler(),
            _LoopbackHTTPHandler(),
        )

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
        _validate_loopback_url(url)
        if platform.system() != "Windows":
            raise TransportFailure("kabuS authenticated transport requires the same Windows PC")
        try:
            request = Request(url, data=body, headers=dict(headers), method=method)
            response = self._opener.open(  # noqa: S310 - URL is checked above
                request,
                timeout=timeout_seconds,
            )
        except HTTPError as exc:
            try:
                try:
                    response_body = _read_bounded(exc, max_response_bytes)
                except (
                    TransportFailure,
                    HTTPException,
                    TimeoutError,
                    OSError,
                    ValueError,
                    UnicodeError,
                ):
                    raise TransportFailure("kabuS loopback error response failed") from None
            finally:
                with suppress(HTTPException, OSError, ValueError):
                    exc.close()
            return TransportResponse(status_code=exc.code, body=response_body)
        except (
            HTTPException,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
            UnicodeError,
        ):
            raise TransportFailure("kabuS loopback transport failed") from None

        try:
            status_code = response.status
            response_body = _read_bounded(response, max_response_bytes)
        except (
            TransportFailure,
            HTTPException,
            TimeoutError,
            OSError,
            ValueError,
            UnicodeError,
        ):
            raise TransportFailure("kabuS loopback response failed") from None
        finally:
            with suppress(HTTPException, OSError, ValueError):
                response.close()
        return TransportResponse(status_code=status_code, body=response_body)


class _NoRedirectHandler(HTTPRedirectHandler):
    """Turn redirect responses into HTTP errors without issuing another request."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class _LoopbackHTTPConnection(HTTPConnection):
    """Fail before request bytes reach a peer outside the fixed numeric loopback socket."""

    def connect(self) -> None:
        tunnel_host = getattr(self, "_tunnel_host", None)
        if self.host != _LOOPBACK_HOST or self.port not in _KABUS_PORTS or tunnel_host:
            raise OSError("kabuS connection target is outside the loopback boundary")

        super().connect()
        if self.sock is None:
            raise OSError("kabuS loopback connection did not expose its peer")

        try:
            peer = self.sock.getpeername()
            peer_address = ip_address(peer[0])
            peer_port = peer[1]
        except (IndexError, TypeError, ValueError, OSError):
            self.close()
            raise OSError("kabuS loopback peer could not be verified") from None

        if peer_address != _LOOPBACK_ADDRESS or peer_port != self.port:
            self.close()
            raise OSError("kabuS connection peer is outside the loopback boundary")


class _LoopbackHTTPHandler(HTTPHandler):
    def http_open(self, req: Request) -> Any:
        return self.do_open(_LoopbackHTTPConnection, req)


def _validate_loopback_url(url: str) -> None:
    """Accept only the two documented, numeric loopback kabuS origins."""
    if not isinstance(url, str) or any(ord(character) <= 0x20 for character in url):
        raise TransportFailure("kabuS URL is outside the loopback boundary")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise TransportFailure("kabuS URL is outside the loopback boundary") from None

    if (
        parsed.scheme != "http"
        or parsed.hostname != _LOOPBACK_HOST
        or port not in _KABUS_PORTS
        or parsed.netloc != f"{_LOOPBACK_HOST}:{port}"
        or parsed.fragment
        or not (parsed.path == "/kabusapi" or parsed.path.startswith("/kabusapi/"))
    ):
        raise TransportFailure("kabuS URL is outside the loopback boundary")


def _read_bounded(stream: Any, max_response_bytes: int) -> bytes:
    body = stream.read(max_response_bytes + 1)
    if not isinstance(body, bytes):
        raise TransportFailure("kabuS response body is not bytes")
    if len(body) > max_response_bytes:
        raise TransportFailure("kabuS response exceeds the byte limit")
    return body


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError
        result[key] = value
    return result


def _reject_non_finite_constant(_: str) -> None:
    raise ValueError


def _validated_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError
        return value
    if isinstance(value, list):
        return [_validated_json_value(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError
            result[key] = _validated_json_value(item)
        return result
    raise ValueError


def _parse_json_response(body: bytes, *, max_response_bytes: int) -> JsonValue:
    if not isinstance(body, bytes) or len(body) > max_response_bytes:
        raise KabuSProtocolError("kabuS returned an invalid bounded response")
    try:
        decoded = body.decode("utf-8")
        parsed: object = json.loads(
            decoded,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
        return _validated_json_value(parsed)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateJsonKeyError,
        RecursionError,
        ValueError,
    ):
        raise KabuSProtocolError("kabuS returned invalid JSON") from None


def _required_int(value: JsonValue, field_name: str) -> int:
    if type(value) is not int:
        raise KabuSProtocolError(f"kabuS response has invalid {field_name}")
    return value


def _required_nonempty_string(value: JsonValue, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or len(value) > 4096:
        raise KabuSProtocolError(f"kabuS response has invalid {field_name}")
    return value


def _required_order_id(value: JsonValue) -> str:
    order_id = _required_nonempty_string(value, "OrderId")
    if _ORDER_ID.fullmatch(order_id) is None:
        raise KabuSProtocolError("kabuS response has invalid OrderId")
    return order_id


def _required_api_token(value: JsonValue) -> str:
    token = _required_nonempty_string(value, "Token")
    if _API_TOKEN.fullmatch(token) is None:
        raise KabuSProtocolError("kabuS response has invalid Token")
    return token


class _KabuSClient:
    """Internal single-attempt client; mutations must be coordinated by the executor."""

    def __init__(
        self,
        environment: Literal["validation", "production"],
        *,
        transport: Transport | None = None,
        timeout_seconds: float = 5.0,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        if environment not in _BASE_URLS:
            raise ValueError("environment must be validation or production")
        if not math.isfinite(timeout_seconds) or not 0.1 <= timeout_seconds <= 30.0:
            raise ValueError("timeout_seconds must be between 0.1 and 30")
        if (
            type(max_response_bytes) is not int
            or not 1 <= max_response_bytes <= MAX_CONFIGURABLE_RESPONSE_BYTES
        ):
            raise ValueError("max_response_bytes is outside the supported range")

        self._environment = environment
        self._base_url = _BASE_URLS[environment]
        self._transport = transport if transport is not None else UrllibTransport()
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._token: str | None = None

    @property
    def environment(self) -> Literal["validation", "production"]:
        return self._environment

    @property
    def authenticated(self) -> bool:
        return self._token is not None

    def clear_token(self) -> None:
        """Forget the in-memory token without retaining any credential material."""
        self._token = None

    def authenticate(self, api_password: str) -> None:
        """Acquire the single active API token; the password is never retained."""
        if not isinstance(api_password, str) or not api_password:
            raise ValueError("api_password must be a non-empty string")
        self._token = None
        response = self._perform_request(
            method="POST",
            path="/token",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            body=_encode_json({"APIPassword": api_password}),
        )
        payload = self._decode_read_response(response, operation="token authentication")
        if not isinstance(payload, dict):
            raise KabuSProtocolError("kabuS token response must be an object")
        result_code = _required_int(payload.get("ResultCode"), "ResultCode")
        if result_code != 0:
            raise KabuSAPIRejection(
                "token authentication",
                status_code=response.status_code,
                api_code=result_code,
            )
        self._token = _required_api_token(payload.get("Token"))

    def get_orders(
        self,
        order_id: str | None = None,
        *,
        product: int = 1,
    ) -> list[JsonObject]:
        if type(product) is not int or product not in (0, 1, 2, 3, 4):
            raise ValueError("order product must be 0, 1, 2, 3, or 4")
        query: list[tuple[str, str]] = [("product", str(product)), ("details", "true")]
        if order_id is not None:
            query.append(("id", _validated_order_id(order_id)))
        payload = self._authorized_read(f"/orders?{urlencode(query)}", "orders")
        return _object_list(payload, "orders")

    def get_positions(self, *, product: Literal[0, 1] = 1) -> list[JsonObject]:
        if product not in (0, 1):
            raise ValueError("position product must be 0 or 1")
        payload = self._authorized_read(
            f"/positions?product={product}&addinfo=true",
            "positions",
        )
        return _object_list(payload, "positions")

    def get_cash_wallet(
        self,
        symbol: str | None = None,
        exchange: Literal[1, 3, 5, 6, 9, 27] | None = None,
    ) -> JsonObject:
        if (symbol is None) != (exchange is None):
            raise ValueError("cash wallet symbol and exchange must be provided together")
        path = "/wallet/cash"
        if symbol is not None and exchange is not None:
            path = f"/wallet/cash/{_cash_wallet_symbol(symbol, exchange)}"
        return _object(self._authorized_read(path, "cash wallet"), "cash wallet")

    def get_board(self, symbol: str, reference_exchange: Literal[1, 3, 5, 6]) -> JsonObject:
        path_symbol = _reference_symbol(symbol, reference_exchange)
        return _object(self._authorized_read(f"/board/{path_symbol}", "board"), "board")

    def get_symbol(self, symbol: str, reference_exchange: Literal[1, 3, 5, 6]) -> JsonObject:
        path_symbol = _reference_symbol(symbol, reference_exchange)
        return _object(self._authorized_read(f"/symbol/{path_symbol}", "symbol"), "symbol")

    def get_api_soft_limit(self) -> JsonObject:
        return _object(
            self._authorized_read("/apisoftlimit", "API soft limit"),
            "API soft limit",
        )

    def send_cash_order(
        self,
        intent: CashOrderIntent,
        *,
        account_type: Literal[2, 4, 12],
        cash_buy_deliv_type: Literal[2, 3],
        cash_buy_fund_type: Literal["02", "AA"],
    ) -> OrderResult:
        payload = _cash_order_payload(
            intent,
            account_type=account_type,
            cash_buy_deliv_type=cash_buy_deliv_type,
            cash_buy_fund_type=cash_buy_fund_type,
        )
        return self._mutation("POST", "/sendorder", payload, operation="send order")

    def cancel_order(self, order_id: str) -> OrderResult:
        payload: JsonObject = {"OrderId": _validated_order_id(order_id)}
        return self._mutation("PUT", "/cancelorder", payload, operation="cancel order")

    def _authorized_read(self, path: str, operation: str) -> JsonValue:
        response = self._perform_request(
            method="GET",
            path=path,
            headers=self._authorized_headers(has_body=False),
            body=None,
        )
        return self._decode_read_response(response, operation=operation)

    def _mutation(
        self,
        method: Literal["POST", "PUT"],
        path: str,
        payload: JsonObject,
        *,
        operation: str,
    ) -> OrderResult:
        try:
            response = self._perform_request(
                method=method,
                path=path,
                headers=self._authorized_headers(has_body=True),
                body=_encode_json(payload),
            )
        except KabuSTransportError:
            raise AmbiguousMutationError(operation) from None

        if response.status_code != 200:
            raise AmbiguousMutationError(operation)

        try:
            decoded = _parse_json_response(
                response.body,
                max_response_bytes=self._max_response_bytes,
            )
            if not isinstance(decoded, dict):
                raise KabuSProtocolError("kabuS mutation response must be an object")
            result_code = _required_int(decoded.get("Result"), "Result")
            if result_code != 0:
                raise AmbiguousMutationError(operation)
            order_id = _required_order_id(decoded.get("OrderId"))
        except AmbiguousMutationError:
            raise
        except KabuSProtocolError:
            raise AmbiguousMutationError(operation) from None
        return OrderResult(result_code=result_code, order_id=order_id)

    def _authorized_headers(self, *, has_body: bool) -> dict[str, str]:
        if self._token is None:
            raise KabuSAuthenticationRequired("authenticate before calling the kabuS API")
        headers = {"Accept": "application/json", "X-API-KEY": self._token}
        if has_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _perform_request(
        self,
        *,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes | None,
    ) -> TransportResponse:
        try:
            response = self._transport.request(
                method=method,
                url=f"{self._base_url}{path}",
                headers=headers,
                body=body,
                timeout_seconds=self._timeout_seconds,
                max_response_bytes=self._max_response_bytes,
            )
        except TransportFailure:
            raise KabuSTransportError("kabuS transport failed") from None
        if (
            type(response.status_code) is not int
            or not 100 <= response.status_code <= 599
            or not isinstance(response.body, bytes)
        ):
            raise KabuSTransportError("kabuS transport returned an invalid response")
        if len(response.body) > self._max_response_bytes:
            raise KabuSTransportError("kabuS response exceeds the byte limit")
        return response

    def _decode_read_response(self, response: TransportResponse, *, operation: str) -> JsonValue:
        if response.status_code != 200:
            raise self._rejection_from_response(response, operation=operation)
        return _parse_json_response(
            response.body,
            max_response_bytes=self._max_response_bytes,
        )

    def _rejection_from_response(
        self,
        response: TransportResponse,
        *,
        operation: str,
    ) -> KabuSAPIRejection:
        decoded = _parse_json_response(
            response.body,
            max_response_bytes=self._max_response_bytes,
        )
        if not isinstance(decoded, dict):
            raise KabuSProtocolError("kabuS error response must be an object")
        code = _required_int(decoded.get("Code"), "Code")
        if not isinstance(decoded.get("Message"), str):
            raise KabuSProtocolError("kabuS error response has invalid Message")
        return KabuSAPIRejection(operation, status_code=response.status_code, api_code=code)


def _encode_json(payload: Mapping[str, JsonValue]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _cash_order_payload(
    intent: CashOrderIntent,
    *,
    account_type: Literal[2, 4, 12],
    cash_buy_deliv_type: Literal[2, 3],
    cash_buy_fund_type: Literal["02", "AA"],
) -> JsonObject:
    """Build the exact non-secret request body used for audit and submission."""

    validated_account_type = _account_type(account_type)
    validated_deliv_type = _cash_buy_deliv_type(cash_buy_deliv_type)
    validated_fund_type = _cash_buy_fund_type(cash_buy_fund_type)
    is_buy = intent.side == "buy"
    return {
        "Symbol": intent.symbol,
        "Exchange": intent.exchange,
        "SecurityType": 1,
        "Side": "2" if is_buy else "1",
        "CashMargin": 1,
        "DelivType": validated_deliv_type if is_buy else 0,
        "FundType": validated_fund_type if is_buy else "  ",
        "AccountType": validated_account_type,
        "Qty": intent.quantity,
        "FrontOrderType": 20,
        "Price": intent.limit_price,
        "ExpireDay": intent.expire_day,
    }


def _validated_order_id(order_id: str) -> str:
    if not isinstance(order_id, str) or _ORDER_ID.fullmatch(order_id) is None:
        raise ValueError("order_id has an invalid format")
    return order_id


def _account_type(value: object) -> Literal[2, 4, 12]:
    if type(value) is not int or value not in (2, 4, 12):
        raise ValueError("account_type must be 2, 4, or 12")
    return cast(Literal[2, 4, 12], value)


def _cash_buy_deliv_type(value: object) -> Literal[2, 3]:
    if type(value) is not int or value not in (2, 3):
        raise ValueError("cash_buy_deliv_type must be 2 or 3")
    return cast(Literal[2, 3], value)


def _cash_buy_fund_type(value: object) -> Literal["02", "AA"]:
    if value not in ("02", "AA"):
        raise ValueError("cash_buy_fund_type must be 02 or AA")
    return value


def _reference_symbol(symbol: str, reference_exchange: int) -> str:
    if not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None:
        raise ValueError("symbol must contain only uppercase ASCII letters and digits")
    if type(reference_exchange) is not int or reference_exchange not in (1, 3, 5, 6):
        raise ValueError("reference_exchange must be 1, 3, 5, or 6")
    return f"{quote(symbol, safe='')}@{reference_exchange}"


def _cash_wallet_symbol(symbol: str, exchange: int) -> str:
    if not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None:
        raise ValueError("symbol must contain only uppercase ASCII letters and digits")
    if type(exchange) is not int or exchange not in (1, 3, 5, 6, 9, 27):
        raise ValueError("cash wallet exchange must be 1, 3, 5, 6, 9, or 27")
    return f"{quote(symbol, safe='')}@{exchange}"


def _object(value: JsonValue, name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise KabuSProtocolError(f"kabuS {name} response must be an object")
    return value


def _object_list(value: JsonValue, name: str) -> list[JsonObject]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise KabuSProtocolError(f"kabuS {name} response must be an array of objects")
    return cast(list[JsonObject], value)
