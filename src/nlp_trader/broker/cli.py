from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from pydantic import ValidationError

from nlp_trader.broker.audit import (
    BrokerAuditLedger,
    BrokerAuditLockError,
    BrokerAuditValidationError,
)
from nlp_trader.broker.config import KabuSConfig, KabuSSecrets, load_kabus_config
from nlp_trader.broker.contracts import (
    MAX_CASH_ORDER_INTENT_BYTES,
    CashOrderIntent,
    load_cash_order_intent,
)
from nlp_trader.broker.execution import BrokerSafetyError, KabuSExecutor
from nlp_trader.broker.kabus import KabuSClientError, _KabuSClient
from nlp_trader.broker.state import KabuSStateLockError, KabuSStatePaths

DEFAULT_BROKER_CONFIG = Path("configs/kabus.validation.yaml")

BrokerConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        dir_okay=False,
        readable=True,
        help="Separate non-secret kabuS broker configuration.",
    ),
]
OrderOption = Annotated[
    Path,
    typer.Option(
        "--order",
        dir_okay=False,
        readable=True,
        help="Explicit kabuS cash-order intent JSON.",
    ),
]

broker_app = typer.Typer(
    help="Explicit, separately configured kabuS cash-equity operations.",
    no_args_is_help=True,
)


def _echo_json(value: object) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


def _fail(exc: Exception) -> None:
    if isinstance(exc, ValidationError):
        errors = exc.errors(include_url=False, include_context=False, include_input=False)
        details = []
        for error in errors[:10]:
            location = ".".join(str(part) for part in error.get("loc", ())) or "input"
            details.append(f"{location} [{error.get('type', 'invalid')}]")
        if len(errors) > len(details):
            details.append(f"and {len(errors) - len(details)} more")
        message = "input validation failed: " + ", ".join(details)
    else:
        message = str(exc)
    typer.echo(f"broker operation failed: {message}", err=True)
    raise typer.Exit(code=1)


def _load_intent(path: Path) -> CashOrderIntent:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot inspect order intent: {path}") from exc
    if size > MAX_CASH_ORDER_INTENT_BYTES:
        raise ValueError("cash order intent exceeds the byte limit")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read order intent: {path}") from exc
    return load_cash_order_intent(payload)


def _offline_executor(config: KabuSConfig) -> KabuSExecutor:
    state_paths = _state_paths()
    client = _KabuSClient(config.environment, timeout_seconds=config.timeout_seconds)
    ledger = BrokerAuditLedger(state_paths.audit_ledger_path)
    return KabuSExecutor(config, client, ledger, state_paths=state_paths)


def _authenticated_executor(config: KabuSConfig) -> KabuSExecutor:
    if platform.system() != "Windows":
        raise BrokerSafetyError(
            "authenticated kabuS commands require the same Windows PC as kabuStation"
        )
    state_paths = _state_paths()
    secrets = KabuSSecrets()  # type: ignore[call-arg]  # populated from environment
    client = _KabuSClient(config.environment, timeout_seconds=config.timeout_seconds)
    ledger = BrokerAuditLedger(state_paths.audit_ledger_path)
    return KabuSExecutor(
        config,
        client,
        ledger,
        state_paths=state_paths,
        authenticate=lambda: client.authenticate(secrets.api_password.get_secret_value()),
    )


def _state_paths() -> KabuSStatePaths:
    """Return the one current-user safety-state scope shared by every broker config."""

    return KabuSStatePaths.for_current_user()


def _command_errors() -> tuple[type[Exception], ...]:
    return (
        ValueError,
        OSError,
        KabuSClientError,
        BrokerSafetyError,
        BrokerAuditValidationError,
        BrokerAuditLockError,
        KabuSStateLockError,
    )


@broker_app.command("validate-config")
def validate_broker_config_command(
    config: BrokerConfigOption = DEFAULT_BROKER_CONFIG,
) -> None:
    """Validate broker-only configuration without reading secrets or making requests."""

    try:
        loaded = load_kabus_config(config)
        state_paths = _state_paths()
        _echo_json(
            {
                "ok": True,
                "environment": loaded.environment,
                "enabled": loaded.enabled,
                "order_submission_enabled": loaded.order_submission_enabled,
                "audit_ledger_path": str(state_paths.audit_ledger_path),
                "kill_switch_path": str(state_paths.kill_switch_path),
                "operation_lock_path": str(state_paths.operation_lock_path),
            }
        )
    except _command_errors() as exc:
        _fail(exc)


@broker_app.command("preview-order")
def preview_order_command(
    order: OrderOption,
    config: BrokerConfigOption = DEFAULT_BROKER_CONFIG,
) -> None:
    """Validate an intent and print its digest without secrets or network access."""

    try:
        loaded = load_kabus_config(config)
        intent = _load_intent(order)
        _echo_json(_offline_executor(loaded).preview_order(intent))
    except _command_errors() as exc:
        _fail(exc)


@broker_app.command("status")
def broker_status_command(
    config: BrokerConfigOption = DEFAULT_BROKER_CONFIG,
) -> None:
    """Read a sanitized kabuS account and local safety-state summary."""

    try:
        executor = _authenticated_executor(load_kabus_config(config))
        _echo_json(executor.status().public_dict())
    except _command_errors() as exc:
        _fail(exc)


@broker_app.command("submit-order")
def submit_order_command(
    order: OrderOption,
    confirmation: Annotated[
        str,
        typer.Option("--confirm", help="Exact digest printed by preview-order."),
    ],
    config: BrokerConfigOption = DEFAULT_BROKER_CONFIG,
    confirm_production: Annotated[
        str | None,
        typer.Option(
            "--confirm-production",
            help="Must equal REAL_ORDERS for production mutations.",
        ),
    ] = None,
) -> None:
    """Preflight and submit exactly one explicitly confirmed cash order."""

    try:
        intent = _load_intent(order)
        executor = _authenticated_executor(load_kabus_config(config))
        result = executor.submit_order(
            intent,
            confirmation=confirmation,
            production_confirmation=confirm_production,
        )
        _echo_json(result.to_dict())
    except _command_errors() as exc:
        _fail(exc)


@broker_app.command("reconcile")
def reconcile_command(
    config: BrokerConfigOption = DEFAULT_BROKER_CONFIG,
) -> None:
    """Record a sanitized broker snapshot without clearing ambiguous outcomes."""

    try:
        executor = _authenticated_executor(load_kabus_config(config))
        _echo_json(executor.reconcile().public_dict())
    except _command_errors() as exc:
        _fail(exc)


@broker_app.command("cancel-order")
def cancel_order_command(
    order_id: Annotated[str, typer.Option("--order-id", help="Broker order ID to cancel.")],
    client_action_id: Annotated[
        str,
        typer.Option("--client-action-id", help="Unique local cancellation identity."),
    ],
    confirmation: Annotated[
        str,
        typer.Option(
            "--confirm",
            help="Exact context-bound digest printed by preview-cancel.",
        ),
    ],
    config: BrokerConfigOption = DEFAULT_BROKER_CONFIG,
    confirm_production: Annotated[
        str | None,
        typer.Option("--confirm-production", help="REAL_ORDERS in production."),
    ] = None,
) -> None:
    """Cancel one confirmed open order; allowed while the kill switch is engaged."""

    try:
        executor = _authenticated_executor(load_kabus_config(config))
        result = executor.cancel_order(
            order_id,
            client_action_id=client_action_id,
            confirmation=confirmation,
            production_confirmation=confirm_production,
        )
        _echo_json({"result_code": result.result_code, "broker_order_id": result.order_id})
    except _command_errors() as exc:
        _fail(exc)


@broker_app.command("preview-cancel")
def preview_cancel_command(
    order_id: Annotated[str, typer.Option("--order-id", help="Broker order ID to inspect.")],
    client_action_id: Annotated[
        str,
        typer.Option("--client-action-id", help="Fresh local cancellation identity."),
    ],
    config: BrokerConfigOption = DEFAULT_BROKER_CONFIG,
) -> None:
    """Inspect one order and print the exact context-bound cancellation digest."""

    try:
        executor = _authenticated_executor(load_kabus_config(config))
        _echo_json(
            executor.preview_cancel(
                order_id,
                client_action_id=client_action_id,
            )
        )
    except _command_errors() as exc:
        _fail(exc)


@broker_app.command("resolve-unknown")
def resolve_unknown_command(
    client_order_id: Annotated[
        str,
        typer.Option("--client-order-id", help="Unresolved local order/action identity."),
    ],
    resolution: Annotated[
        str,
        typer.Option("--resolution", help="accepted or not-accepted."),
    ],
    confirmation: Annotated[
        str,
        typer.Option("--confirm", help="Exact documented resolution confirmation."),
    ],
    config: BrokerConfigOption = DEFAULT_BROKER_CONFIG,
    broker_order_id: Annotated[
        str | None,
        typer.Option("--broker-order-id", help="Required for accepted resolution."),
    ] = None,
    confirm_production: Annotated[
        str | None,
        typer.Option("--confirm-production", help="REAL_ORDERS in production."),
    ] = None,
) -> None:
    """Resolve an ambiguous mutation after reconciliation and operator review."""

    if resolution not in {"accepted", "not-accepted"}:
        _fail(ValueError("resolution must be accepted or not-accepted"))
    try:
        executor = _authenticated_executor(load_kabus_config(config))
        event = executor.resolve_unknown(
            client_order_id,
            resolution=cast(Literal["accepted", "not-accepted"], resolution),
            confirmation=confirmation,
            broker_order_id=broker_order_id,
            production_confirmation=confirm_production,
        )
        _echo_json(event)
    except _command_errors() as exc:
        _fail(exc)


@broker_app.command("engage-kill-switch")
def engage_kill_switch_command(
    reason: Annotated[str, typer.Option("--reason", help="Required audit reason.")],
    config: BrokerConfigOption = DEFAULT_BROKER_CONFIG,
) -> None:
    """Engage the local risk-off marker without broker credentials."""

    try:
        executor = _offline_executor(load_kabus_config(config))
        _echo_json(executor.engage_kill_switch(reason))
    except _command_errors() as exc:
        _fail(exc)


@broker_app.command("release-kill-switch")
def release_kill_switch_command(
    confirmation: Annotated[
        str,
        typer.Option(
            "--confirm",
            help="Exact RELEASE_KILL_SWITCH:<marker-digest> from status.",
        ),
    ],
    config: BrokerConfigOption = DEFAULT_BROKER_CONFIG,
) -> None:
    """Audit and release the local risk-off marker without broker credentials."""

    try:
        executor = _offline_executor(load_kabus_config(config))
        _echo_json(executor.release_kill_switch(confirmation=confirmation))
    except _command_errors() as exc:
        _fail(exc)
