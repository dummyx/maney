from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from nlp_trader.broker import cli as broker_cli
from nlp_trader.broker.config import KabuSConfig, KabuSSecrets, load_kabus_config
from nlp_trader.broker.contracts import CashOrderIntent, load_cash_order_intent
from nlp_trader.broker.execution import BrokerSubmission
from nlp_trader.broker.state import KabuSStatePaths
from nlp_trader.cli import app


def _config_payload() -> dict[str, Any]:
    return {
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
        "max_order_quantity": 100,
        "max_order_notional_jpy": 500_000,
        "max_daily_order_notional_jpy": 500_000,
        "max_position_quantity_per_symbol": 100,
        "max_position_notional_jpy_per_symbol": 500_000,
        "max_gross_cash_position_notional_jpy": 1_000_000,
        "max_total_unrealized_loss_jpy": 100_000,
        "max_open_orders": 1,
        "max_intent_age_seconds": 3_600,
        "max_future_intent_skew_seconds": 5,
        "max_preflight_duration_seconds": 30,
        "max_reconciliation_match_seconds": 40,
        "max_quote_age_seconds": 30,
        "max_price_deviation_bps": 50,
        "timeout_seconds": 5,
    }


def _write_config(
    tmp_path: Path,
    *,
    changes: dict[str, Any] | None = None,
    name: str = "broker.yaml",
) -> Path:
    payload = _config_payload()
    payload.update(changes or {})
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_order(
    tmp_path: Path,
    *,
    changes: dict[str, Any] | None = None,
    name: str = "order.json",
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": "kabus-cash-order-v1",
        "client_order_id": "client-001",
        "strategy_id": "strategy-001",
        "created_at": datetime.now(UTC).isoformat(),
        "symbol": "9433",
        "exchange": 27,
        "reference_exchange": 1,
        "side": "buy",
        "quantity": 100,
        "order_type": "limit",
        "limit_price": 123.5,
        "expire_day": int((datetime.now(UTC) + timedelta(days=7)).strftime("%Y%m%d")),
    }
    payload.update(changes or {})
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _forbidden_factory(*args: object, **kwargs: object) -> object:
    del args, kwargs
    raise AssertionError("secret or network factory must not be used")


def _use_test_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> KabuSStatePaths:
    state_paths = KabuSStatePaths((tmp_path / "shared-broker-state").resolve())
    monkeypatch.setattr(broker_cli, "_state_paths", lambda: state_paths)
    return state_paths


def test_root_mounts_broker_and_both_help_surfaces_are_available() -> None:
    runner = CliRunner()

    root_help = runner.invoke(app, ["--help"])
    broker_help = runner.invoke(app, ["broker", "--help"])

    assert root_help.exit_code == 0
    assert "broker" in root_help.stdout
    assert broker_help.exit_code == 0
    for command in (
        "validate-config",
        "preview-order",
        "status",
        "submit-order",
        "reconcile",
        "preview-cancel",
        "cancel-order",
        "resolve-unknown",
        "engage-kill-switch",
        "release-kill-switch",
    ):
        assert command in broker_help.stdout


def test_validate_config_is_secret_free_and_does_not_construct_a_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    state_paths = _use_test_state(tmp_path, monkeypatch)
    monkeypatch.setattr(broker_cli, "KabuSSecrets", _forbidden_factory)
    monkeypatch.setattr(broker_cli, "_KabuSClient", _forbidden_factory)

    result = CliRunner().invoke(
        app,
        ["broker", "validate-config", "--config", str(config_path)],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "audit_ledger_path": str(state_paths.audit_ledger_path),
        "enabled": True,
        "environment": "validation",
        "kill_switch_path": str(state_paths.kill_switch_path),
        "ok": True,
        "operation_lock_path": str(state_paths.operation_lock_path),
        "order_submission_enabled": True,
    }


def test_preview_order_is_secret_free_offline_and_prints_exact_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    order_path = _write_order(tmp_path)
    intent = load_cash_order_intent(order_path.read_bytes())
    config = load_kabus_config(config_path)
    _use_test_state(tmp_path, monkeypatch)
    constructed: list[tuple[str, float]] = []

    class OfflineClient:
        def __init__(self, environment: str, *, timeout_seconds: float) -> None:
            self.environment = environment
            constructed.append((environment, timeout_seconds))

        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"preview attempted client operation {name}")

    monkeypatch.setattr(broker_cli, "KabuSSecrets", _forbidden_factory)
    monkeypatch.setattr(broker_cli, "_KabuSClient", OfflineClient)

    result = CliRunner().invoke(
        app,
        [
            "broker",
            "preview-order",
            "--config",
            str(config_path),
            "--order",
            str(order_path),
        ],
    )

    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert output["confirmation_digest"] != intent.confirmation_digest()
    assert len(output["confirmation_digest"]) == 64
    assert output["intent_digest"] == intent.confirmation_digest()
    assert output["config_digest"] == config.digest()
    assert output["intent"] == json.loads(intent.canonical_json())
    assert output["effective_request_payload"]["Price"] == 123.5
    assert output["effective_request_payload"]["ExpireDay"] == intent.expire_day
    assert output["network_request_made"] is False
    assert output["limit_notional_jpy"] == 12_350
    assert constructed == [("validation", 5.0)]


def test_status_with_missing_secret_fails_without_leaking_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    _use_test_state(tmp_path, monkeypatch)
    secret_canary = "must-never-appear-in-cli-output"
    monkeypatch.delenv("NLP_TRADER_KABUS_API_PASSWORD", raising=False)
    monkeypatch.setenv("UNRELATED_SECRET_CANARY", secret_canary)

    def missing_secrets() -> KabuSSecrets:
        return KabuSSecrets(_env_file=None)

    monkeypatch.setattr(broker_cli, "KabuSSecrets", missing_secrets)
    monkeypatch.setattr(broker_cli, "_offline_executor", _forbidden_factory)
    monkeypatch.setattr(broker_cli.platform, "system", lambda: "Windows")

    result = CliRunner().invoke(
        app,
        ["broker", "status", "--config", str(config_path)],
    )

    assert result.exit_code == 1
    assert "broker operation failed" in result.stderr
    assert "api_password" in result.stderr
    assert secret_canary not in result.stdout
    assert secret_canary not in result.stderr


def test_submit_forwards_confirmations_and_emits_only_sanitized_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    order_path = _write_order(tmp_path)
    expected_intent = load_cash_order_intent(order_path.read_bytes())
    captured: dict[str, object] = {}
    secret_canary = "raw-api-password"
    monkeypatch.setenv("NLP_TRADER_KABUS_API_PASSWORD", secret_canary)

    class FakeExecutor:
        def submit_order(
            self,
            intent: CashOrderIntent,
            *,
            confirmation: str,
            production_confirmation: str | None,
        ) -> BrokerSubmission:
            captured.update(
                intent=intent,
                confirmation=confirmation,
                production_confirmation=production_confirmation,
            )
            return BrokerSubmission(
                client_order_id=intent.client_order_id,
                broker_order_id="broker-order-001",
                intent_digest=intent.confirmation_digest(),
                environment="validation",
                notional_jpy=12_350.0,
            )

    def fake_authenticated_executor(config: KabuSConfig) -> FakeExecutor:
        captured["config"] = config
        return FakeExecutor()

    monkeypatch.setattr(broker_cli, "_authenticated_executor", fake_authenticated_executor)

    result = CliRunner().invoke(
        app,
        [
            "broker",
            "submit-order",
            "--config",
            str(config_path),
            "--order",
            str(order_path),
            "--confirm",
            "exact-intent-digest",
            "--confirm-production",
            "REAL_ORDERS",
        ],
    )

    assert result.exit_code == 0
    assert captured["intent"] == expected_intent
    assert captured["confirmation"] == "exact-intent-digest"
    assert captured["production_confirmation"] == "REAL_ORDERS"
    output = json.loads(result.stdout)
    assert output == {
        "broker_order_id": "broker-order-001",
        "client_order_id": "client-001",
        "environment": "validation",
        "intent_digest": expected_intent.confirmation_digest(),
        "notional_jpy": 12_350.0,
    }
    assert secret_canary not in result.stdout
    assert "password" not in result.stdout.casefold()
    assert "token" not in result.stdout.casefold()


def test_preview_cancel_forwards_context_identity_and_prints_bound_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    captured: dict[str, object] = {}
    expected = {
        "environment": "validation",
        "production_cancellation_is_real": False,
        "client_action_id": "cancel-action-001",
        "broker_order": {
            "ID": "broker-order-001",
            "Symbol": "9433",
            "State": 2,
        },
        "config_digest": "config-digest",
        "confirmation_digest": "context-bound-cancel-digest",
        "network_mutation_made": False,
    }

    class FakeExecutor:
        def preview_cancel(self, order_id: str, *, client_action_id: str) -> dict[str, Any]:
            captured.update(
                order_id=order_id,
                client_action_id=client_action_id,
            )
            return expected

    def fake_authenticated_executor(config: KabuSConfig) -> FakeExecutor:
        captured["config"] = config
        return FakeExecutor()

    monkeypatch.setattr(broker_cli, "_authenticated_executor", fake_authenticated_executor)

    result = CliRunner().invoke(
        app,
        [
            "broker",
            "preview-cancel",
            "--config",
            str(config_path),
            "--order-id",
            "broker-order-001",
            "--client-action-id",
            "cancel-action-001",
        ],
    )

    assert result.exit_code == 0
    assert captured["order_id"] == "broker-order-001"
    assert captured["client_action_id"] == "cancel-action-001"
    assert isinstance(captured["config"], KabuSConfig)
    assert json.loads(result.stdout) == expected


@pytest.mark.parametrize("invalid_input", ["config", "order"])
def test_content_validation_errors_exit_one(
    tmp_path: Path,
    invalid_input: str,
) -> None:
    if invalid_input == "config":
        config_path = _write_config(tmp_path, changes={"unexpected": True})
        arguments = ["broker", "validate-config", "--config", str(config_path)]
    else:
        config_path = _write_config(tmp_path)
        order_path = _write_order(tmp_path, changes={"quantity": 0})
        arguments = [
            "broker",
            "preview-order",
            "--config",
            str(config_path),
            "--order",
            str(order_path),
        ]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 1
    assert "broker operation failed" in result.stderr


@pytest.mark.parametrize("invalid_input", ["config", "order"])
def test_validation_errors_do_not_echo_accidental_secret_values(
    tmp_path: Path,
    invalid_input: str,
) -> None:
    secret_canary = "must-never-appear-in-cli-validation-output"
    config_path = _write_config(tmp_path)
    if invalid_input == "config":
        config_path = _write_config(tmp_path, changes={"api_password": secret_canary})
        arguments = ["broker", "validate-config", "--config", str(config_path)]
    else:
        order_path = _write_order(tmp_path, changes={"password": secret_canary})
        arguments = [
            "broker",
            "preview-order",
            "--config",
            str(config_path),
            "--order",
            str(order_path),
        ]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 1
    assert "broker operation failed" in result.stderr
    assert secret_canary not in result.stdout
    assert secret_canary not in result.stderr


def test_engage_and_release_kill_switch_are_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    state_paths = _use_test_state(tmp_path, monkeypatch)
    kill_switch_path = state_paths.kill_switch_path
    constructed: list[str] = []

    class OfflineClient:
        def __init__(self, environment: str, *, timeout_seconds: float) -> None:
            del timeout_seconds
            self.environment = environment
            constructed.append(environment)

        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"kill-switch command attempted client operation {name}")

    monkeypatch.setattr(broker_cli, "KabuSSecrets", _forbidden_factory)
    monkeypatch.setattr(broker_cli, "_KabuSClient", OfflineClient)

    engage = CliRunner().invoke(
        app,
        [
            "broker",
            "engage-kill-switch",
            "--config",
            str(config_path),
            "--reason",
            "operator risk-off",
        ],
    )

    assert engage.exit_code == 0
    assert kill_switch_path.is_file()
    engage_output = json.loads(engage.stdout)
    assert engage_output["event_type"] == "broker_kill_switch_engaged"
    marker_digest = engage_output["marker_digest"]

    stale_confirmation = CliRunner().invoke(
        app,
        [
            "broker",
            "release-kill-switch",
            "--config",
            str(config_path),
            "--confirm",
            "RELEASE_KILL_SWITCH",
        ],
    )

    assert stale_confirmation.exit_code == 1
    assert kill_switch_path.is_file()

    release = CliRunner().invoke(
        app,
        [
            "broker",
            "release-kill-switch",
            "--config",
            str(config_path),
            "--confirm",
            f"RELEASE_KILL_SWITCH:{marker_digest}",
        ],
    )

    assert release.exit_code == 0
    assert not kill_switch_path.exists()
    assert json.loads(release.stdout)["event_type"] == ("broker_kill_switch_release_authorized")
    assert constructed == ["validation", "validation", "validation"]


def test_resolve_unknown_rejects_invalid_resolution_before_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(broker_cli, "_authenticated_executor", _forbidden_factory)

    result = CliRunner().invoke(
        app,
        [
            "broker",
            "resolve-unknown",
            "--config",
            str(config_path),
            "--client-order-id",
            "client-001",
            "--resolution",
            "maybe",
            "--confirm",
            "NOT_ACCEPTED:client-001",
        ],
    )

    assert result.exit_code == 1
    assert "resolution must be accepted or not-accepted" in result.stderr
