from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from nlp_trader.broker.config import KabuSConfig, KabuSSecrets, load_kabus_config


def _payload(tmp_path: Path) -> dict[str, Any]:
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
        "max_order_quantity": 100,
        "max_order_notional_jpy": 500_000,
        "max_daily_order_notional_jpy": 500_000,
        "max_position_quantity_per_symbol": 100,
        "max_position_notional_jpy_per_symbol": 500_000,
        "max_gross_cash_position_notional_jpy": 500_000,
        "max_total_unrealized_loss_jpy": 100_000,
        "max_open_orders": 1,
        "max_intent_age_seconds": 60,
        "max_future_intent_skew_seconds": 5,
        "max_preflight_duration_seconds": 15,
        "max_reconciliation_match_seconds": 30,
        "max_quote_age_seconds": 30,
        "max_price_deviation_bps": 50,
        "timeout_seconds": 5,
    }


def _write_config(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "nested" / "broker.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_bundled_validation_config_is_enabled_conservative_and_secret_free() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_kabus_config(root / "configs" / "kabus.validation.yaml")

    assert config.schema_version == "kabus-broker-v1"
    assert config.provider == "kabus"
    assert config.environment == "validation"
    assert config.enabled is True
    assert config.order_submission_enabled is True
    assert config.production_acknowledgement is None
    assert config.single_user_private_use is True
    assert config.allowed_symbols == ("9433",)
    assert config.allowed_exchanges == (27,)
    assert config.allow_market_orders is False
    assert config.cash_buy_deliv_type == 2
    assert config.cash_buy_fund_type == "02"
    assert config.max_open_orders == 1
    assert "audit_ledger_path" not in KabuSConfig.model_fields
    assert "kill_switch_path" not in KabuSConfig.model_fields
    assert len(config.digest()) == 64
    assert "password" not in str(config.model_dump(mode="json")).casefold()


def test_operational_state_paths_are_not_configurable(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _payload(tmp_path))

    config = load_kabus_config(path)

    assert "audit_ledger_path" not in config.model_dump()
    payload = _payload(tmp_path)
    payload["audit_ledger_path"] = "alternate/audit.jsonl"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_kabus_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "environment": "production",
                "production_acknowledgement": None,
            },
            "requires production_acknowledgement=REAL_ORDERS",
        ),
        (
            {"production_acknowledgement": "REAL_ORDERS"},
            "production_acknowledgement is forbidden",
        ),
        (
            {
                "environment": "production",
                "order_submission_enabled": False,
                "production_acknowledgement": "REAL_ORDERS",
            },
            "production_acknowledgement is forbidden",
        ),
        (
            {"enabled": False, "order_submission_enabled": True},
            "order_submission_enabled requires enabled=true",
        ),
    ],
)
def test_acknowledgement_and_enablement_fail_closed(
    tmp_path: Path,
    changes: dict[str, Any],
    message: str,
) -> None:
    payload = _payload(tmp_path)
    payload.update(changes)

    with pytest.raises(ValidationError, match=message):
        load_kabus_config(_write_config(tmp_path, payload))


def test_production_submission_accepts_exact_acknowledgement(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload.update(
        environment="production",
        production_acknowledgement="REAL_ORDERS",
    )

    config = load_kabus_config(_write_config(tmp_path, payload))

    assert config.production_acknowledgement == "REAL_ORDERS"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(allowed_symbols=[]), "allowed_symbols must not be empty"),
        (lambda value: value.update(allowed_exchanges=[]), "allowed_exchanges must not be empty"),
        (lambda value: value.update(allowed_symbols=["9433", "9433"]), "must be unique"),
        (
            lambda value: value.update(allowed_symbols=["9433-T"]),
            "uppercase ASCII letters and digits",
        ),
        (lambda value: value.update(allowed_symbols=["abcd"]), "uppercase ASCII"),
        (lambda value: value.update(allowed_exchanges=[27, 27]), "must be unique"),
        (lambda value: value.update(allowed_exchanges=[1]), "allowed_exchanges"),
        (lambda value: value.update(allowed_exchanges=[2]), "allowed_exchanges"),
        (lambda value: value.update(allow_market_orders=True), "False"),
        (lambda value: value.update(max_order_quantity=0), "greater than 0"),
        (
            lambda value: value.update(max_future_intent_skew_seconds=-1),
            "greater than or equal to 0",
        ),
        (
            lambda value: value.update(max_order_notional_jpy=600_000),
            "cannot exceed max_daily_order_notional_jpy",
        ),
        (
            lambda value: value.update(max_order_quantity=101),
            "cannot exceed max_position_quantity_per_symbol",
        ),
        (lambda value: value.update(max_price_deviation_bps=-1), "greater than or equal to 0"),
        (lambda value: value.update(timeout_seconds=0.09), "greater than or equal to 0.1"),
        (lambda value: value.update(timeout_seconds=31), "less than or equal to 30"),
        (lambda value: value.update(enabled="yes"), "valid boolean"),
        (lambda value: value.update(order_submission_enabled=1), "valid boolean"),
        (lambda value: value.update(max_order_quantity="100"), "valid integer"),
        (
            lambda value: value.update(max_preflight_duration_seconds=61),
            "less than or equal to 60",
        ),
        (
            lambda value: value.update(max_quote_age_seconds=301),
            "less than or equal to 300",
        ),
        (
            lambda value: value.update(max_reconciliation_match_seconds=4),
            "cannot be less than the combined preflight and transport timeout",
        ),
        (lambda value: value.update(unexpected=True), "Extra inputs are not permitted"),
    ],
)
def test_strict_boundaries_reject_unsafe_values(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    payload = _payload(tmp_path)
    mutation(payload)

    with pytest.raises(ValidationError, match=message):
        load_kabus_config(_write_config(tmp_path, payload))


def test_config_is_frozen_and_secrets_are_environment_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_kabus_config(_write_config(tmp_path, _payload(tmp_path)))
    monkeypatch.setenv("NLP_TRADER_KABUS_API_PASSWORD", "local-secret")

    secrets = KabuSSecrets(_env_file=None)

    assert secrets.api_password.get_secret_value() == "local-secret"
    assert "api_password" not in KabuSConfig.model_fields
    with pytest.raises(ValidationError, match="frozen"):
        config.enabled = False  # type: ignore[misc]


def test_missing_environment_secret_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NLP_TRADER_KABUS_API_PASSWORD", raising=False)

    with pytest.raises(ValidationError, match="api_password"):
        KabuSSecrets(_env_file=None)


def test_config_rejects_credential_keys_without_echoing_the_value(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    secret_canary = "must-never-appear-in-validation-output"
    payload["api_password"] = secret_canary

    with pytest.raises(ValueError) as exc_info:
        load_kabus_config(_write_config(tmp_path, payload))

    assert "credential-like key" in str(exc_info.value)
    assert secret_canary not in str(exc_info.value)


def test_config_rejects_nested_credential_keys_without_echoing_the_value(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    secret_canary = "must-never-appear-from-nested-config"
    payload["unexpected"] = {"Authorization": secret_canary}

    with pytest.raises(ValueError) as exc_info:
        load_kabus_config(_write_config(tmp_path, payload))

    assert "credential-like key" in str(exc_info.value)
    assert secret_canary not in str(exc_info.value)


def test_non_mapping_and_ambiguous_yaml_are_actionable_without_echoing_values(
    tmp_path: Path,
) -> None:
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("kabus\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_kabus_config(scalar)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "order_submission_enabled: false\norder_submission_enabled: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid broker YAML"):
        load_kabus_config(duplicate)

    ambiguous_bool = tmp_path / "ambiguous-bool.yaml"
    payload = yaml.safe_dump(_payload(tmp_path), sort_keys=False).replace(
        "enabled: true",
        "enabled: yes",
    )
    ambiguous_bool.write_text(payload, encoding="utf-8")
    with pytest.raises(ValidationError, match="valid boolean"):
        load_kabus_config(ambiguous_bool)

    malformed = tmp_path / "malformed.yaml"
    secret_canary = "must-never-appear-from-yaml-parser"
    malformed.write_text(f"api_password: [{secret_canary}\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_kabus_config(malformed)
    assert secret_canary not in str(exc_info.value)


@pytest.mark.parametrize(
    "document",
    [
        "enabled: true\nnested:\n  repeated: 1\n  repeated: 2\n",
        "defaults: &defaults\n  enabled: true\ncopy: *defaults\n",
        "<<: {enabled: true}\n",
    ],
)
def test_yaml_duplicate_alias_and_merge_syntax_is_rejected_before_validation(
    tmp_path: Path,
    document: str,
) -> None:
    path = tmp_path / "ambiguous.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid broker YAML"):
        load_kabus_config(path)


@pytest.mark.parametrize("spelling", ["yes", "on", "TRUE", "False"])
def test_yaml_accepts_only_lowercase_unambiguous_boolean_spellings(
    tmp_path: Path,
    spelling: str,
) -> None:
    document = yaml.safe_dump(_payload(tmp_path), sort_keys=False).replace(
        "enabled: true",
        f"enabled: {spelling}",
    )
    path = tmp_path / "ambiguous-boolean.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValidationError, match="valid boolean"):
        load_kabus_config(path)


def test_direct_model_validation_uses_the_same_strict_contract(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    config = KabuSConfig.model_validate(payload)

    assert config.account_type == 4
    assert config.allow_market_orders is False
