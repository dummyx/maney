from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Hashable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, Node

_SYMBOL = re.compile(r"^[A-Z0-9]+$")
_SENSITIVE_KEY_PARTS = frozenset(
    {"password", "token", "secret", "apikey", "authorization", "xapikey"}
)


class _BrokerYamlLoader(yaml.SafeLoader):
    """Safe YAML subset with unambiguous mappings, aliases, and booleans."""

    def compose_node(self, parent: Node | None, index: int) -> Node:
        if self.check_event(yaml.AliasEvent):  # type: ignore[no-untyped-call]
            event = self.peek_event()  # type: ignore[no-untyped-call]
            raise ConstructorError(
                None,
                None,
                "broker YAML aliases are not supported",
                event.start_mark,
            )
        result = super().compose_node(parent, index)
        if result is None:  # pragma: no cover - PyYAML composer contract
            raise ConstructorError(None, None, "broker YAML node is missing", None)
        return result

    def construct_mapping(self, node: Node, deep: bool = False) -> dict[Hashable, Any]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None, "expected a mapping", node.start_mark)
        result: dict[Hashable, Any] = {}
        for key_node, value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise ConstructorError(
                    None,
                    None,
                    "broker YAML merge keys are not supported",
                    key_node.start_mark,
                )
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConstructorError(
                    None,
                    None,
                    "broker YAML mapping keys must be strings",
                    key_node.start_mark,
                )
            if key in result:
                raise ConstructorError(
                    None,
                    None,
                    "broker YAML repeats a mapping key",
                    key_node.start_mark,
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


_BrokerYamlLoader.yaml_implicit_resolvers = {
    first: [(tag, pattern) for tag, pattern in resolvers if tag != "tag:yaml.org,2002:bool"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_BrokerYamlLoader.add_implicit_resolver(  # type: ignore[no-untyped-call]
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$"),
    ["t", "f"],
)


class _FrozenBrokerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class KabuSConfig(_FrozenBrokerModel):
    """Strict non-secret configuration for the separately invoked kabuS adapter."""

    schema_version: Literal["kabus-broker-v1"]
    provider: Literal["kabus"]
    environment: Literal["validation", "production"]
    enabled: bool
    order_submission_enabled: bool
    production_acknowledgement: Literal["REAL_ORDERS"] | None = None
    single_user_private_use: Literal[True]
    account_type: Literal[2, 4, 12]
    cash_buy_deliv_type: Literal[2, 3]
    cash_buy_fund_type: Literal["02", "AA"]
    allowed_symbols: tuple[str, ...] = ()
    allowed_exchanges: tuple[Literal[3, 5, 6, 9, 27], ...] = ()
    allow_market_orders: Literal[False] = False
    max_order_quantity: int = Field(gt=0)
    max_order_notional_jpy: float = Field(gt=0)
    max_daily_order_notional_jpy: float = Field(gt=0)
    max_position_quantity_per_symbol: int = Field(gt=0)
    max_position_notional_jpy_per_symbol: float = Field(gt=0)
    max_gross_cash_position_notional_jpy: float = Field(gt=0)
    max_total_unrealized_loss_jpy: float = Field(gt=0)
    max_open_orders: int = Field(gt=0)
    max_intent_age_seconds: int = Field(gt=0)
    max_future_intent_skew_seconds: int = Field(ge=0)
    max_preflight_duration_seconds: int = Field(gt=0, le=60)
    max_reconciliation_match_seconds: int = Field(gt=0, le=300)
    max_quote_age_seconds: int = Field(gt=0, le=300)
    max_price_deviation_bps: float = Field(ge=0)
    timeout_seconds: float = Field(ge=0.1, le=30.0)

    @field_validator("allowed_symbols", "allowed_exchanges", mode="before")
    @classmethod
    def normalize_yaml_sequences(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_safety_contract(self) -> KabuSConfig:
        if self.order_submission_enabled and not self.enabled:
            raise ValueError("order_submission_enabled requires enabled=true")

        production_submission = self.environment == "production" and self.order_submission_enabled
        if production_submission and self.production_acknowledgement != "REAL_ORDERS":
            raise ValueError(
                "production order submission requires production_acknowledgement=REAL_ORDERS"
            )
        if not production_submission and self.production_acknowledgement is not None:
            raise ValueError(
                "production_acknowledgement is forbidden unless production order submission "
                "is enabled"
            )

        if self.enabled and not self.allowed_symbols:
            raise ValueError("allowed_symbols must not be empty when the broker is enabled")
        if self.enabled and not self.allowed_exchanges:
            raise ValueError("allowed_exchanges must not be empty when the broker is enabled")
        if len(self.allowed_symbols) != len(set(self.allowed_symbols)):
            raise ValueError("allowed_symbols must be unique")
        invalid_symbols = [
            symbol for symbol in self.allowed_symbols if _SYMBOL.fullmatch(symbol) is None
        ]
        if invalid_symbols:
            raise ValueError(
                "allowed_symbols must contain only non-empty uppercase ASCII letters and digits"
            )
        if len(self.allowed_exchanges) != len(set(self.allowed_exchanges)):
            raise ValueError("allowed_exchanges must be unique")

        if self.max_order_notional_jpy > self.max_daily_order_notional_jpy:
            raise ValueError("max_order_notional_jpy cannot exceed max_daily_order_notional_jpy")
        if self.max_order_quantity > self.max_position_quantity_per_symbol:
            raise ValueError("max_order_quantity cannot exceed max_position_quantity_per_symbol")
        if self.max_order_notional_jpy > self.max_position_notional_jpy_per_symbol:
            raise ValueError(
                "max_order_notional_jpy cannot exceed max_position_notional_jpy_per_symbol"
            )
        if self.max_position_notional_jpy_per_symbol > self.max_gross_cash_position_notional_jpy:
            raise ValueError(
                "max_position_notional_jpy_per_symbol cannot exceed "
                "max_gross_cash_position_notional_jpy"
            )
        if self.max_preflight_duration_seconds > self.max_intent_age_seconds:
            raise ValueError("max_preflight_duration_seconds cannot exceed max_intent_age_seconds")
        minimum_match_window = self.max_preflight_duration_seconds + self.timeout_seconds
        if self.max_reconciliation_match_seconds < minimum_match_window:
            raise ValueError(
                "max_reconciliation_match_seconds cannot be less than the combined "
                "preflight and transport timeout"
            )
        return self

    def canonical_json(self) -> str:
        """Return normalized non-secret configuration for approvals and audit."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def digest(self) -> str:
        """Return a stable identity for the complete effective broker configuration."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class KabuSSecrets(BaseSettings):
    """Environment-only kabuS credentials, kept separate from serializable config."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="NLP_TRADER_KABUS_",
        extra="ignore",
        frozen=True,
    )

    api_password: SecretStr = Field(min_length=1)


def load_kabus_config(path: str | Path) -> KabuSConfig:
    """Load broker-only YAML without instantiating secrets or performing network access."""

    config_path = Path(path).expanduser().resolve()
    try:
        loaded = yaml.load(
            config_path.read_text(encoding="utf-8"),
            Loader=_BrokerYamlLoader,
        )
    except FileNotFoundError as exc:
        raise ValueError(f"broker config file does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
        raise ValueError(f"invalid broker YAML in {config_path}{location}") from None
    if not isinstance(loaded, dict):
        raise ValueError(f"broker config root must be a mapping: {config_path}")

    raw: dict[str, Any] = {str(key): value for key, value in loaded.items()}
    if _contains_sensitive_key(raw):
        raise ValueError("broker config must not contain credentials or credential-like key names")
    return KabuSConfig.model_validate(raw)


def _contains_sensitive_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                return True
            if _contains_sensitive_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False
