"""Read-only Hyperliquid Testnet client for HIP-4 outcome market data.

This module deliberately has no signing, account, order, cancel, transfer, or
HyperEVM methods. The REST and WebSocket endpoints are pinned to Testnet. The
REST client only posts explicitly allowed, read-only Info requests.
"""

from __future__ import annotations

import json
import math
import os
import re
from decimal import Decimal, InvalidOperation
from http.client import HTTPException, HTTPSConnection
from typing import Any
from urllib.parse import urlsplit

from hyperliquid.utils import constants as hyperliquid_constants

_EXPECTED_TESTNET_REST_URL = "https://api.hyperliquid-testnet.xyz"
TESTNET_REST_URL = hyperliquid_constants.TESTNET_API_URL
TESTNET_WEBSOCKET_URL = "wss://api.hyperliquid-testnet.xyz/ws"
_INFO_PATH = "/info"
_ALLOWED_INFO_TYPES = frozenset({"outcomeMeta", "spotMetaAndAssetCtxs", "l2Book"})
_OUTCOME_PRICE_FIELDS = (
    "markPx",
    "midPx",
    "prevDayPx",
    "dayNtlVlm",
    "dayBaseVlm",
)
_BINARY_PRICE_FIELDS = frozenset({"markPx", "midPx", "prevDayPx"})
_VOLUME_FIELDS = frozenset({"dayNtlVlm", "dayBaseVlm"})
_MAX_INFO_COIN_LENGTH = 128
_INFO_COIN_RE = re.compile(r"[A-Za-z0-9_#@./:-]{1,128}\Z", re.ASCII)
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024


class TestnetOnlyError(ValueError):
    """Raised when configuration would leave the pinned Testnet boundary."""


class HyperliquidTestnetError(RuntimeError):
    """Raised for transport or response errors from the Testnet Info API."""


def _validate_outcome_context_value(field: str, value: Any) -> Any:
    """Validate an optional numeric HIP-4 price or volume without coercing it."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise HyperliquidTestnetError(
            f"Outcome context field {field} must be a finite number"
        )
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise HyperliquidTestnetError(
            f"Outcome context field {field} must be a finite number"
        ) from exc
    if not numeric.is_finite():
        raise HyperliquidTestnetError(
            f"Outcome context field {field} must be a finite number"
        )
    if field in _BINARY_PRICE_FIELDS and not Decimal(0) <= numeric <= Decimal(1):
        raise HyperliquidTestnetError(
            f"Outcome context price field {field} must be between 0 and 1"
        )
    if field in _VOLUME_FIELDS and numeric < 0:
        raise HyperliquidTestnetError(
            f"Outcome context volume field {field} must be nonnegative"
        )
    return value


def _validate_rest_url(url: str) -> None:
    """Require the exact official Testnet REST base URL, with no overrides."""
    sdk_url = hyperliquid_constants.TESTNET_API_URL
    if (
        sdk_url != _EXPECTED_TESTNET_REST_URL
        or url != sdk_url
        or url != _EXPECTED_TESTNET_REST_URL
    ):
        raise TestnetOnlyError(
            f"Hyperliquid Testnet REST URL must be exactly {_EXPECTED_TESTNET_REST_URL!r}"
        )
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.netloc != "api.hyperliquid-testnet.xyz"
        or parts.path
        or parts.query
        or parts.fragment
    ):
        raise TestnetOnlyError("Invalid Hyperliquid Testnet REST endpoint")


def _validate_websocket_url(url: str) -> None:
    """Require the exact official Testnet WebSocket URL."""
    if url != TESTNET_WEBSOCKET_URL:
        raise TestnetOnlyError(
            f"Hyperliquid WebSocket URL must be exactly {TESTNET_WEBSOCKET_URL!r}"
        )
    parts = urlsplit(url)
    if (
        parts.scheme != "wss"
        or parts.netloc != "api.hyperliquid-testnet.xyz"
        or parts.path != "/ws"
        or parts.query
        or parts.fragment
    ):
        raise TestnetOnlyError("Invalid Hyperliquid Testnet WebSocket endpoint")


def _validate_environment() -> None:
    """Treat HL_ENV as an additional restriction, never as a URL selector."""
    environment = os.environ.get("HL_ENV")
    if environment != "testnet":
        raise TestnetOnlyError(
            "HL_ENV must be explicitly set to exactly 'testnet'; Mainnet is unsupported"
        )


def _validate_info_payload(payload: dict[str, Any]) -> None:
    """Allow only the exact request schema for each approved read-only query."""
    if not isinstance(payload, dict):
        raise TestnetOnlyError("Info request payload must be a JSON object")
    request_type = payload.get("type")
    if not isinstance(request_type, str) or request_type not in _ALLOWED_INFO_TYPES:
        raise TestnetOnlyError(
            "Only outcomeMeta, spotMetaAndAssetCtxs, and l2Book Info requests are allowed"
        )

    if request_type in {"outcomeMeta", "spotMetaAndAssetCtxs"}:
        if set(payload) != {"type"}:
            raise TestnetOnlyError(
                f"{request_type} payload must contain exactly the 'type' field"
            )
        return

    if set(payload) != {"type", "coin"}:
        raise TestnetOnlyError(
            "l2Book payload must contain exactly the 'type' and 'coin' fields"
        )
    coin = payload["coin"]
    if (
        not isinstance(coin, str)
        or len(coin) > _MAX_INFO_COIN_LENGTH
        or _INFO_COIN_RE.fullmatch(coin) is None
    ):
        raise TestnetOnlyError(
            "l2Book coin must be a non-empty symbol of at most 128 allowed characters"
        )


def _post_info_json(url: str, payload: dict[str, Any], timeout: float) -> Any:
    """POST JSON directly to the fixed Testnet host; never follow redirects."""
    expected_url = f"{TESTNET_REST_URL}{_INFO_PATH}"
    if url != expected_url:
        raise TestnetOnlyError(f"Info requests are restricted to {expected_url!r}")
    _validate_rest_url(TESTNET_REST_URL)
    _validate_info_payload(payload)

    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    host = urlsplit(url).hostname
    if host != "api.hyperliquid-testnet.xyz":
        raise TestnetOnlyError("Invalid Hyperliquid Testnet host")
    connection = HTTPSConnection(host, 443, timeout=timeout)
    try:
        connection.request(
            "POST",
            _INFO_PATH,
            body=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        response = connection.getresponse()
        if 300 <= response.status < 400:
            raise HyperliquidTestnetError(
                "Redirect rejected; endpoint must remain on Testnet"
            )
        if response.status != 200:
            raise HyperliquidTestnetError(
                f"Hyperliquid Testnet Info API returned HTTP {response.status}"
            )
        body = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(body) > _MAX_RESPONSE_BYTES:
            raise HyperliquidTestnetError(
                "Hyperliquid Testnet Info response exceeded the size limit"
            )
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HyperliquidTestnetError("Info API returned invalid JSON") from exc
    except HyperliquidTestnetError:
        raise
    except (OSError, HTTPException) as exc:
        raise HyperliquidTestnetError(
            "Could not reach the Hyperliquid Testnet Info API"
        ) from exc
    finally:
        connection.close()


def _outcome_coin(outcome_id: int, side: int) -> str:
    """Return the documented HIP-4 API coin symbol for an outcome side."""
    if (
        not isinstance(outcome_id, int)
        or isinstance(outcome_id, bool)
        or outcome_id < 0
    ):
        raise ValueError("outcome_id must be a non-negative integer")
    if not isinstance(side, int) or isinstance(side, bool) or side not in (0, 1):
        raise ValueError("side must be 0 or 1")
    return f"#{10 * outcome_id + side}"


def _validated_side_specs(outcome: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the binary side labels needed for stable index-based mapping."""
    side_specs = outcome.get("sideSpecs")
    if (
        not isinstance(side_specs, list)
        or len(side_specs) != 2
        or any(
            not isinstance(spec, dict) or not isinstance(spec.get("name"), str)
            for spec in side_specs
        )
    ):
        raise HyperliquidTestnetError("Outcome metadata has malformed binary sideSpecs")
    return side_specs


def _validate_l2_book_response(response: Any, expected_coin: str) -> dict[str, Any]:
    """Reject malformed or mismatched order-book snapshots from the API."""
    if not isinstance(response, dict) or response.get("coin") != expected_coin:
        raise HyperliquidTestnetError("Info API returned a malformed l2Book response")
    timestamp = response.get("time")
    levels = response.get("levels")
    if (
        not isinstance(timestamp, int)
        or isinstance(timestamp, bool)
        or not isinstance(levels, list)
        or len(levels) != 2
        or any(not isinstance(book_side, list) for book_side in levels)
    ):
        raise HyperliquidTestnetError("Info API returned a malformed l2Book response")
    for book_side in levels:
        for level in book_side:
            if (
                not isinstance(level, dict)
                or not isinstance(level.get("px"), str)
                or not isinstance(level.get("sz"), str)
                or not isinstance(level.get("n"), int)
                or isinstance(level.get("n"), bool)
            ):
                raise HyperliquidTestnetError(
                    "Info API returned a malformed l2Book level"
                )
    return response


class HyperliquidTestnetClient:
    """Read-only Testnet client for HIP-4 outcome metadata and market data."""

    def __init__(
        self,
        *,
        rest_url: str = TESTNET_REST_URL,
        websocket_url: str = TESTNET_WEBSOCKET_URL,
        timeout: float = 10.0,
    ) -> None:
        self._rest_url = rest_url
        self._websocket_url = websocket_url
        self._timeout = float(timeout)
        self._validate_configuration()
        if not math.isfinite(self._timeout) or self._timeout <= 0:
            raise ValueError("timeout must be a finite positive number")

    def _validate_configuration(self) -> None:
        _validate_environment()
        _validate_rest_url(self._rest_url)
        _validate_websocket_url(self._websocket_url)

    def get_outcome_metadata(self) -> list[dict[str, Any]]:
        """Fetch all outcome metadata using the documented read-only Info API."""
        self._validate_configuration()
        payload = {"type": "outcomeMeta"}
        response = _post_info_json(
            f"{self._rest_url}{_INFO_PATH}", payload, self._timeout
        )
        if not isinstance(response, dict) or not isinstance(
            response.get("outcomes"), list
        ):
            raise HyperliquidTestnetError("Info API response has no outcomes list")
        if any(not isinstance(outcome, dict) for outcome in response["outcomes"]):
            raise HyperliquidTestnetError(
                "Info API returned a malformed outcome record"
            )
        return response["outcomes"]

    def get_outcome_prices(self) -> list[dict[str, Any]]:
        """Join HIP-4 side metadata to price contexts by the documented coin ID.

        Outcome IDs are encoded as ``#<10 * outcome + side>``. The
        ``spotMetaAndAssetCtxs`` response includes outcome contexts in its
        asset-context array, but they are not ordinary entries in the spot
        metadata universe, so contexts are matched by their ``coin`` field.
        """
        outcomes = self.get_outcome_metadata()
        self._validate_configuration()
        response = _post_info_json(
            f"{self._rest_url}{_INFO_PATH}",
            {"type": "spotMetaAndAssetCtxs"},
            self._timeout,
        )
        if (
            not isinstance(response, list)
            or len(response) != 2
            or not isinstance(response[0], dict)
            or not isinstance(response[0].get("tokens"), list)
            or not isinstance(response[0].get("universe"), list)
            or not isinstance(response[1], list)
            or any(not isinstance(context, dict) for context in response[1])
        ):
            raise HyperliquidTestnetError(
                "Info API returned malformed spot metadata or asset contexts"
            )

        outcome_ids: set[int] = set()
        expected_coins: set[str] = set()
        for outcome in outcomes:
            outcome_id = outcome.get("outcome")
            if (
                not isinstance(outcome_id, int)
                or isinstance(outcome_id, bool)
                or outcome_id < 0
                or outcome_id in outcome_ids
            ):
                raise HyperliquidTestnetError(
                    "Outcome metadata has an invalid or duplicate outcome ID"
                )
            outcome_ids.add(outcome_id)
            side_specs = _validated_side_specs(outcome)
            expected_coins.update(
                _outcome_coin(outcome_id, side) for side in range(len(side_specs))
            )

        contexts_by_coin: dict[str, dict[str, Any]] = {}
        for context in response[1]:
            coin = context.get("coin")
            if isinstance(coin, str) and coin in expected_coins:
                if coin in contexts_by_coin:
                    raise HyperliquidTestnetError(
                        "Info API returned duplicate outcome asset contexts"
                    )
                contexts_by_coin[coin] = context

        prices: list[dict[str, Any]] = []
        for outcome in outcomes:
            outcome_id = outcome["outcome"]
            side_specs = _validated_side_specs(outcome)
            for side, side_spec in enumerate(side_specs):
                coin = _outcome_coin(outcome_id, side)
                context = contexts_by_coin.get(coin)
                price_record: dict[str, Any] = {
                    "outcome": outcome_id,
                    "name": outcome.get("name"),
                    "description": outcome.get("description"),
                    "side": side,
                    "side_name": side_spec["name"],
                    "coin": coin,
                    "context_available": context is not None,
                }
                for field in _OUTCOME_PRICE_FIELDS:
                    value = context.get(field) if context else None
                    price_record[field] = _validate_outcome_context_value(field, value)
                prices.append(price_record)
        return prices

    def get_outcome_book(self, outcome_id: int, side: int) -> dict[str, Any]:
        """Fetch one HIP-4 side's L2 book using its documented ``#encoding``."""
        coin = _outcome_coin(outcome_id, side)
        outcomes = self.get_outcome_metadata()
        matching = [
            outcome for outcome in outcomes if outcome.get("outcome") == outcome_id
        ]
        if len(matching) != 1:
            raise HyperliquidTestnetError(
                "outcome_id does not identify one listed Testnet outcome"
            )
        _validated_side_specs(matching[0])

        self._validate_configuration()
        response = _post_info_json(
            f"{self._rest_url}{_INFO_PATH}",
            {"type": "l2Book", "coin": coin},
            self._timeout,
        )
        return _validate_l2_book_response(response, coin)

    def find_outcomes(
        self,
        *,
        underlying: str | None = None,
        text: str | None = None,
    ) -> list[dict[str, Any]]:
        """Filter the read-only outcome metadata locally, without extra requests."""
        if underlying is not None and not underlying.strip():
            raise ValueError("underlying must be non-empty when provided")
        if text is not None and not text.strip():
            raise ValueError("text must be non-empty when provided")

        outcomes = self.get_outcome_metadata()
        filtered: list[dict[str, Any]] = []
        for outcome in outcomes:
            description = str(outcome.get("description", ""))
            name = str(outcome.get("name", ""))
            if underlying is not None:
                attributes = {}
                for part in description.split("|"):
                    if ":" in part:
                        key, value = part.split(":", 1)
                        attributes[key.strip().casefold()] = value.strip()
                if (
                    attributes.get("underlying", "").casefold()
                    != underlying.strip().casefold()
                ):
                    continue
            if (
                text is not None
                and text.strip().casefold() not in f"{name} {description}".casefold()
            ):
                continue
            filtered.append(outcome)
        return filtered
