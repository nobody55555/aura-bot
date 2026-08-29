"""
Live exchange adapter based on CCXT.

Implements the exact same AbstractBroker interface as PaperBroker
so the OrderManager can switch between paper and live without changes.

Key guarantees (from DeepSeek review):
- Precision & min-notional checks before every submit
- Idempotent via clientOrderId
- Retries on transient network / rate-limit errors
- Clean mapping of CCXT exceptions → our typed exceptions
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import ccxt
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from live.broker.base import AbstractBroker
from live.broker.utils import validate_and_round, is_duplicate_client_order_error
from live.exceptions import (
    BrokerError,
    DuplicateOrder,
    InsufficientFunds,
    InvalidOrder,
    NetworkError,
    OrderNotFound,
)
from live.models import (
    Order,
    OrderSide,
    OrderType,
    OrderState,
    generate_client_order_id,
)

logger = logging.getLogger(__name__)


# Transient errors that should be retried
_RETRYABLE = (
    ccxt.NetworkError,
    ccxt.RequestTimeout,
    ccxt.ExchangeNotAvailable,
    ccxt.RateLimitExceeded,
)


class CcxtBroker(AbstractBroker):
    """
    Production broker that talks to a real exchange via CCXT.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: str = "",
        secret: str = "",
        password: Optional[str] = None,
        sandbox: bool = False,
        default_type: str = "spot",          # "spot" | "future"
        options: Optional[dict] = None,
    ):
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unknown exchange: {exchange_id}")

        config: dict[str, Any] = {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": default_type},
        }
        if password:
            config["password"] = password
        if options:
            config["options"].update(options)

        self.exchange: ccxt.Exchange = exchange_class(config)

        if sandbox and hasattr(self.exchange, "set_sandbox_mode"):
            self.exchange.set_sandbox_mode(True)

        # Cache markets on first use
        self._markets_loaded = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True

    def _map_exception(self, exc: Exception) -> Exception:
        """Translate CCXT exceptions into our typed exceptions."""
        if isinstance(exc, ccxt.InsufficientFunds):
            return InsufficientFunds(str(exc))
        if isinstance(exc, (ccxt.InvalidOrder, ccxt.BadRequest, ccxt.BadSymbol)):
            return InvalidOrder(str(exc))
        if isinstance(exc, ccxt.OrderNotFound):
            return OrderNotFound(str(exc))
        if isinstance(exc, _RETRYABLE):
            return NetworkError(str(exc))
        if is_duplicate_client_order_error(exc):
            return DuplicateOrder(str(exc))
        return BrokerError(str(exc))

    def _ccxt_to_order(self, raw: dict, client_order_id: str, symbol: str) -> Order:
        """Convert a raw CCXT order dict into our Order model."""
        status_map = {
            "open": OrderState.OPEN,
            "closed": OrderState.FILLED,
            "canceled": OrderState.CANCELED,
            "cancelled": OrderState.CANCELED,
            "expired": OrderState.EXPIRED,
            "rejected": OrderState.REJECTED,
        }
        raw_status = (raw.get("status") or "open").lower()
        status = status_map.get(raw_status, OrderState.UNKNOWN)

        filled = float(raw.get("filled") or 0)
        amount = float(raw.get("amount") or 0)
        remaining = float(raw.get("remaining") or max(0.0, amount - filled))

        if status == OrderState.OPEN and filled > 0:
            status = OrderState.PARTIALLY_FILLED
        if status == OrderState.FILLED:
            remaining = 0.0

        side = OrderSide(raw.get("side", "buy").lower())
        raw_type = (raw.get("type") or "market").lower().replace(" ", "_")
        try:
            otype = OrderType(raw_type)
        except ValueError:
            otype = OrderType.MARKET

        fee_cost = 0.0
        fee_currency = None
        fee_info = raw.get("fee") or {}
        if isinstance(fee_info, dict):
            fee_cost = float(fee_info.get("cost") or 0)
            fee_currency = fee_info.get("currency")

        return Order(
            client_order_id=client_order_id or raw.get("clientOrderId") or raw.get("id", ""),
            exchange_order_id=str(raw.get("id")) if raw.get("id") else None,
            symbol=symbol or raw.get("symbol", ""),
            side=side,
            type=otype,
            amount=amount,
            price=float(raw["price"]) if raw.get("price") else None,
            stop_price=float(raw["stopPrice"]) if raw.get("stopPrice") else None,
            status=status,
            filled=filled,
            remaining=remaining,
            average_price=float(raw["average"]) if raw.get("average") else None,
            fee=fee_cost,
            fee_currency=fee_currency,
            created_at=datetime.fromtimestamp(raw["timestamp"] / 1000, tz=timezone.utc)
            if raw.get("timestamp")
            else datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            params={},
        )

    # ------------------------------------------------------------------
    # AbstractBroker implementation
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(NetworkError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        type: OrderType,
        amount: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> Order:
        self._ensure_markets()
        client_order_id = client_order_id or generate_client_order_id()

        try:
            amount, price, stop_price = validate_and_round(
                self.exchange, symbol, amount, price, stop_price
            )
        except InvalidOrder:
            raise
        except Exception as e:
            raise InvalidOrder(str(e)) from e

        params = dict(params or {})
        params["clientOrderId"] = client_order_id

        # Map our OrderType to CCXT type string
        ccxt_type = type.value
        if type == OrderType.STOP_MARKET:
            ccxt_type = "market"
            if stop_price is not None:
                params["stopPrice"] = stop_price
                params["stopLossPrice"] = stop_price  # some exchanges
        elif type == OrderType.STOP_LIMIT:
            ccxt_type = "limit"
            if stop_price is not None:
                params["stopPrice"] = stop_price
        elif type in (OrderType.TAKE_PROFIT_MARKET, OrderType.TAKE_PROFIT_LIMIT):
            # Simplified – many exchanges use different param names
            if stop_price is not None:
                params["stopPrice"] = stop_price

        try:
            raw = self.exchange.create_order(
                symbol=symbol,
                type=ccxt_type,
                side=side.value,
                amount=amount,
                price=price,
                params=params,
            )
            order = self._ccxt_to_order(raw, client_order_id, symbol)
            logger.info(
                "Live order submitted %s → %s (exchange id %s)",
                client_order_id, order.status.value, order.exchange_order_id,
            )
            return order

        except Exception as e:
            mapped = self._map_exception(e)
            if isinstance(mapped, DuplicateOrder):
                # Idempotency: try to fetch the existing order
                logger.warning("Duplicate clientOrderId %s – fetching existing", client_order_id)
                try:
                    return self.fetch_order(client_order_id, symbol)
                except Exception:
                    raise mapped from e
            raise mapped from e

    @retry(
        retry=retry_if_exception_type(NetworkError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    def cancel_order(self, client_order_id: str, symbol: str) -> Order:
        try:
            # Prefer cancel by clientOrderId when supported
            raw = self.exchange.cancel_order(
                id=client_order_id,
                symbol=symbol,
                params={"clientOrderId": client_order_id},
            )
            return self._ccxt_to_order(raw, client_order_id, symbol)
        except Exception as e:
            # Some exchanges need the exchange order id; fall back to fetch + cancel
            mapped = self._map_exception(e)
            if isinstance(mapped, OrderNotFound):
                raise
            # Try to resolve exchange id first
            try:
                existing = self.fetch_order(client_order_id, symbol)
                if existing.exchange_order_id:
                    raw = self.exchange.cancel_order(existing.exchange_order_id, symbol)
                    return self._ccxt_to_order(raw, client_order_id, symbol)
            except Exception:
                pass
            raise mapped from e

    @retry(
        retry=retry_if_exception_type(NetworkError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    def fetch_order(self, client_order_id: str, symbol: str) -> Order:
        try:
            # Most exchanges support fetch by clientOrderId via params
            raw = self.exchange.fetch_order(
                id=client_order_id,
                symbol=symbol,
                params={"clientOrderId": client_order_id},
            )
            return self._ccxt_to_order(raw, client_order_id, symbol)
        except Exception as e:
            mapped = self._map_exception(e)
            if isinstance(mapped, OrderNotFound):
                # Last resort: scan open + closed orders (expensive, rare)
                try:
                    for o in self.exchange.fetch_orders(symbol, limit=50):
                        if o.get("clientOrderId") == client_order_id:
                            return self._ccxt_to_order(o, client_order_id, symbol)
                except Exception:
                    pass
            raise mapped from e

    def fetch_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        try:
            raw_list = self.exchange.fetch_open_orders(symbol)
            result = []
            for raw in raw_list:
                cid = raw.get("clientOrderId") or raw.get("id", "")
                result.append(self._ccxt_to_order(raw, cid, raw.get("symbol", symbol or "")))
            return result
        except Exception as e:
            raise self._map_exception(e) from e

    def fetch_balance(self) -> dict[str, float]:
        try:
            raw = self.exchange.fetch_balance()
            # Normalise to the shape PaperBroker also uses
            result = {}
            for currency, info in raw.items():
                if not isinstance(info, dict):
                    continue
                if "free" in info or "total" in info:
                    result[currency] = {
                        "free": float(info.get("free") or 0),
                        "total": float(info.get("total") or 0),
                    }
            return result
        except Exception as e:
            raise self._map_exception(e) from e

    def fetch_ticker(self, symbol: str) -> dict[str, float]:
        try:
            t = self.exchange.fetch_ticker(symbol)
            return {
                "last": float(t.get("last") or 0),
                "bid": float(t.get("bid") or 0),
                "ask": float(t.get("ask") or 0),
                "close": float(t.get("close") or 0),
            }
        except Exception as e:
            raise self._map_exception(e) from e

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> list[list]:
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            raise self._map_exception(e) from e

    def close(self) -> None:
        if hasattr(self.exchange, "close"):
            try:
                self.exchange.close()
            except Exception:
                pass
