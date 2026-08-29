"""
Abstract Broker Interface.

Both PaperBroker and CcxtBroker implement this contract.
This allows the OrderManager and strategy layer to be completely
agnostic about whether we are in paper or live mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from live.models import Order, OrderSide, OrderType, Fill


class AbstractBroker(ABC):
    """
    Common interface for paper and live brokers.

    Design principles:
    - All order creation is idempotent via client_order_id
    - Exchange (or paper simulation) is the source of truth
    - Methods raise typed exceptions that the OrderManager can handle
    """

    @abstractmethod
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
        """
        Submit a new order.

        Returns the Order object with status set to OPEN, FILLED,
        REJECTED, etc. depending on the result.
        """
        ...

    @abstractmethod
    def cancel_order(self, client_order_id: str, symbol: str) -> Order:
        """Cancel an existing order by client_order_id."""
        ...

    @abstractmethod
    def fetch_order(self, client_order_id: str, symbol: str) -> Order:
        """Fetch current state of an order from the exchange / paper engine."""
        ...

    @abstractmethod
    def fetch_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Return all currently open orders."""
        ...

    @abstractmethod
    def fetch_balance(self) -> dict[str, float]:
        """
        Return free and total balances.
        Expected shape: {"USDT": {"free": x, "total": y}, ...}
        """
        ...

    @abstractmethod
    def fetch_ticker(self, symbol: str) -> dict[str, float]:
        """Return last, bid, ask, etc."""
        ...

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> list[list]:
        """Return OHLCV candles (same format as ccxt)."""
        ...

    def close(self) -> None:
        """Optional cleanup (connections, websockets, etc.)."""
        pass
