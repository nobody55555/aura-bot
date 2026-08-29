"""
Paper Broker – simulates order lifecycle against live or cached market data.

This broker is the primary development and testing surface.
It produces the same Order / Fill objects as the live CcxtBroker so that
the OrderManager can be fully exercised before any real capital is used.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from live.broker.base import AbstractBroker
from live.models import (
    Order,
    OrderSide,
    OrderType,
    OrderState,
    Fill,
    generate_client_order_id,
)

logger = logging.getLogger(__name__)


class PaperBroker(AbstractBroker):
    """
    Simple but realistic paper trading engine.

    - Market orders fill immediately at last price (or mid).
    - Limit orders fill when price crosses the limit.
    - Stop orders trigger when price crosses stop_price.
    - Maintains its own balance and open order book.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        fee_rate: float = 0.001,          # 0.1% per side
        slippage_bps: float = 2.0,        # 2 bps simulated slippage
        data_source: Optional[AbstractBroker] = None,
    ):
        self.initial_balance = initial_balance
        self.balance = {"USDT": {"free": initial_balance, "total": initial_balance}}
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.data_source = data_source    # optional live data feed for realistic prices

        self._orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._last_prices: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
        client_order_id = client_order_id or generate_client_order_id()
        now = datetime.now(timezone.utc)

        order = Order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            type=type,
            amount=amount,
            price=price,
            stop_price=stop_price,
            status=OrderState.PENDING,
            created_at=now,
            updated_at=now,
            params=params or {},
        )

        # Basic validation
        if amount <= 0:
            order.status = OrderState.REJECTED
            order.error_message = "Amount must be positive"
            self._orders[client_order_id] = order
            return order

        free = self.balance.get("USDT", {}).get("free", 0.0)
        if side == OrderSide.BUY:
            # Rough cost check (we don't know exact fill price yet)
            estimated_cost = amount * (price or self._get_last_price(symbol) or 0)
            if estimated_cost > free * 1.05:  # 5% buffer
                order.status = OrderState.REJECTED
                order.error_message = "Insufficient balance"
                self._orders[client_order_id] = order
                return order

        # Market orders fill immediately
        if type == OrderType.MARKET:
            fill_price = self._apply_slippage(self._get_last_price(symbol), side)
            self._execute_fill(order, fill_price, amount)
            return order

        # Limit / stop orders stay open until triggered
        order.status = OrderState.OPEN
        order.remaining = amount
        self._orders[client_order_id] = order
        logger.info(f"[PAPER] Opened {type.value} order {client_order_id} {side.value} {amount} {symbol}")
        return order

    def cancel_order(self, client_order_id: str, symbol: str) -> Order:
        order = self._orders.get(client_order_id)
        if not order:
            raise ValueError(f"Order {client_order_id} not found")

        if order.status in (OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED):
            return order

        order.status = OrderState.CANCELED
        order.updated_at = datetime.now(timezone.utc)
        order.remaining = 0.0
        logger.info(f"[PAPER] Canceled order {client_order_id}")
        return order

    def fetch_order(self, client_order_id: str, symbol: str) -> Order:
        order = self._orders.get(client_order_id)
        if not order:
            raise ValueError(f"Order {client_order_id} not found")
        # Check if any open limit/stop orders should now fill
        self._check_pending_orders(symbol)
        return order

    def fetch_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        open_states = {OrderState.OPEN, OrderState.PARTIALLY_FILLED, OrderState.SUBMITTING}
        orders = [
            o for o in self._orders.values()
            if o.status in open_states and (symbol is None or o.symbol == symbol)
        ]
        return orders

    def fetch_balance(self) -> dict[str, float]:
        return self.balance

    def fetch_ticker(self, symbol: str) -> dict[str, float]:
        last = self._get_last_price(symbol)
        return {"last": last, "bid": last * 0.9999, "ask": last * 1.0001}

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> list[list]:
        if self.data_source:
            return self.data_source.fetch_ohlcv(symbol, timeframe, limit)
        # Fallback: empty (caller should provide data)
        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_last_price(self, symbol: str) -> float:
        if symbol in self._last_prices:
            return self._last_prices[symbol]
        if self.data_source:
            try:
                ticker = self.data_source.fetch_ticker(symbol)
                price = float(ticker.get("last") or ticker.get("close") or 0)
                self._last_prices[symbol] = price
                return price
            except Exception:
                pass
        # Hard fallback for testing
        return 60_000.0

    def update_price(self, symbol: str, price: float) -> None:
        """Allow external price updates (e.g. from a live feed or backtest)."""
        self._last_prices[symbol] = price
        self._check_pending_orders(symbol)

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        slip = price * (self.slippage_bps / 10_000)
        return price + slip if side == OrderSide.BUY else price - slip

    def _execute_fill(self, order: Order, fill_price: float, fill_amount: float) -> None:
        fee = fill_amount * fill_price * self.fee_rate
        cost = fill_amount * fill_price

        if order.side == OrderSide.BUY:
            self.balance["USDT"]["free"] -= (cost + fee)
            self.balance["USDT"]["total"] -= fee
            # In a real system we would track BTC balance too
        else:
            self.balance["USDT"]["free"] += (cost - fee)
            self.balance["USDT"]["total"] -= fee

        order.filled += fill_amount
        order.remaining = max(0.0, order.amount - order.filled)
        order.average_price = fill_price  # simplified
        order.fee += fee
        order.fee_currency = "USDT"
        order.updated_at = datetime.now(timezone.utc)

        if order.remaining <= 1e-12:
            order.status = OrderState.FILLED
            order.remaining = 0.0
        else:
            order.status = OrderState.PARTIALLY_FILLED

        fill = Fill(
            order_id=order.client_order_id,
            exchange_fill_id=f"paper_{int(time.time()*1000)}",
            price=fill_price,
            amount=fill_amount,
            fee=fee,
            fee_currency="USDT",
            timestamp=datetime.now(timezone.utc),
        )
        self._fills.append(fill)
        self._orders[order.client_order_id] = order
        logger.info(
            f"[PAPER] Filled {fill_amount:.6f} {order.symbol} @ {fill_price:.2f} "
            f"(order {order.client_order_id})"
        )

    def _check_pending_orders(self, symbol: str) -> None:
        """Very simple limit / stop trigger logic."""
        last = self._get_last_price(symbol)
        for order in list(self._orders.values()):
            if order.symbol != symbol:
                continue
            if order.status not in (OrderState.OPEN, OrderState.PARTIALLY_FILLED):
                continue

            should_fill = False
            if order.type == OrderType.LIMIT and order.price is not None:
                if order.side == OrderSide.BUY and last <= order.price:
                    should_fill = True
                elif order.side == OrderSide.SELL and last >= order.price:
                    should_fill = True
            elif order.type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT) and order.stop_price:
                if order.side == OrderSide.SELL and last <= order.stop_price:  # stop loss long
                    should_fill = True
                elif order.side == OrderSide.BUY and last >= order.stop_price:
                    should_fill = True

            if should_fill:
                fill_price = order.price or last
                fill_price = self._apply_slippage(fill_price, order.side)
                remaining = order.remaining or order.amount
                self._execute_fill(order, fill_price, remaining)
