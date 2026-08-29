"""
OrderManager – central lifecycle controller for all orders.

Responsibilities:
- Create entry orders and track them through the state machine
- Persist every state transition *before* considering the action done
- Place protection orders (SL / TP) after an entry is filled
- Handle partial fills, cancellations and broker errors
- Provide a clean interface for the strategy / main loop

Design notes (from DeepSeek review):
- Every transition is written to the DB first
- clientOrderId is always used for idempotency
- Protection orders are linked via parent_order_id
- Ready to swap separate SL/TP for OCO later without large refactor
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from live.broker.base import AbstractBroker
from live.db.repository import OrderRepository
from live.exceptions import (
    BrokerError,
    DuplicateOrder,
    InsufficientFunds,
    InvalidOrder,
    OrderNotFound,
    SystemHalted,
)
from live.models import (
    Order,
    OrderSide,
    OrderType,
    OrderState,
    Fill,
    generate_client_order_id,
)

logger = logging.getLogger(__name__)


class OrderManager:
    def __init__(
        self,
        broker: AbstractBroker,
        repository: Optional[OrderRepository] = None,
        default_sl_pct: float = 0.02,          # 2 % stop-loss
        default_tp_pct: float = 0.06,          # 6 % take-profit (≈ 3:1 RR)
    ):
        self.broker = broker
        self.repo = repository or OrderRepository()
        self.default_sl_pct = default_sl_pct
        self.default_tp_pct = default_tp_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_halted(self) -> bool:
        return self.repo.get_system_state("kill_switch", "0") == "1"

    def halt(self, reason: str = "manual") -> None:
        self.repo.set_system_state("kill_switch", "1")
        self.repo.record_risk_event("HALT", reason)
        logger.warning("System HALTED – reason: %s", reason)

    def resume(self) -> None:
        self.repo.set_system_state("kill_switch", "0")
        self.repo.record_risk_event("RESUME", "manual resume")
        logger.info("System RESUMED")

    def create_entry_order(
        self,
        symbol: str,
        side: OrderSide,
        amount: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        strategy_tag: Optional[str] = None,
        place_protection: bool = True,
    ) -> Order:
        """
        Create and submit an entry order.
        If place_protection=True and the order fills, SL + TP are placed automatically.
        """
        if self.is_halted():
            raise SystemHalted("Kill-switch is active – no new orders allowed")

        client_id = generate_client_order_id("entry")
        order = Order(
            client_order_id=client_id,
            symbol=symbol,
            side=side,
            type=order_type,
            amount=amount,
            price=price,
            status=OrderState.PENDING,
            strategy_tag=strategy_tag,
        )

        # 1. Persist PENDING state first
        self.repo.save_order(order)
        logger.info("Order %s created (PENDING) %s %.6f %s", client_id, side.value, amount, symbol)

        # 2. Submit to broker
        try:
            order.status = OrderState.SUBMITTING
            order.updated_at = datetime.now(timezone.utc)
            self.repo.save_order(order)

            submitted = self.broker.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                amount=amount,
                price=price,
                client_order_id=client_id,
            )

            # Merge broker response into our order object
            order.status = submitted.status
            order.exchange_order_id = submitted.exchange_order_id
            order.filled = submitted.filled
            order.remaining = submitted.remaining
            order.average_price = submitted.average_price
            order.fee = submitted.fee
            order.fee_currency = submitted.fee_currency
            order.error_message = submitted.error_message
            order.updated_at = datetime.now(timezone.utc)

            self.repo.save_order(order)
            logger.info(
                "Order %s → %s (filled=%.6f)",
                client_id, order.status.value, order.filled,
            )

            # 3. If fully (or partially) filled → place protection
            if place_protection and order.status in (OrderState.FILLED, OrderState.PARTIALLY_FILLED):
                self._place_protection_orders(order)

            return order

        except DuplicateOrder:
            # Idempotency: order already exists on the broker
            logger.warning("Duplicate clientOrderId %s – fetching existing order", client_id)
            existing = self.broker.fetch_order(client_id, symbol)
            self.repo.save_order(existing)
            return existing

        except (InsufficientFunds, InvalidOrder) as e:
            order.status = OrderState.REJECTED
            order.error_message = str(e)
            order.updated_at = datetime.now(timezone.utc)
            self.repo.save_order(order)
            logger.error("Order %s REJECTED: %s", client_id, e)
            raise

        except BrokerError as e:
            order.status = OrderState.UNKNOWN
            order.error_message = str(e)
            order.updated_at = datetime.now(timezone.utc)
            self.repo.save_order(order)
            logger.error("Order %s broker error: %s", client_id, e)
            raise

    def cancel_order(self, client_order_id: str, symbol: str) -> Order:
        order = self.repo.get_order(client_order_id)
        if not order:
            raise OrderNotFound(f"Local order {client_order_id} not found")

        if order.status in (OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED):
            return order

        try:
            canceled = self.broker.cancel_order(client_order_id, symbol)
            order.status = canceled.status
            order.updated_at = datetime.now(timezone.utc)
            self.repo.save_order(order)
            logger.info("Order %s canceled", client_order_id)
            return order
        except BrokerError as e:
            logger.error("Failed to cancel %s: %s", client_order_id, e)
            raise

    def sync_order(self, client_order_id: str, symbol: str) -> Order:
        """
        Fetch current state from broker and update local DB.
        Useful for reconciliation and after restarts.
        """
        try:
            remote = self.broker.fetch_order(client_order_id, symbol)
            self.repo.save_order(remote)
            return remote
        except OrderNotFound:
            local = self.repo.get_order(client_order_id)
            if local and local.status not in (OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED):
                local.status = OrderState.UNKNOWN
                local.updated_at = datetime.now(timezone.utc)
                self.repo.save_order(local)
            raise

    def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        return self.repo.get_open_orders(symbol)

    # ------------------------------------------------------------------
    # Protection orders (SL / TP)
    # ------------------------------------------------------------------

    def _place_protection_orders(self, entry: Order) -> None:
        """
        After an entry is filled, place stop-loss and take-profit.
        Currently uses separate orders. Designed so that OCO can be
        swapped in later without changing the call sites.
        """
        if entry.average_price is None or entry.filled <= 0:
            logger.warning("Cannot place protection for %s – no fill price", entry.client_order_id)
            return

        entry_price = entry.average_price
        amount = entry.filled

        # Determine SL / TP prices
        if entry.side == OrderSide.BUY:  # long
            sl_price = entry_price * (1 - self.default_sl_pct)
            tp_price = entry_price * (1 + self.default_tp_pct)
            sl_side = OrderSide.SELL
            tp_side = OrderSide.SELL
        else:  # short
            sl_price = entry_price * (1 + self.default_sl_pct)
            tp_price = entry_price * (1 - self.default_tp_pct)
            sl_side = OrderSide.BUY
            tp_side = OrderSide.BUY

        # Stop-loss (stop-market for simplicity)
        try:
            sl = self._create_protection_order(
                parent=entry,
                side=sl_side,
                order_type=OrderType.STOP_MARKET,
                amount=amount,
                stop_price=sl_price,
                tag="sl",
            )
            logger.info("SL placed for %s → %s @ stop %.2f", entry.client_order_id, sl.client_order_id, sl_price)
        except Exception as e:
            logger.error("Failed to place SL for %s: %s", entry.client_order_id, e)
            self.repo.record_risk_event("PROTECTION_FAILED", f"SL for {entry.client_order_id}: {e}")

        # Take-profit (limit)
        try:
            tp = self._create_protection_order(
                parent=entry,
                side=tp_side,
                order_type=OrderType.LIMIT,
                amount=amount,
                price=tp_price,
                tag="tp",
            )
            logger.info("TP placed for %s → %s @ %.2f", entry.client_order_id, tp.client_order_id, tp_price)
        except Exception as e:
            logger.error("Failed to place TP for %s: %s", entry.client_order_id, e)
            self.repo.record_risk_event("PROTECTION_FAILED", f"TP for {entry.client_order_id}: {e}")

    def _create_protection_order(
        self,
        parent: Order,
        side: OrderSide,
        order_type: OrderType,
        amount: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        tag: str = "prot",
    ) -> Order:
        client_id = generate_client_order_id(tag)
        order = Order(
            client_order_id=client_id,
            symbol=parent.symbol,
            side=side,
            type=order_type,
            amount=amount,
            price=price,
            stop_price=stop_price,
            status=OrderState.PENDING,
            parent_order_id=parent.client_order_id,
            strategy_tag=parent.strategy_tag,
        )
        self.repo.save_order(order)

        order.status = OrderState.SUBMITTING
        self.repo.save_order(order)

        submitted = self.broker.create_order(
            symbol=parent.symbol,
            side=side,
            type=order_type,
            amount=amount,
            price=price,
            stop_price=stop_price,
            client_order_id=client_id,
        )

        order.status = submitted.status
        order.exchange_order_id = submitted.exchange_order_id
        order.filled = submitted.filled
        order.remaining = submitted.remaining
        order.average_price = submitted.average_price
        order.error_message = submitted.error_message
        order.updated_at = datetime.now(timezone.utc)
        self.repo.save_order(order)
        return order
