"""
Repository layer for persisting orders, fills and system state.

Every state transition is written to the database *before*
the corresponding broker action is considered complete.
This enables crash recovery and reconciliation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from live.db.database import get_connection, init_db, DEFAULT_DB_PATH
from live.models import Order, OrderState, OrderSide, OrderType, Fill

logger = logging.getLogger(__name__)


class OrderRepository:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        init_db(self.db_path)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def save_order(self, order: Order) -> None:
        """Insert or update an order (upsert by client_order_id)."""
        conn = get_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO orders (
                    client_order_id, exchange_order_id, symbol, side, type,
                    amount, price, stop_price, status, filled, remaining,
                    average_price, fee, fee_currency, created_at, updated_at,
                    params, parent_order_id, strategy_tag, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                    exchange_order_id = excluded.exchange_order_id,
                    status            = excluded.status,
                    filled            = excluded.filled,
                    remaining         = excluded.remaining,
                    average_price     = excluded.average_price,
                    fee               = excluded.fee,
                    fee_currency      = excluded.fee_currency,
                    updated_at        = excluded.updated_at,
                    params            = excluded.params,
                    error_message     = excluded.error_message
                """,
                (
                    order.client_order_id,
                    order.exchange_order_id,
                    order.symbol,
                    order.side.value,
                    order.type.value,
                    order.amount,
                    order.price,
                    order.stop_price,
                    order.status.value,
                    order.filled,
                    order.remaining,
                    order.average_price,
                    order.fee,
                    order.fee_currency,
                    order.created_at.isoformat(),
                    order.updated_at.isoformat(),
                    json.dumps(order.params) if order.params else None,
                    order.parent_order_id,
                    order.strategy_tag,
                    order.error_message,
                ),
            )
            conn.commit()
            logger.debug("Saved order %s → %s", order.client_order_id, order.status.value)
        finally:
            conn.close()

    def get_order(self, client_order_id: str) -> Optional[Order]:
        conn = get_connection(self.db_path)
        try:
            row = conn.execute(
                "SELECT * FROM orders WHERE client_order_id = ?",
                (client_order_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_order(row)
        finally:
            conn.close()

    def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        conn = get_connection(self.db_path)
        try:
            if symbol:
                rows = conn.execute(
                    """
                    SELECT * FROM orders
                    WHERE status IN ('PENDING','SUBMITTING','OPEN','PARTIALLY_FILLED')
                      AND symbol = ?
                    ORDER BY created_at
                    """,
                    (symbol,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM orders
                    WHERE status IN ('PENDING','SUBMITTING','OPEN','PARTIALLY_FILLED')
                    ORDER BY created_at
                    """
                ).fetchall()
            return [self._row_to_order(r) for r in rows]
        finally:
            conn.close()

    def get_orders_by_parent(self, parent_order_id: str) -> list[Order]:
        """Return protection orders linked to an entry order."""
        conn = get_connection(self.db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM orders WHERE parent_order_id = ? ORDER BY created_at",
                (parent_order_id,),
            ).fetchall()
            return [self._row_to_order(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    def save_fill(self, fill: Fill) -> None:
        conn = get_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO fills (
                    order_id, exchange_fill_id, price, amount,
                    fee, fee_currency, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill.order_id,
                    fill.exchange_fill_id,
                    fill.price,
                    fill.amount,
                    fill.fee,
                    fill.fee_currency,
                    fill.timestamp.isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_fills(self, order_id: str) -> list[Fill]:
        conn = get_connection(self.db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM fills WHERE order_id = ? ORDER BY timestamp",
                (order_id,),
            ).fetchall()
            return [
                Fill(
                    order_id=r["order_id"],
                    exchange_fill_id=r["exchange_fill_id"],
                    price=r["price"],
                    amount=r["amount"],
                    fee=r["fee"] or 0.0,
                    fee_currency=r["fee_currency"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                )
                for r in rows
            ]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # System state (kill-switch etc.)
    # ------------------------------------------------------------------

    def set_system_state(self, key: str, value: str) -> None:
        conn = get_connection(self.db_path)
        try:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO system_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, now),
            )
            conn.commit()
        finally:
            conn.close()

    def get_system_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        conn = get_connection(self.db_path)
        try:
            row = conn.execute(
                "SELECT value FROM system_state WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default
        finally:
            conn.close()

    def record_risk_event(self, event_type: str, details: str = "") -> None:
        conn = get_connection(self.db_path)
        try:
            conn.execute(
                "INSERT INTO risk_events (timestamp, event_type, details) VALUES (?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), event_type, details),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_order(self, row) -> Order:
        params = {}
        if row["params"]:
            try:
                params = json.loads(row["params"])
            except Exception:
                params = {}

        return Order(
            client_order_id=row["client_order_id"],
            exchange_order_id=row["exchange_order_id"],
            symbol=row["symbol"],
            side=OrderSide(row["side"]),
            type=OrderType(row["type"]),
            amount=row["amount"],
            price=row["price"],
            stop_price=row["stop_price"],
            status=OrderState(row["status"]),
            filled=row["filled"] or 0.0,
            remaining=row["remaining"],
            average_price=row["average_price"],
            fee=row["fee"] or 0.0,
            fee_currency=row["fee_currency"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            params=params,
            parent_order_id=row["parent_order_id"],
            strategy_tag=row["strategy_tag"],
            error_message=row["error_message"],
        )
