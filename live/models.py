"""
Order and state models for the live trading system.

Uses simple dataclasses + SQLite persistence.
Exchange is always the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid


class OrderState(str, Enum):
    PENDING = "PENDING"               # created locally, not yet submitted
    SUBMITTING = "SUBMITTING"         # request sent, waiting for ack
    OPEN = "OPEN"                     # accepted by exchange
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"               # reconciliation found inconsistency


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


def generate_client_order_id(prefix: str = "aura") -> str:
    """Generate a unique, exchange-compatible client order ID."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass
class Order:
    """Represents a single order throughout its lifecycle."""

    client_order_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    amount: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderState = OrderState.PENDING
    exchange_order_id: Optional[str] = None
    filled: float = 0.0
    remaining: Optional[float] = None
    average_price: Optional[float] = None
    fee: float = 0.0
    fee_currency: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    params: dict[str, Any] = field(default_factory=dict)
    parent_order_id: Optional[str] = None  # for protection orders linked to entry
    strategy_tag: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.remaining is None:
            self.remaining = self.amount - self.filled

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["type"] = self.type.value
        d["status"] = self.status.value
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Order":
        data = data.copy()
        data["side"] = OrderSide(data["side"])
        data["type"] = OrderType(data["type"])
        data["status"] = OrderState(data["status"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class Fill:
    """A single fill / trade execution."""

    order_id: str                    # client_order_id
    exchange_fill_id: Optional[str]
    price: float
    amount: float
    fee: float = 0.0
    fee_currency: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_maker: bool = False


@dataclass
class Position:
    """Current position snapshot."""

    symbol: str
    side: str                        # "long" | "short" | "flat"
    size: float
    entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EquitySnapshot:
    timestamp: datetime
    total_equity: float
    available_balance: float
    unrealized_pnl: float = 0.0
    used_margin: float = 0.0
