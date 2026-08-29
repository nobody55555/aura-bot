"""
Basic unit tests for OrderManager + PaperBroker.

These tests verify the happy path and a few error cases
without needing a real exchange connection.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from live.broker.paper import PaperBroker
from live.db.repository import OrderRepository
from live.models import OrderSide, OrderType, OrderState
from live.order_manager import OrderManager
from live.exceptions import SystemHalted


@pytest.fixture
def tmp_repo(tmp_path: Path):
    db_path = tmp_path / "test_aura.db"
    return OrderRepository(db_path=db_path)


@pytest.fixture
def paper_broker():
    return PaperBroker(initial_balance=10_000.0, fee_rate=0.001, slippage_bps=1.0)


@pytest.fixture
def manager(paper_broker, tmp_repo):
    return OrderManager(broker=paper_broker, repository=tmp_repo, default_sl_pct=0.02, default_tp_pct=0.06)


def test_market_entry_creates_and_fills(manager, paper_broker):
    """A market buy should fill immediately and place protection orders."""
    paper_broker.update_price("BTC/USDT", 60_000.0)

    order = manager.create_entry_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=0.01,
        order_type=OrderType.MARKET,
        strategy_tag="test",
    )

    assert order.status == OrderState.FILLED
    assert order.filled == pytest.approx(0.01)
    assert order.average_price is not None

    # Protection orders should have been created
    protections = manager.repo.get_orders_by_parent(order.client_order_id)
    assert len(protections) == 2  # SL + TP

    sides = {p.side for p in protections}
    assert OrderSide.SELL in sides


def test_kill_switch_blocks_new_orders(manager):
    manager.halt("test halt")
    assert manager.is_halted()

    with pytest.raises(SystemHalted):
        manager.create_entry_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            amount=0.01,
        )

    manager.resume()
    assert not manager.is_halted()


def test_limit_order_stays_open(manager, paper_broker):
    paper_broker.update_price("BTC/USDT", 60_000.0)

    order = manager.create_entry_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=0.01,
        order_type=OrderType.LIMIT,
        price=59_000.0,  # below market → stays open
        place_protection=False,
    )

    assert order.status == OrderState.OPEN
    open_orders = manager.get_open_orders("BTC/USDT")
    assert any(o.client_order_id == order.client_order_id for o in open_orders)


def test_cancel_open_order(manager, paper_broker):
    paper_broker.update_price("BTC/USDT", 60_000.0)

    order = manager.create_entry_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        amount=0.01,
        order_type=OrderType.LIMIT,
        price=55_000.0,
        place_protection=False,
    )
    assert order.status == OrderState.OPEN

    canceled = manager.cancel_order(order.client_order_id, "BTC/USDT")
    assert canceled.status == OrderState.CANCELED
