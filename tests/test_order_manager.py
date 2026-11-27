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
from live.exceptions import BrokerError, InvalidOrder, ProtectionOrderError, SystemHalted


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
    sl = next(p for p in protections if p.type == OrderType.STOP_MARKET)
    tp = next(p for p in protections if p.type == OrderType.LIMIT)
    assert sl.status == tp.status == OrderState.OPEN
    assert sl.side == tp.side == OrderSide.SELL
    assert sl.amount == tp.amount == pytest.approx(order.filled)
    assert sl.stop_price == pytest.approx(order.average_price * 0.98)
    assert tp.price == pytest.approx(order.average_price * 1.06)


def test_kill_switch_blocks_new_orders(manager):
    manager.halt("test halt")
    assert manager.is_halted()

    with pytest.raises(SystemHalted):
        manager.create_entry_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            amount=0.01,
        )

    with pytest.raises(SystemHalted, match="Unvalidated resume is disabled"):
        manager.resume()
    assert manager.is_halted()


def test_halt_persistence_error_still_blocks_orders(manager, monkeypatch):
    def fail_write(key, value):
        raise OSError("database unavailable")

    monkeypatch.setattr(manager.repo, "set_system_state", fail_write)
    with pytest.raises(OSError, match="database unavailable"):
        manager.halt("protection failure")

    assert manager.is_halted()
    with pytest.raises(SystemHalted):
        manager.create_entry_order("BTC/USDT", OrderSide.BUY, 0.01)


def test_persisted_kill_switch_blocks_new_manager(manager, paper_broker, tmp_repo):
    manager.halt("persisted safety halt")
    restarted_manager = OrderManager(broker=paper_broker, repository=tmp_repo)

    with pytest.raises(SystemHalted):
        restarted_manager.create_entry_order("BTC/USDT", OrderSide.BUY, 0.01)

    assert restarted_manager.is_halted()
    assert restarted_manager.get_open_orders("BTC/USDT") == []


@pytest.mark.parametrize("corrupt_value", ["true", "2", ""])
def test_corrupt_persisted_kill_switch_blocks_new_manager(
    paper_broker, tmp_repo, corrupt_value
):
    tmp_repo.set_system_state("kill_switch", corrupt_value)
    restarted_manager = OrderManager(broker=paper_broker, repository=tmp_repo)

    with pytest.raises(SystemHalted):
        restarted_manager.create_entry_order("BTC/USDT", OrderSide.BUY, 0.01)

    assert restarted_manager.is_halted()
    assert restarted_manager.get_open_orders("BTC/USDT") == []


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


def test_invalid_amount_is_rejected_before_submission(manager):
    with pytest.raises(InvalidOrder):
        manager.create_entry_order("BTC/USDT", OrderSide.BUY, float("nan"))


def test_stop_loss_failure_halts_and_skips_take_profit(manager, paper_broker, monkeypatch):
    monkeypatch.setattr(
        "live.order_manager.generate_client_order_id",
        lambda prefix: f"test_{prefix}",
    )
    original_create_order = paper_broker.create_order
    calls = []

    def fail_stop(**kwargs):
        calls.append(kwargs["type"])
        if kwargs["type"] == OrderType.STOP_MARKET:
            raise BrokerError("stop rejected")
        return original_create_order(**kwargs)

    monkeypatch.setattr(paper_broker, "create_order", fail_stop)
    paper_broker.update_price("BTC/USDT", 60_000.0)

    with pytest.raises(ProtectionOrderError):
        manager.create_entry_order("BTC/USDT", OrderSide.BUY, 0.01)

    assert calls == [OrderType.MARKET, OrderType.STOP_MARKET]
    assert manager.is_halted()
    assert manager.repo.get_order("test_entry").status == OrderState.FILLED
    assert manager.repo.get_order("test_sl").status == OrderState.UNKNOWN
    assert manager.repo.get_order("test_tp") is None


def test_take_profit_failure_keeps_stop_and_halts_new_entries(manager, paper_broker, monkeypatch):
    monkeypatch.setattr(
        "live.order_manager.generate_client_order_id",
        lambda prefix: f"test_{prefix}",
    )
    original_create_order = paper_broker.create_order

    def fail_take_profit(**kwargs):
        if kwargs["type"] == OrderType.LIMIT:
            raise BrokerError("take-profit rejected")
        return original_create_order(**kwargs)

    monkeypatch.setattr(paper_broker, "create_order", fail_take_profit)
    paper_broker.update_price("BTC/USDT", 60_000.0)

    with pytest.raises(ProtectionOrderError):
        manager.create_entry_order("BTC/USDT", OrderSide.BUY, 0.01)

    assert manager.is_halted()
    with pytest.raises(SystemHalted):
        manager.create_entry_order("BTC/USDT", OrderSide.BUY, 0.01)
    assert manager.repo.get_order("test_sl").status == OrderState.OPEN
    assert manager.repo.get_order("test_tp").status == OrderState.UNKNOWN
    assert any(o.client_order_id == "test_sl" for o in paper_broker.fetch_open_orders())
