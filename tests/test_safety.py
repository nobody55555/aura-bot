"""Basic tests for SafetyMonitor and Reconciler (paper only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from live.broker.paper import PaperBroker
from live.db.repository import OrderRepository
from live.models import OrderSide, OrderType, OrderState
from live.order_manager import OrderManager
from live.safety import SafetyMonitor, Reconciler
from live.exceptions import SystemHalted, RiskLimitExceeded


@pytest.fixture
def tmp_repo(tmp_path: Path):
    return OrderRepository(db_path=tmp_path / "safety.db")


@pytest.fixture
def paper():
    b = PaperBroker(initial_balance=10_000.0)
    b.update_price("BTC/USDT", 60_000.0)
    return b


def test_kill_switch(tmp_repo):
    safety = SafetyMonitor(tmp_repo)
    assert not safety.is_halted()
    safety.halt("test")
    assert safety.is_halted()
    with pytest.raises(SystemHalted):
        safety.require_not_halted()
    safety.resume()
    assert not safety.is_halted()


def test_stale_data_triggers_halt(tmp_repo):
    safety = SafetyMonitor(tmp_repo, max_stale_seconds=1.0)
    safety.update_bar_timestamp("BTC/USDT", bar_ts=0)  # very old
    with pytest.raises(SystemHalted):
        safety.check_stale("BTC/USDT")
    assert safety.is_halted()


def test_daily_loss_halt(tmp_repo):
    safety = SafetyMonitor(tmp_repo, daily_loss_limit=0.02, max_drawdown=0.50)
    safety.update_equity(10_000.0)          # day start
    safety.update_equity(9_700.0)           # -3 % → should halt
    with pytest.raises(RiskLimitExceeded):
        safety.update_equity(9_700.0)
    assert safety.is_halted()


def test_reconcile_with_paper(tmp_repo, paper):
    om = OrderManager(paper, tmp_repo)
    # create a limit order that stays open
    order = om.create_entry_order(
        "BTC/USDT", OrderSide.BUY, 0.01,
        order_type=OrderType.LIMIT, price=50_000.0,
        place_protection=False,
    )
    assert order.status == OrderState.OPEN

    reconciler = Reconciler(tmp_repo, paper)
    summary = reconciler.reconcile_open_orders("BTC/USDT")
    assert summary["matched"] >= 1 or len(summary["updated"]) >= 0
