"""SafetyMonitor and Reconciler tests (paper only)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from live.broker.paper import PaperBroker
from live.db.database import get_connection
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
    with pytest.raises(SystemHalted, match="explicit operator approval"):
        safety.resume("unapproved", equity=10_000.0, exchange_required=False)
    safety.resume(
        "approved maintenance",
        equity=10_000.0,
        exchange_required=False,
        operator_approved=True,
    )
    assert not safety.is_halted()


def test_persisted_halt_survives_monitor_restart_without_resume(tmp_repo):
    first = SafetyMonitor(tmp_repo)
    first.halt("operator review required")

    restarted = SafetyMonitor(tmp_repo)
    assert restarted.is_halted()
    assert restarted.get_halt_reason() == "operator review required"
    with pytest.raises(SystemHalted):
        restarted.require_not_halted()
    assert tmp_repo.get_system_state("kill_switch") == "1"


@pytest.mark.parametrize("corrupt_value", ["true", "2", ""])
def test_corrupt_persisted_kill_switch_fails_closed_after_restart_and_denies_resume(
    tmp_repo, corrupt_value
):
    tmp_repo.set_system_state("kill_switch", corrupt_value)

    restarted = SafetyMonitor(tmp_repo)
    assert restarted.is_halted()
    assert "Invalid persisted kill-switch state" in restarted.get_halt_reason()
    with pytest.raises(SystemHalted):
        restarted.require_not_halted()

    with pytest.raises(SystemHalted, match="explicit operator approval"):
        restarted.resume(
            "unapproved recovery",
            equity=10_000.0,
            exchange_required=False,
        )

    assert restarted.is_halted()
    assert tmp_repo.get_system_state("kill_switch") == corrupt_value


@pytest.mark.parametrize(
    "exchange_time",
    [None, float("nan"), float("inf"), float("-inf"), "not-a-time", True, 0, 4102444800001],
)
def test_invalid_exchange_time_persistently_halts(tmp_repo, exchange_time):
    safety = SafetyMonitor(tmp_repo)
    with pytest.raises(SystemHalted):
        safety.check_clock_drift(exchange_time)

    assert safety.is_halted()
    assert tmp_repo.get_system_state("kill_switch") == "1"
    assert "exchange time" in safety.get_halt_reason().lower()


def test_valid_exchange_time_passes_strict_clock_check(tmp_repo):
    safety = SafetyMonitor(tmp_repo)
    safety.check_clock_drift(time.time() * 1000)
    assert not safety.is_halted()


@pytest.mark.parametrize("equity", [None, 0, -1, float("nan"), float("inf"), float("-inf")])
def test_resume_refuses_invalid_equity_and_preserves_halt(tmp_repo, equity):
    safety = SafetyMonitor(tmp_repo)
    safety.halt("manual risk review")

    with pytest.raises(SystemHalted):
        safety.resume(
            "reviewed",
            equity=equity,
            exchange_required=False,
            operator_approved=True,
        )

    assert safety.is_halted()
    assert tmp_repo.get_system_state("kill_switch") == "1"


@pytest.mark.parametrize(
    "exchange_time",
    [None, float("nan"), float("inf"), "bad", 0, 4102444800001],
)
def test_resume_refuses_invalid_exchange_time_and_preserves_halt(tmp_repo, exchange_time):
    safety = SafetyMonitor(tmp_repo)
    safety.halt("manual clock review")

    with pytest.raises(SystemHalted):
        safety.resume(
            "reviewed",
            equity=10_000.0,
            exchange_time_ms=exchange_time,
            exchange_required=True,
            operator_approved=True,
        )

    assert safety.is_halted()
    assert tmp_repo.get_system_state("kill_switch") == "1"


def test_resume_records_halt_reason_and_operator_approval(tmp_repo):
    safety = SafetyMonitor(tmp_repo)
    safety.halt("unexpected clock drift")

    safety.resume(
        "checked logs and restored clock",
        equity=10_000.0,
        exchange_required=False,
        operator_approved=True,
    )

    assert not safety.is_halted()
    conn = get_connection(tmp_repo.db_path)
    try:
        event = conn.execute(
            "SELECT details FROM risk_events WHERE event_type = 'RESUME' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert "unexpected clock drift" in event["details"]
    assert "checked logs and restored clock" in event["details"]


def test_halt_database_failure_remains_locally_fail_closed(tmp_repo, monkeypatch):
    safety = SafetyMonitor(tmp_repo)

    def fail_write(key, value):
        raise OSError("database unavailable")

    monkeypatch.setattr(tmp_repo, "set_system_state", fail_write)
    with pytest.raises(OSError, match="database unavailable"):
        safety.halt("cannot persist halt")

    assert safety.is_halted()
    with pytest.raises(SystemHalted):
        safety.require_not_halted()


def test_stale_data_triggers_halt(tmp_repo):
    safety = SafetyMonitor(tmp_repo, max_stale_seconds=1.0)
    safety.update_bar_timestamp("BTC/USDT", bar_ts=0.0)  # epoch → very old
    with pytest.raises(SystemHalted):
        safety.check_stale("BTC/USDT")
    assert safety.is_halted()


def test_missing_data_timestamp_triggers_halt(tmp_repo):
    safety = SafetyMonitor(tmp_repo)
    with pytest.raises(SystemHalted):
        safety.check_stale("BTC/USDT")
    assert safety.is_halted()


def test_missing_exchange_time_triggers_halt(tmp_repo):
    safety = SafetyMonitor(tmp_repo)
    with pytest.raises(SystemHalted):
        safety.check_clock_drift(None)
    assert safety.is_halted()


def test_daily_loss_halt(tmp_repo):
    safety = SafetyMonitor(tmp_repo, daily_loss_limit=0.02, max_drawdown=0.05)
    safety.update_equity(10_000.0)  # day start
    with pytest.raises(RiskLimitExceeded):
        safety.update_equity(9_700.0)  # -3 % → halt
    assert safety.is_halted()


def test_daily_loss_halts_at_exact_limit(tmp_repo):
    safety = SafetyMonitor(tmp_repo, daily_loss_limit=0.02, max_drawdown=0.05)
    safety.update_equity(10_000.0)
    with pytest.raises(RiskLimitExceeded):
        safety.update_equity(9_800.0)
    assert safety.is_halted()


def test_invalid_equity_triggers_halt(tmp_repo):
    safety = SafetyMonitor(tmp_repo)
    with pytest.raises(RiskLimitExceeded):
        safety.update_equity(float("nan"))
    assert safety.is_halted()


def test_risk_thresholds_cannot_be_relaxed_past_limits(tmp_repo):
    with pytest.raises(ValueError):
        SafetyMonitor(tmp_repo, daily_loss_limit=0.021)
    with pytest.raises(ValueError):
        SafetyMonitor(tmp_repo, max_drawdown=0.051)


def test_invalid_persisted_equity_state_triggers_halt(tmp_repo):
    tmp_repo.set_system_state("peak_equity", "nan")
    tmp_repo.set_system_state("day_start_equity", "10000")
    tmp_repo.set_system_state("current_day", "2026-11-22")

    safety = SafetyMonitor(tmp_repo)
    safety.restore_equity_state()
    assert safety.is_halted()


def test_unknown_remote_order_triggers_halt(tmp_repo, paper):
    remote = paper.create_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        amount=0.01,
        price=50_000.0,
        client_order_id="external_order",
    )
    assert remote.status == OrderState.OPEN

    safety = SafetyMonitor(tmp_repo)
    reconciler = Reconciler(tmp_repo, paper, safety=safety)
    summary = reconciler.reconcile_open_orders("BTC/USDT")

    assert summary["remote_only"] == ["external_order"]
    assert safety.is_halted()


def test_reconcile_with_paper(tmp_repo, paper):
    om = OrderManager(paper, tmp_repo)
    order = om.create_entry_order(
        "BTC/USDT",
        OrderSide.BUY,
        0.01,
        order_type=OrderType.LIMIT,
        price=50_000.0,
        place_protection=False,
    )
    assert order.status == OrderState.OPEN

    reconciler = Reconciler(tmp_repo, paper)
    summary = reconciler.reconcile_open_orders("BTC/USDT")
    assert summary["matched"] == 1
