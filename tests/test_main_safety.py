"""Runtime safety gates: no signal or order path before verified inputs."""

from __future__ import annotations

import time

import pytest

import main
from config import Settings
from live.broker.ccxt_broker import CcxtBroker
from live.db.database import get_connection
from live.db.repository import OrderRepository
from live.exceptions import SystemHalted
from live.models import OrderSide, OrderType
from live.safety import SafetyMonitor


class FakeAnalyzer:
    def __init__(self, ohlcv=None):
        self.ohlcv = ohlcv if ohlcv is not None else [object()] * 50
        self.signal_calls = 0

    def fetch_ohlcv(self, *args, **kwargs):
        return self.ohlcv

    def signal(self, ohlcv):
        self.signal_calls += 1
        raise AssertionError("signal path must not be reached")


class FakeOrderManager:
    def __init__(self):
        self.order_calls = 0

    def create_entry_order(self, **kwargs):
        self.order_calls += 1
        raise AssertionError("order path must not be reached")


class FakeRisk:
    def update_equity(self, equity, on):
        return None


class FakeBroker:
    def __init__(self, balance=None, exchange=None, error=None):
        self.balance = balance
        self.exchange = exchange
        self.error = error
        self.balance_calls = 0

    def fetch_balance(self):
        self.balance_calls += 1
        if self.error is not None:
            raise self.error
        return self.balance


class FakeExchange:
    def fetch_time(self):
        return time.time() * 1000


@pytest.fixture
def safety(tmp_path):
    return SafetyMonitor(OrderRepository(db_path=tmp_path / "runtime-safety.db"))


@pytest.mark.parametrize(
    "balance",
    [
        {},
        {"USDT": {"free": 12_345.0}},
        {"USDT": {"total": None, "free": 12_345.0}},
        {"USDT": {"total": 0.0, "free": 12_345.0}},
        {"USDT": {"total": -1.0, "free": 12_345.0}},
        {"USDT": {"total": float("nan"), "free": 12_345.0}},
        {"USDT": {"total": float("inf"), "free": 12_345.0}},
    ],
)
def test_unknown_or_invalid_total_equity_halts_without_using_free(safety, balance):
    broker = FakeBroker(balance=balance)

    with pytest.raises(SystemHalted):
        main._checked_total_equity(broker, safety)

    assert safety.is_halted()
    assert safety.repo.get_system_state("kill_switch") == "1"


def test_failed_balance_retrieval_persistently_halts(safety):
    broker = FakeBroker(error=RuntimeError("balance endpoint unavailable"))

    with pytest.raises(SystemHalted, match="Equity unavailable"):
        main._checked_total_equity(broker, safety)

    assert broker.balance_calls == 1
    assert safety.is_halted()
    assert safety.repo.get_system_state("kill_switch") == "1"


def test_valid_total_equity_is_used_instead_of_free(safety):
    broker = FakeBroker(balance={"USDT": {"total": 8_500.0, "free": 12_345.0}})

    assert main._checked_total_equity(broker, safety) == 8_500.0
    assert not safety.is_halted()


def test_ccxt_missing_total_remains_unknown_and_halts(safety):
    class RawBalanceExchange:
        def fetch_balance(self):
            return {"USDT": {"free": 12_345.0, "used": 0.0}}

    broker = object.__new__(CcxtBroker)
    broker.exchange = RawBalanceExchange()

    normalized = broker.fetch_balance()
    assert normalized["USDT"]["total"] is None
    with pytest.raises(SystemHalted, match="Equity unavailable"):
        main._checked_total_equity(broker, safety)
    assert safety.repo.get_system_state("kill_switch") == "1"


def test_missing_live_exchange_attribute_stops_before_signal_or_order(safety):
    analyzer = FakeAnalyzer()
    broker = FakeBroker(balance={"USDT": {"total": 10_000.0}})
    del broker.exchange
    order_manager = FakeOrderManager()
    ctx = {
        "safety": safety,
        "analyzer": analyzer,
        "broker": broker,
        "risk": FakeRisk(),
        "order_manager": order_manager,
    }

    with pytest.raises(SystemHalted, match="Cannot verify exchange clock"):
        main._process_cycle(ctx, mode="live")

    assert analyzer.signal_calls == 0
    assert broker.balance_calls == 1
    assert order_manager.order_calls == 0
    assert safety.repo.get_system_state("kill_switch") == "1"


def test_invalid_live_equity_stops_before_signal_or_order(safety):
    analyzer = FakeAnalyzer()
    broker = FakeBroker(
        balance={"USDT": {"total": None, "free": 10_000.0}},
        exchange=FakeExchange(),
    )
    order_manager = FakeOrderManager()
    ctx = {
        "safety": safety,
        "analyzer": analyzer,
        "broker": broker,
        "risk": FakeRisk(),
        "order_manager": order_manager,
    }

    with pytest.raises(SystemHalted, match="Equity unavailable"):
        main._process_cycle(ctx, mode="live")

    assert analyzer.signal_calls == 0
    assert order_manager.order_calls == 0
    assert safety.repo.get_system_state("kill_switch") == "1"


@pytest.mark.parametrize(
    "exchange_time",
    [None, float("nan"), float("inf"), float("-inf"), "not-a-time", 0, 4102444800001],
)
def test_invalid_exchange_time_stops_before_signal_or_order(safety, exchange_time):
    class StaticExchange:
        def fetch_time(self):
            return exchange_time

    analyzer = FakeAnalyzer()
    broker = FakeBroker(
        balance={"USDT": {"total": 10_000.0}},
        exchange=StaticExchange(),
    )
    order_manager = FakeOrderManager()
    ctx = {
        "safety": safety,
        "analyzer": analyzer,
        "broker": broker,
        "risk": FakeRisk(),
        "order_manager": order_manager,
    }

    with pytest.raises(SystemHalted):
        main._process_cycle(ctx, mode="live")

    assert safety.repo.get_system_state("kill_switch") == "1"
    assert analyzer.signal_calls == 0
    assert order_manager.order_calls == 0


def test_exchange_time_fetch_error_stops_before_signal_or_order(safety):
    class BrokenExchange:
        def fetch_time(self):
            raise RuntimeError("clock endpoint unavailable")

    analyzer = FakeAnalyzer()
    broker = FakeBroker(
        balance={"USDT": {"total": 10_000.0}},
        exchange=BrokenExchange(),
    )
    order_manager = FakeOrderManager()
    ctx = {
        "safety": safety,
        "analyzer": analyzer,
        "broker": broker,
        "risk": FakeRisk(),
        "order_manager": order_manager,
    }

    with pytest.raises(SystemHalted, match="Cannot verify exchange clock"):
        main._process_cycle(ctx, mode="live")

    assert safety.repo.get_system_state("kill_switch") == "1"
    assert analyzer.signal_calls == 0
    assert order_manager.order_calls == 0


def test_halt_persistence_error_cannot_fall_through_to_signal_or_order(
    safety, monkeypatch
):
    analyzer = FakeAnalyzer()
    broker = FakeBroker(balance={"USDT": {"total": 10_000.0}})
    del broker.exchange
    order_manager = FakeOrderManager()

    original_write = safety.repo.set_system_state

    def fail_halt_write(key, value):
        if key == "kill_switch":
            raise OSError("database unavailable")
        original_write(key, value)

    monkeypatch.setattr(safety.repo, "set_system_state", fail_halt_write)
    ctx = {
        "safety": safety,
        "analyzer": analyzer,
        "broker": broker,
        "risk": FakeRisk(),
        "order_manager": order_manager,
    }

    with pytest.raises(SystemHalted):
        main._process_cycle(ctx, mode="live")

    assert safety.is_halted()
    with pytest.raises(SystemHalted):
        safety.require_not_halted()
    assert analyzer.signal_calls == 0
    assert order_manager.order_calls == 0


def test_explicit_operator_resume_requires_clean_preflight_and_logs_approval(
    safety
):
    safety.halt("manual investigation")
    analyzer = FakeAnalyzer()
    broker = FakeBroker(balance={"USDT": {"total": 10_000.0}})
    ctx = {"safety": safety, "analyzer": analyzer, "broker": broker}

    main._operator_resume(
        ctx,
        "reviewed logs and confirmed account",
        {"local_only": [], "remote_only": []},
        None,
        mode="paper",
    )

    assert not safety.is_halted()
    assert safety.repo.get_system_state("kill_switch") == "0"
    conn = get_connection(safety.repo.db_path)
    try:
        event = conn.execute(
            "SELECT details FROM risk_events WHERE event_type = 'RESUME' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert "manual investigation" in event["details"]
    assert "reviewed logs and confirmed account" in event["details"]


def test_operator_resume_with_missing_exchange_keeps_saved_halt(safety):
    safety.halt("clock review required")
    analyzer = FakeAnalyzer()
    broker = FakeBroker(balance={"USDT": {"total": 10_000.0}})
    del broker.exchange
    ctx = {"safety": safety, "analyzer": analyzer, "broker": broker}

    with pytest.raises(SystemHalted):
        main._operator_resume(
            ctx,
            "operator approved after review",
            {"local_only": [], "remote_only": []},
            None,
            mode="live",
        )

    assert safety.is_halted()
    assert safety.repo.get_system_state("kill_switch") == "1"
    assert "exchange" in safety.get_halt_reason().lower()


def test_operator_resume_refuses_unresolved_startup_reconciliation(safety):
    safety.halt("previous halt")
    analyzer = FakeAnalyzer()
    broker = FakeBroker(balance={"USDT": {"total": 10_000.0}})
    ctx = {"safety": safety, "analyzer": analyzer, "broker": broker}

    with pytest.raises(SystemHalted):
        main._operator_resume(
            ctx,
            "operator request",
            {"local_only": [], "remote_only": ["unmatched-order"]},
            None,
            mode="paper",
        )

    assert safety.is_halted()
    assert safety.repo.get_system_state("kill_switch") == "1"
    assert broker.balance_calls == 0


def test_normal_runtime_start_never_clears_persisted_halt(safety, monkeypatch):
    safety.halt("restart requires operator approval")
    analyzer = FakeAnalyzer()
    broker = FakeBroker(balance={"USDT": {"total": 10_000.0}})
    order_manager = FakeOrderManager()

    class Reconciler:
        def reconcile_on_startup(self, symbol):
            return {"local_only": [], "remote_only": [], "updated": [], "matched": 0}

    ctx = {
        "alerts": object(),
        "broker": broker,
        "safety": safety,
        "reconciler": Reconciler(),
        "order_manager": order_manager,
        "analyzer": analyzer,
        "risk": FakeRisk(),
        "OrderSide": None,
        "OrderType": None,
    }
    monkeypatch.setattr(main, "build_components", lambda: ctx)
    monkeypatch.setattr(main, "settings", Settings(mode="paper"))

    def stop_runtime(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(main.time, "sleep", stop_runtime)
    with pytest.raises(KeyboardInterrupt):
        main.run()

    assert safety.repo.get_system_state("kill_switch") == "1"
    assert safety.is_halted()
    assert analyzer.signal_calls == 0
    assert order_manager.order_calls == 0
    assert broker.balance_calls == 0


def test_live_buy_signal_submits_through_order_manager_after_preflight(
    safety, monkeypatch
):
    class Values:
        iloc = [60_000.0]

    class Bars:
        def __len__(self):
            return 50

        def __getitem__(self, key):
            if key == "close":
                return Values()
            raise KeyError(key)

    class BullishAnalyzer:
        signal_calls = 0

        def fetch_ohlcv(self, *args, **kwargs):
            return Bars()

        def signal(self, ohlcv):
            self.signal_calls += 1
            return type(
                "Signal",
                (),
                {"action": "buy", "confidence": 0.9, "regime": "bull", "reasons": ()},
            )()

        def features(self, ohlcv):
            return {"atr_14": Values()}

    class ReadyRisk(FakeRisk):
        def can_trade(self):
            return True

        def position_size(self, equity, entry, stop):
            return 0.001

    class RecordingOrderManager:
        def __init__(self):
            self.calls = []

        def create_entry_order(self, **kwargs):
            self.calls.append(kwargs)
            return type(
                "Order",
                (),
                {
                    "client_order_id": "live-entry-test",
                    "status": type("Status", (), {"value": "filled"})(),
                    "filled": kwargs["amount"],
                },
            )()

    analyzer = BullishAnalyzer()
    broker = FakeBroker(
        balance={"USDT": {"total": 10_000.0}}, exchange=FakeExchange()
    )
    order_manager = RecordingOrderManager()
    ctx = {
        "safety": safety,
        "analyzer": analyzer,
        "broker": broker,
        "risk": ReadyRisk(),
        "order_manager": order_manager,
        "OrderSide": OrderSide,
        "OrderType": OrderType,
    }
    now = time.time()
    period = max(300, main.settings.poll_seconds * 5)
    target = main.settings.poll_seconds + 10
    test_time = now + (target - now % period) % period
    monkeypatch.setattr(main.time, "time", lambda: test_time)

    main._process_cycle(ctx, mode="live")

    assert analyzer.signal_calls == 1
    assert len(order_manager.calls) == 1
    assert order_manager.calls[0] == {
        "symbol": main.settings.symbol,
        "side": OrderSide.BUY,
        "amount": 0.001,
        "order_type": OrderType.MARKET,
        "strategy_tag": "aura-signal",
        "place_protection": True,
    }
