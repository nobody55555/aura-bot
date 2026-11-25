"""Risk limits shared by runtime configuration and RiskManager."""

from datetime import date

import pytest

from config import Settings
from risk import RiskManager


def _settings(**overrides):
    values = {
        "mode": "paper",
        "poll_seconds": 300,
        "risk_per_trade": 0.01,
        "max_position_pct": 0.25,
        "daily_loss_limit": 0.02,
        "max_drawdown": 0.05,
    }
    values.update(overrides)
    return Settings(**values)


def test_settings_accept_plan_limits():
    _settings().validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"risk_per_trade": 0.0101},
        {"daily_loss_limit": 0.0201},
        {"max_drawdown": 0.0501},
        {"daily_loss_limit": float("nan")},
    ],
)
def test_settings_reject_relaxed_or_invalid_risk_limits(overrides):
    with pytest.raises(ValueError):
        _settings(**overrides).validate()


def test_risk_manager_defaults_match_hard_caps():
    risk = RiskManager()
    assert risk.risk_per_trade == pytest.approx(0.01)
    assert risk.daily_loss_limit == pytest.approx(0.02)
    assert risk.max_drawdown == pytest.approx(0.05)


def test_risk_manager_rejects_unsafe_limits():
    with pytest.raises(ValueError):
        RiskManager(risk_per_trade=0.0101)
    with pytest.raises(ValueError):
        RiskManager(daily_loss_limit=0.0201)
    with pytest.raises(ValueError):
        RiskManager(max_drawdown=0.0501)


def test_risk_manager_halts_at_daily_loss_boundary():
    risk = RiskManager()
    today = date(2026, 11, 22)
    risk.update_equity(10_000.0, today)
    assert risk.can_trade()

    risk.update_equity(9_800.0, today)
    assert not risk.can_trade()
    assert risk.state is not None and risk.state.halted


def test_invalid_equity_disables_risk_manager():
    risk = RiskManager()
    risk.update_equity(float("nan"), date(2026, 11, 22))
    assert not risk.can_trade()
    assert risk.state is not None and risk.state.halt_reason == "invalid_equity"


def test_direct_begin_with_infinite_equity_is_halted():
    risk = RiskManager()
    risk.begin(float("inf"), date(2026, 11, 22))
    assert not risk.can_trade()
    assert risk.state is not None and risk.state.halt_reason == "invalid_equity"


def test_position_size_rejects_invalid_market_values():
    risk = RiskManager()
    assert risk.position_size(10_000.0, float("nan"), 59_000.0) == 0.0
    assert risk.position_size(10_000.0, 60_000.0, 0.0) == 0.0
