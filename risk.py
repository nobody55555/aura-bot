"""Risk controls shared by backtest, paper, and live execution paths."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date


@dataclass
class RiskState:
    equity: float
    peak_equity: float
    day_start_equity: float
    trading_day: date
    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    def __init__(
        self,
        risk_per_trade: float = 0.01,
        max_position_pct: float = 0.25,
        daily_loss_limit: float = 0.02,
        max_drawdown: float = 0.05,
        trailing_atr: float = 2.0,
    ):
        values = (risk_per_trade, max_position_pct, daily_loss_limit, max_drawdown, trailing_atr)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("risk settings must be finite")
        if not 0 < risk_per_trade <= 0.01:
            raise ValueError("risk_per_trade must be in (0, 0.01]")
        if not 0 < max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")
        if not 0 < daily_loss_limit <= 0.02:
            raise ValueError("daily_loss_limit must be in (0, 0.02]")
        if not 0 < max_drawdown <= 0.05:
            raise ValueError("max_drawdown must be in (0, 0.05]")
        if trailing_atr <= 0:
            raise ValueError("trailing_atr must be positive")
        self.risk_per_trade, self.max_position_pct = risk_per_trade, max_position_pct
        self.daily_loss_limit, self.max_drawdown, self.trailing_atr = (
            daily_loss_limit,
            max_drawdown,
            trailing_atr,
        )
        self.state: RiskState | None = None

    def begin(self, equity: float, on: date) -> None:
        valid_equity = math.isfinite(equity) and equity > 0
        self.state = RiskState(
            equity,
            equity,
            equity,
            on,
            halted=not valid_equity,
            halt_reason="" if valid_equity else "invalid_equity",
        )

    def update_equity(self, equity: float, on: date) -> None:
        if not math.isfinite(equity) or equity <= 0:
            if self.state is None:
                self.state = RiskState(equity, equity, equity, on, True, "invalid_equity")
            else:
                self.halt("invalid_equity")
            return
        if self.state is None:
            self.begin(equity, on)
            return
        if on != self.state.trading_day:
            self.state.trading_day, self.state.day_start_equity = on, equity
        self.state.equity, self.state.peak_equity = (
            equity,
            max(self.state.peak_equity, equity),
        )
        if equity <= self.state.day_start_equity * (1 - self.daily_loss_limit):
            self.halt("daily_loss_limit")
        if equity <= self.state.peak_equity * (1 - self.max_drawdown):
            self.halt("max_drawdown")

    def halt(self, reason: str) -> None:
        if self.state:
            self.state.halted, self.state.halt_reason = True, reason

    def can_trade(self) -> bool:
        return (
            self.state is not None and not self.state.halted and self.state.equity > 0
        )

    def position_size(self, equity: float, entry: float, stop: float) -> float:
        if not all(math.isfinite(value) for value in (equity, entry, stop)):
            return 0.0
        if equity <= 0 or entry <= 0 or stop <= 0:
            return 0.0
        distance = abs(entry - stop)
        if distance <= 0 or equity <= 0:
            return 0.0
        return min(
            equity * self.risk_per_trade / distance,
            equity * self.max_position_pct / entry,
        )

    def trailing_stop(
        self, highest_price: float, atr: float, current_stop: float
    ) -> float:
        return max(current_stop, highest_price - self.trailing_atr * atr)
