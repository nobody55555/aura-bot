"""Risk controls shared by backtest, paper, and live execution paths."""

from __future__ import annotations

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
        daily_loss_limit: float = 0.03,
        max_drawdown: float = 0.15,
        trailing_atr: float = 2.0,
    ):
        if not 0 < risk_per_trade <= 0.02:
            raise ValueError("risk_per_trade must be in (0, 0.02]")
        self.risk_per_trade, self.max_position_pct = risk_per_trade, max_position_pct
        self.daily_loss_limit, self.max_drawdown, self.trailing_atr = (
            daily_loss_limit,
            max_drawdown,
            trailing_atr,
        )
        self.state: RiskState | None = None

    def begin(self, equity: float, on: date) -> None:
        self.state = RiskState(equity, equity, equity, on)

    def update_equity(self, equity: float, on: date) -> None:
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
