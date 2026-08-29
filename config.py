"""Environment-backed configuration with safe paper mode as the default."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    mode: str = os.getenv("AURA_MODE", "paper").lower()
    exchange: str = os.getenv("EXCHANGE", "binance")
    symbol: str = os.getenv("SYMBOL", "BTC/USDT")
    timeframe: str = os.getenv("TIMEFRAME", "1h")
    poll_seconds: int = int(os.getenv("POLL_SECONDS", "300"))
    api_key: str = os.getenv("BINANCE_API_KEY", "")
    secret: str = os.getenv("BINANCE_SECRET", "")
    enable_llm: bool = os.getenv("ENABLE_LLM", "false").lower() == "true"
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.01"))
    max_position_pct: float = float(os.getenv("MAX_POSITION_PCT", "0.25"))
    daily_loss_limit: float = float(os.getenv("DAILY_LOSS_LIMIT", "0.03"))
    max_drawdown: float = float(os.getenv("MAX_DRAWDOWN", "0.15"))
    trailing_atr: float = float(os.getenv("TRAILING_ATR", "2.0"))

    def validate(self) -> None:
        if self.mode not in {"backtest", "paper", "live"}:
            raise ValueError("AURA_MODE must be backtest, paper, or live")
        if self.mode == "live" and (not self.api_key or not self.secret):
            raise ValueError("Live mode requires exchange credentials")
        if self.poll_seconds < 5:
            raise ValueError("POLL_SECONDS must be >= 5")
        if not 0 < self.risk_per_trade <= 0.02:
            raise ValueError("RISK_PER_TRADE must be in (0, 0.02]")
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("MAX_POSITION_PCT must be in (0, 1]")


settings = Settings()
