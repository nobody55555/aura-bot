"""Compatibility facade for data access and classical signal generation."""

from __future__ import annotations

import ccxt
import pandas as pd

from features import add_features, validate_ohlcv
from strategy import ClassicalStrategy, Signal


class AuraAnalyzer:
    def __init__(self, exchange: object | None = None, exchange_name: str = "binance"):
        self.ex = exchange or getattr(ccxt, exchange_name)({"enableRateLimit": True})
        self.strategy = ClassicalStrategy()

    def fetch_ohlcv(
        self, symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 250
    ) -> pd.DataFrame:
        rows = self.ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        frame = pd.DataFrame(
            rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame = frame.set_index("timestamp")
        validate_ohlcv(frame)
        return frame

    def features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        return add_features(ohlcv)

    def signal(self, ohlcv: pd.DataFrame) -> Signal:
        return self.strategy.signal(ohlcv)

    def signal_strength(self, ohlcv: pd.DataFrame) -> bool:
        return self.signal(ohlcv).action == "buy"
