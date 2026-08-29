"""Leakage-safe, lightweight OHLCV features used by Aura."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    out = df.copy().sort_index()
    close = out["close"].astype(float)
    high, low, volume = (
        out["high"].astype(float),
        out["low"].astype(float),
        out["volume"].astype(float),
    )
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    out["return_1"] = close.pct_change()
    out["log_return"] = np.log(close).diff()
    out["atr_14"] = tr.rolling(14, min_periods=14).mean()
    out["atr_pct"] = out["atr_14"] / close
    out["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    out["ema_50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    rs = gain.rolling(14, min_periods=14).mean() / loss.rolling(
        14, min_periods=14
    ).mean().replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    ema12, ema26 = (
        close.ewm(span=12, adjust=False, min_periods=26).mean(),
        close.ewm(span=26, adjust=False, min_periods=26).mean(),
    )
    out["macd_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    out["bb_position"] = (close - (mid - 2 * std)) / (4 * std).replace(0, np.nan)
    out["volume_ratio"] = volume / volume.rolling(20, min_periods=20).mean().replace(
        0, np.nan
    )
    out["volatility_20"] = out["log_return"].rolling(
        20, min_periods=20
    ).std() * np.sqrt(24 * 365)
    body = (close - out["open"]).abs()
    candle_range = (high - low).replace(0, np.nan)
    lower_wick = pd.concat([out["open"], close], axis=1).min(axis=1) - low
    upper_wick = high - pd.concat([out["open"], close], axis=1).max(axis=1)
    out["bullish_hammer"] = (
        (lower_wick >= 2 * body) & (upper_wick <= body) & (body / candle_range < 0.45)
    ).astype(int)
    out["bearish_shooting_star"] = (
        (upper_wick >= 2 * body) & (lower_wick <= body) & (body / candle_range < 0.45)
    ).astype(int)
    prev_open, prev_close = out["open"].shift(1), close.shift(1)
    out["bullish_engulfing"] = (
        (close > out["open"])
        & (prev_close < prev_open)
        & (close >= prev_open)
        & (out["open"] <= prev_close)
    ).astype(int)
    out["bearish_engulfing"] = (
        (close < out["open"])
        & (prev_close > prev_open)
        & (close <= prev_open)
        & (out["open"] >= prev_close)
    ).astype(int)
    out["regime"] = np.select(
        [out["ema_20"] > out["ema_50"] * 1.002, out["ema_20"] < out["ema_50"] * 0.998],
        ["bull", "bear"],
        default="range",
    )
    return out


def validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV index must be unique and sorted ascending")
    if (df[["open", "high", "low", "close"]] <= 0).any().any() or (
        df["volume"] < 0
    ).any():
        raise ValueError("Prices must be positive and volume cannot be negative")
    if (df["high"] < df[["open", "close"]].max(axis=1)).any() or (
        df["low"] > df[["open", "close"]].min(axis=1)
    ).any():
        raise ValueError("OHLC relationships are invalid")
