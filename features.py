"""Leakage-safe, lightweight OHLCV features used by Aura."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import talib
except ImportError:  # pragma: no cover
    talib = None


def validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("OHLCV frame is empty")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    if talib is not None:
        return pd.Series(talib.RSI(close.values, timeperiod=period), index=close.index)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if talib is not None:
        return pd.Series(
            talib.ATR(df["high"].values, df["low"].values, df["close"].values, timeperiod=period),
            index=df.index,
        )
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Add classical features. All calculations are causal (no future peek).
    """
    validate_ohlcv(ohlcv)
    df = ohlcv.copy()

    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"]).diff()

    df["ema_20"] = _ema(df["close"], 20)
    df["ema_50"] = _ema(df["close"], 50)
    df["ema_200"] = _ema(df["close"], 200)

    df["rsi_14"] = _rsi(df["close"], 14)
    df["atr_14"] = _atr(df, 14)

    # Simple regime proxy: trend strength vs noise
    df["trend_strength"] = (df["ema_20"] - df["ema_50"]).abs() / df["atr_14"].replace(0, np.nan)
    df["vol_regime"] = df["atr_14"] / df["close"]

    # Volume relative to recent average
    df["vol_ma_20"] = df["volume"].rolling(20, min_periods=5).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma_20"].replace(0, np.nan)

    return df
