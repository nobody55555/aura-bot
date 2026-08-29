"""Classical, inspectable signal engine for Aura."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from elliott import analyze_structure
from features import add_features


@dataclass(frozen=True)
class Signal:
    action: str
    score: float
    confidence: float
    regime: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


class ClassicalStrategy:
    def __init__(self, min_confidence: float = 0.62, pivot_order: int = 3):
        self.min_confidence, self.pivot_order = min_confidence, pivot_order

    def signal(self, raw: pd.DataFrame) -> Signal:
        df = add_features(raw)
        row = df.iloc[-1]
        if pd.isna(
            row[["atr_pct", "ema_20", "ema_50", "rsi_14", "macd_hist", "bb_position"]]
        ).any():
            return Signal("hold", 0.0, 0.0, "unknown", ("insufficient_history",))
        regime = str(row["regime"])
        bull, bear, reasons = 0.0, 0.0, []
        # Regime-aware weights: trend signals matter more in trends; mean reversion in ranges.
        if regime == "bull" and row["ema_20"] > row["ema_50"]:
            bull += 0.28
            reasons.append("ema_trend_bull")
        elif regime == "bear" and row["ema_20"] < row["ema_50"]:
            bear += 0.28
            reasons.append("ema_trend_bear")
        if row["macd_hist"] > 0:
            bull += 0.18
            reasons.append("macd_positive")
        elif row["macd_hist"] < 0:
            bear += 0.18
            reasons.append("macd_negative")
        if 50 <= row["rsi_14"] <= 68:
            bull += 0.12
            reasons.append("rsi_supportive")
        elif 32 <= row["rsi_14"] <= 50:
            bear += 0.12
            reasons.append("rsi_weak")
        if row["bullish_hammer"] or row["bullish_engulfing"]:
            bull += 0.16
            reasons.append("bullish_candle")
        if row["bearish_shooting_star"] or row["bearish_engulfing"]:
            bear += 0.16
            reasons.append("bearish_candle")
        wave = analyze_structure(df, self.pivot_order)
        if wave["direction"] == "bull":
            bull += 0.18 * wave["confidence"]
            reasons.append("elliott_bull_structure")
        elif wave["direction"] == "bear":
            bear += 0.18 * wave["confidence"]
            reasons.append("elliott_bear_structure")
        if regime == "range":
            if row["bb_position"] < 0.15 and row["rsi_14"] < 40:
                bull += 0.15
                reasons.append("range_lower_band")
            if row["bb_position"] > 0.85 and row["rsi_14"] > 60:
                bear += 0.15
                reasons.append("range_upper_band")
        total = bull + bear
        if total == 0 or abs(bull - bear) < 0.14:
            return Signal("hold", bull - bear, min(1.0, total), regime, tuple(reasons))
        action, leader = ("buy", bull) if bull > bear else ("sell", bear)
        confidence = min(1.0, leader / 0.9)
        if confidence < self.min_confidence:
            action = "hold"
        return Signal(action, bull - bear, confidence, regime, tuple(reasons))
