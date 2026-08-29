"""A conservative, explainable Elliott-style pivot analyzer.

It deliberately returns no structure until pivots are confirmed by bars on both sides.
This reduces repainting but introduces a confirmation delay that backtests must preserve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Pivot:
    index: int
    price: float
    kind: str


def confirmed_pivots(
    close: pd.Series, order: int = 3, min_move: float = 0.005
) -> list[Pivot]:
    values = close.to_numpy(dtype=float)
    raw: list[Pivot] = []
    for i in range(order, len(values) - order):
        window = values[i - order : i + order + 1]
        if (
            values[i] == window.max()
            and values[i] > values[i - 1]
            and values[i] > values[i + 1]
        ):
            raw.append(Pivot(i, float(values[i]), "high"))
        elif (
            values[i] == window.min()
            and values[i] < values[i - 1]
            and values[i] < values[i + 1]
        ):
            raw.append(Pivot(i, float(values[i]), "low"))
    filtered: list[Pivot] = []
    for pivot in raw:
        if not filtered:
            filtered.append(pivot)
            continue
        previous = filtered[-1]
        if pivot.kind == previous.kind:
            if (pivot.kind == "high" and pivot.price > previous.price) or (
                pivot.kind == "low" and pivot.price < previous.price
            ):
                filtered[-1] = pivot
        elif abs(pivot.price / previous.price - 1) >= min_move:
            filtered.append(pivot)
    return filtered


def analyze_structure(
    df: pd.DataFrame, order: int = 3, min_move: float = 0.005
) -> dict:
    pivots = confirmed_pivots(df["close"], order, min_move)
    if len(pivots) < 5:
        return {
            "direction": "neutral",
            "confidence": 0.0,
            "pivots": pivots,
            "pattern": None,
        }
    p = pivots[-5:]
    prices = [x.price for x in p]
    moves = np.diff(prices)
    bullish = [
        p[0].kind == "low",
        p[1].kind == "high",
        p[2].kind == "low",
        p[3].kind == "high",
        p[4].kind == "low",
    ]
    bearish = [
        p[0].kind == "high",
        p[1].kind == "low",
        p[2].kind == "high",
        p[3].kind == "low",
        p[4].kind == "high",
    ]
    direction = "neutral"
    pattern = None
    score = 0.0
    if all(bullish) and moves[0] > 0 and moves[1] < 0 and moves[2] > 0 and moves[3] < 0:
        w1, retrace, w3 = abs(moves[0]), abs(moves[1]), abs(moves[2])
        ratio = retrace / w1 if w1 else 99
        if 0.236 <= ratio <= 0.886 and w3 >= 0.618 * w1 and prices[3] > prices[1]:
            direction, pattern, score = "bull", "impulse_candidate", 0.8
    elif (
        all(bearish) and moves[0] < 0 and moves[1] > 0 and moves[2] < 0 and moves[3] > 0
    ):
        w1, retrace, w3 = abs(moves[0]), abs(moves[1]), abs(moves[2])
        ratio = retrace / w1 if w1 else 99
        if 0.236 <= ratio <= 0.886 and w3 >= 0.618 * w1 and prices[3] < prices[1]:
            direction, pattern, score = "bear", "impulse_candidate", 0.8
    return {
        "direction": direction,
        "confidence": score,
        "pivots": p,
        "pattern": pattern,
    }
