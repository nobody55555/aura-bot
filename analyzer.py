import ccxt
import pandas as pd
import talib
import numpy as np
from typing import List
from openai import OpenAI
import os

class AuraAnalyzer:
    def __init__(self, exchange='binance'):
        self.ex = getattr(ccxt, exchange)()
        self.llm_client = OpenAI(
            base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
            api_key=os.getenv('OPENAI_API_KEY', 'sk-')
        )

    def fetch_ohlcv(self, symbol='BTC/USDT', timeframe='1h', limit=100):
        return pd.DataFrame(self.ex.fetch_ohlcv(symbol, timeframe, limit=limit),
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    def elliott_wave(self, ohlcv: pd.DataFrame) -> float:
        highs = talib.MAX(ohlcv['high'], timeperiod=5)
        lows = talib.MIN(ohlcv['low'], timeperiod=5)
        peaks = self._detect_peaks(highs.values)
        valleys = self._detect_peaks(-lows.values)
        if len(peaks) >= 3 and len(valleys) >= 2:
            w1 = highs[peaks[1]] - lows[valleys[0]]
            w2 = lows[valleys[1]] - highs[peaks[1]]
            if w2 < w1 and highs[peaks[2]] > highs[peaks[1]]:
                return 0.4
        return 0.0

    def _detect_peaks(self, x: np.ndarray, prominence=0.01) -> List[int]:
        peaks = []
        for i in range(1, len(x) - 1):
            if x[i] > max(x[max(0, i-1)], x[i+1]) * (1 + prominence):
                peaks.append(i)
        return peaks

    def candlestick_patterns(self, ohlcv: pd.DataFrame) -> float:
        patterns = {
            'hammer': talib.CDLHAMMER(ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close']),
            'doji': talib.CDLDOJI(ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close']),
            'engulfing': talib.CDLENGULFING(ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close']),
            'morning_star': talib.CDLMORNINGSTAR(ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close']),
            'evening_star': talib.CDLEVENINGSTAR(ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close']),
        }
        strength = sum(np.max(np.abs(p)) / 100 for p in patterns.values() if np.any(p != 0))
        return min(0.3, strength)

    def indicators(self, ohlcv: pd.DataFrame) -> float:
        rsi = talib.RSI(ohlcv['close'])
        macd, signal, _ = talib.MACD(ohlcv['close'])
        bb_upper, bb_middle, bb_lower = talib.BBANDS(ohlcv['close'])
        if 30 < rsi[-1] < 70 and macd[-1] > signal[-1] and ohlcv['close'][-1] > bb_middle[-1]:
            return 0.3
        return 0.0

    def ml_prediction(self, ohlcv: pd.DataFrame) -> float:
        data_str = ohlcv.tail(10).to_string()
prompt = f"""
You are a crypto market analyst. Analyze the following BTC/USDT market data (last 10 hourly candles):
{data_str}

Key indicators:
- RSI (last): {rsi[-1]:.2f} (overbought >70, oversold <30)
- MACD: {macd[-1]:.2f} (positive and crossing up: bullish)
- Bollinger Bands: Close relative to middle band ({bb_middle[-1]:.2f})

Predict the short-term direction: bullish (up), bearish (down), or sideways (neutral).
Provide a brief reasoning and a confidence score (0.0 to 1.0).
Output format: Prediction: [bullish/bearish/sideways] | Confidence: [score] | Reasoning: [short text]
"""
        response = self.llm_client.chat.completions.create(
            model="qwen-2.5-3b-instruct",  # Passe an dein Modell an
            messages=[{"role": "user", "content": prompt}]
        )
        pred = response.choices[0].message.content.lower()
        if 'bullish' in pred:
            return 0.1
        elif 'bearish' in pred:
            return -0.1
        return 0.0  # Sideways

    def signal_strength(self, ohlcv: pd.DataFrame) -> bool:
        ew = self.elliott_wave(ohlcv)
        cs = self.candlestick_patterns(ohlcv)
        ind = self.indicators(ohlcv)
        ml = self.ml_prediction(ohlcv)  # +0.1/-0.1 boost
        total = ew + cs + ind + ml
        print(f"Signal: EW={ew:.2f}, CS={cs:.2f}, IND={ind:.2f}, ML={ml:.2f} | Total={total:.2f}")
        return total > 0.6
