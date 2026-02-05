import time
import os
from dotenv import load_dotenv
from analyzer import AuraAnalyzer
from risk import RiskManager
import pandas as pd

load_dotenv()

analyzer = AuraAnalyzer()
risk_mgr = RiskManager()

while True:
    try:
        ohlcv = analyzer.fetch_ohlcv()
        if analyzer.signal_strength(ohlcv):
            balance = analyzer.ex.fetch_balance()['USDT']['free']
            entry_price = ohlcv['close'][-1]
            stop_loss = entry_price * 0.98
            size = risk_mgr.position_size(balance, entry_price, stop_loss)
            if size > 0:
                # analyzer.ex.create_market_buy_order('BTC/USDT', size)  # Live: Entkommentieren
                sl, tp = risk_mgr.set_orders(analyzer.ex, 'BTC/USDT', size, entry_price)
                print(f"Buy {size:.4f} BTC @ {entry_price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
        time.sleep(300)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60)
