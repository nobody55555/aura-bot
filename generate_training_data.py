import ccxt
import pandas as pd
import os

ex = ccxt.binance()
ohlcv = pd.DataFrame(ex.fetch_ohlcv('BTC/USDT', '1h', limit=1000),
                     columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Generate labels: bullish (>1% rise next candle), bearish (<-1%), sideways
ohlcv['label'] = 'sideways'
for i in range(len(ohlcv) - 1):
    change = (ohlcv['close'][i+1] - ohlcv['close'][i]) / ohlcv['close'][i]
    if change > 0.01:
        ohlcv.at[i, 'label'] = 'bullish'
    elif change < -0.01:
        ohlcv.at[i, 'label'] = 'bearish'

ohlcv.to_csv('training_data.csv', index=False)
print("CSV generated: training_data.csv")

# For JSON (Instruction/Output pairs with roles)
data_json = []
for i in range(len(ohlcv) - 1):
    instr = f"Analyze OHLCV: {ohlcv.iloc[i].to_dict()}"
    output = ohlcv['label'][i]
    data_json.append({"system": "Market Analyst", "user": instr, "assistant": output})

with open('training_data.json', 'w') as f:
    import json
    json.dump(data_json, f)
print("JSON generated: training_data.json")
