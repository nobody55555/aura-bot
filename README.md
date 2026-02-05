# Aura AI Trading Bot

AI-powered Bitcoin trading bot using Elliott Wave, candlestick patterns, technical indicators, and optional self-hosted LLM for predictions. Educational purpose only – high risk in crypto trading!

## Disclaimer
This is for education. Start with paper trading. Never risk more than you can afford to lose.

## Features
- Real-time BTC data via CCXT.
- Elliott Wave detection.
- Candlestick recognition (TA-Lib).
- Indicators: RSI, MACD, Bollinger Bands.
- Risk management: 1-2% per trade, 3:1 RR.
- LLM integration: Use self-hosted model (e.g., Qwen 2.5) via OpenAI API for price predictions (bullish/bearish/sideways).
- Training data generation: Export OHLCV + labels to CSV for fine-tuning.

## Setup
1. Clone repo: `git clone https://github.com/dein_username/aura-bot.git`
2. Create venv: `python -m venv venv && source venv/bin/activate`
3. Install deps: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and fill in keys (use paper trading API first!).
5. Run: `python main.py`

For Debian: `apt install python3 python3-venv python3-pip libta-lib-dev` (for TA-Lib).

## LLM Integration
- Set `OPENAI_BASE_URL` in .env to your self-hosted endpoint (e.g., http://localhost:8000/v1).
- Model prompts for market sentiment (bullish/bearish/sideways).

## Training Data
Run `python generate_training_data.py` to generate CSV with OHLCV and labels for your QLora training. Adapt to JSON if needed.

## Systemd Service (Optional)
Copy `aura-bot.service` to `/etc/systemd/system/`, edit paths, then `systemctl enable --now aura-bot`.

## Performance
Target: 5-15% monthly with $50 start. Test with backtesting!

License: MIT
