# Aura Trading Bot

Classical, inspectable trading framework for BTC/USDT with a full **paper-trading runtime** and a **safe path toward live execution**.

Educational / research use. Crypto trading is high risk — never risk money you cannot afford to lose.

---

## Current status (feature/live-foundation)

| Layer | Status |
|-------|--------|
| Signal engine (EMA, RSI, MACD, BB, candles, Elliott pivots) | ✅ Working |
| Risk manager (position size, daily loss, max drawdown) | ✅ Working |
| PaperBroker + OrderManager (virtual balance, SL/TP) | ✅ Working |
| SQLite persistence + reconciliation | ✅ Working |
| SafetyMonitor (kill-switch, stale data, clock drift) | ✅ Working |
| E-mail alerts (SMTP via env) | ✅ Working |
| CcxtBroker (live adapter) | ✅ Implemented, **live orders still gated** |
| Live order placement | 🔒 Explicitly blocked in `main.py` until you enable it |

**Default mode is paper.** No exchange API keys required for paper mode.

---

## Architecture

```
signal (analyzer / strategy / features / elliott)
    → risk checks
    → OrderManager
        → PaperBroker  (default)  or  CcxtBroker (live, gated)
    → SQLite (orders, fills, system state)
    → SafetyMonitor + optional e-mail alerts
```

Design principles:

- Exchange is the source of truth (reconciliation on startup and periodically).
- Every order uses a unique `clientOrderId` (idempotency).
- Live mode requires explicit config + credentials; order placement remains gated.
- LLM is optional and **never** required for the primary signal path.

---

## Quick start (Debian / paper mode)

```bash
git clone https://github.com/nobody55555/aura-bot.git
cd aura-bot
git checkout feature/live-foundation

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# TA-Lib C library (once) — use su if you have no sudo:
#   su -c 'apt install -y build-essential wget'
#   then build ta-lib 0.4.0 from source into /usr (see project history / issues)
pip install TA-Lib

cp .env.example .env   # edit only if you want e-mail alerts
export AURA_MODE=paper
python main.py
```

You should see startup reconciliation and periodic lines like:

```text
signal=hold confidence=0.44 regime=bear reasons=(...)
```

When a **buy** signal passes the confidence filter, paper mode will:

1. Place a virtual market order against the simulated balance (default 10 000 USDT).
2. Attach stop-loss and take-profit protection orders.
3. Persist everything in `data/aura_live.db`.

`sell` signals are currently **logged only** (no automatic short/exit yet).

---

## Configuration

All settings via environment / `.env` (see `.env.example`):

| Variable | Default | Notes |
|----------|---------|--------|
| `AURA_MODE` | `paper` | `paper` \| `live` \| `backtest` |
| `SYMBOL` | `BTC/USDT` | |
| `TIMEFRAME` | `1h` | |
| `POLL_SECONDS` | `300` | Loop interval |
| `RISK_PER_TRADE` | `0.01` | 1 % |
| `DAILY_LOSS_LIMIT` | `0.03` | Hard halt |
| `MAX_DRAWDOWN` | `0.15` | Hard halt |
| `BINANCE_API_KEY` / `SECRET` | empty | Required only for live |
| `ALERT_EMAIL_*` / `SMTP_*` | empty | Optional e-mail alerts |

**Never commit real API keys or SMTP passwords.**

---

## Package layout

```
aura-bot/
├── main.py                 # Paper/live runtime loop
├── config.py               # Env-backed settings
├── analyzer.py             # Data + signal facade
├── features.py             # Causal OHLCV features
├── strategy.py             # ClassicalStrategy (confidence filter)
├── elliott.py              # Conservative pivot structure
├── risk.py                 # Position sizing + limits
├── live/
│   ├── models.py           # Order, Fill, states
│   ├── order_manager.py    # Lifecycle + protection orders
│   ├── safety.py           # Kill-switch, stale, drawdown, reconciler
│   ├── alerts.py           # SMTP e-mail (IRC placeholder)
│   ├── exceptions.py
│   ├── broker/
│   │   ├── base.py         # AbstractBroker
│   │   ├── paper.py        # Simulated exchange
│   │   ├── ccxt_broker.py  # Live CCXT adapter
│   │   └── utils.py        # Precision / min-notional
│   └── db/
│       ├── database.py     # SQLite schema
│       └── repository.py   # Persistence
├── tests/
└── LIVE_IMPLEMENTATION_PLAN.md
```

---

## Why so many `hold` signals?

The strategy is intentionally selective (`min_confidence ≈ 0.62`). In a sustained **bear** regime, bullish scores rarely clear the threshold — that is by design, not a bug. Occasional `sell` or near-threshold scores show the engine is reacting to candles/RSI/MACD.

To see more paper trades while testing you can temporarily lower `ClassicalStrategy(min_confidence=...)` in `strategy.py`, or wait for a regime shift.

---

## Backtesting

```bash
python -m backtest_cli   # if backtest modules are present on your branch
```

Prefer validating signal quality in backtests before relying on paper/live behaviour.

---

## Live mode (not enabled by default)

1. Set `AURA_MODE=live` and valid exchange credentials in `.env`.
2. Run extensively in paper first.
3. Remove the explicit live-order gate in `main.py` only when you accept the risk.
4. Start with very small size and strict daily loss / drawdown limits.

---

## License

MIT — use at your own risk.
