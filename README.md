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
| CcxtBroker (live adapter) | ✅ Implemented for explicitly selected live mode |
| Live order placement | ✅ Available after safety preflight and risk checks |

**Default mode is paper.** No exchange API keys required for paper mode.

For the current reviewed state, exact code changes, tests, operating flow, and open review questions, see [the 27 November 2026 review handoff](docs/AURA_BOT_REVIEW_HANDOFF_2026-11-27.md). The older [25 November handoff](HANDOFF_AURA_2026-11-25.md) is a historical snapshot and is superseded where its status differs.

---

## Architecture

```
signal (analyzer / strategy / features / elliott)
    → risk checks
    → OrderManager
        → PaperBroker (default) or CcxtBroker (live, explicit opt-in)
    → SQLite (orders, fills, system state)
    → SafetyMonitor + optional e-mail alerts
```

Design principles:

- Exchange is the source of truth (reconciliation on startup and periodically).
- Every order uses a unique `clientOrderId` (idempotency).
- Live mode requires explicit config + credentials; eligible orders pass through the OrderManager and its protection-order handling.
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
| `RISK_PER_TRADE` | `0.01` | Maximum 1 % per trade |
| `DAILY_LOSS_LIMIT` | `0.02` | Maximum 2 %; hard halt at the threshold |
| `MAX_DRAWDOWN` | `0.05` | Maximum 5 %; hard halt at the threshold |
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

## Hyperliquid Testnet (read-only)

`live/hyperliquid_testnet.py` is an isolated, read-only client for HIP-4 outcome metadata and market data. Its `find_outcomes` method fetches `{"type":"outcomeMeta"}` and filters the returned outcomes locally; additional read-only methods retrieve price contexts via `spotMetaAndAssetCtxs` and order books via `l2Book`. It has no signing, wallet, order, cancel, transfer, or HyperEVM methods, and is not wired into the trading runtime.

The REST base comes from the official SDK's `constants.TESTNET_API_URL` (SDK version pinned in `requirements.txt`); the client revalidates the exact Testnet endpoints before every request, and URL overrides, a missing `HL_ENV`, or any value other than `testnet` fail closed. Set `HL_ENV=testnet` explicitly before constructing the client. It uses a direct fixed-host HTTPS request rather than constructing an SDK `Info` or `Exchange` object, so there is no nullable SDK `base_url` default or signing engine in this path. For example:

```python
import os

os.environ["HL_ENV"] = "testnet"

from live.hyperliquid_testnet import HyperliquidTestnetClient

client = HyperliquidTestnetClient()
btc_outcomes = client.find_outcomes(underlying="BTC")
print(btc_outcomes)
```

The official REST base is `https://api.hyperliquid-testnet.xyz` (not the host without `api.`), as listed in the [SDK constants](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/hyperliquid/utils/constants.py); the [API docs](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api) say the same requests can be made against Testnet. HIP-4's [`outcomeMeta` Info request](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/spot) returns the outcome records. The official [Testnet WebSocket URL](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket) is `wss://api.hyperliquid-testnet.xyz/ws`, but this module does not open WebSocket connections. No Testnet orders are sent.

---

## Live mode (available, not enabled by default)

1. Set `AURA_MODE=live` and valid exchange credentials in `.env`; the default remains `paper`.
2. Run extensively in paper first and independently review strategy behavior and order sizing.
3. In live mode, eligible buy signals can submit real market orders through the configured CCXT exchange after account-equity, exchange-clock, kill-switch, and risk checks pass. Filled entries request stop-loss and take-profit protection through the OrderManager.
4. Real execution, exchange-specific order parameters, and protective-order behavior must be verified for the selected exchange before risking funds. Use API keys restricted to the minimum required permissions, start with very small size, and set strict daily-loss / drawdown limits.

---

## License

MIT — use at your own risk.
