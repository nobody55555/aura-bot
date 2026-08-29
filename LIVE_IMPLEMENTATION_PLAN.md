# Aura Bot – Combined Live Trading Implementation Plan

**Date:** 29 August 2026  
**Status:** Phase 1 started (structure + PaperBroker + Models)

---

## 1. Goal

Turn the current research / paper framework into a **safe, minimal viable live trading system** while keeping the educational and backtesting parts intact.

**Core principles (from DeepSeek + refinements):**
- Exchange is the source of truth
- All order operations are idempotent (clientOrderId)
- Live mode requires explicit activation
- PaperBroker must exercise the exact same code path as live
- Safety monitors can halt trading at any moment
- No over-engineering – start simple, add complexity only when needed

---

## 2. Repository Decision

**Continue on the existing repository `nobody55555/aura-bot`.**

- All live code lives under a new top-level package `live/`
- Existing signal generation, risk, backtest, and features remain untouched
- Clear separation keeps the educational value while allowing production use

---

## 3. Target Directory Structure

```
aura-bot/
├── live/
│   ├── __init__.py
│   ├── models.py              # Order, Fill, Position, OrderState, etc.
│   ├── order_manager.py       # State machine + lifecycle
│   ├── safety.py              # Kill switch, stale data, clock drift
│   ├── alerts.py              # Telegram / logging alerts
│   ├── broker/
│   │   ├── __init__.py
│   │   ├── base.py            # AbstractBroker
│   │   ├── paper.py           # PaperBroker (realistic simulation)
│   │   ├── ccxt_broker.py     # Live CcxtBroker (P0)
│   │   └── utils.py           # precision, retries, helpers
│   └── db/
│       ├── __init__.py
│       ├── database.py        # SQLite connection + schema
│       └── repository.py      # CRUD for orders, fills, equity
├── ... (existing research code)
└── LIVE_IMPLEMENTATION_PLAN.md
```

---

## 4. Implementation Phases

### Phase 1 – Foundation (Current)
- [x] `live/models.py` – Order, Fill, Position, states, client_order_id generator
- [x] `live/broker/base.py` – AbstractBroker interface
- [x] `live/broker/paper.py` – Realistic PaperBroker with balance, fees, slippage, limit/stop simulation
- [x] `live/db/` – SQLite schema + repository
- [ ] Basic `OrderManager` that works with PaperBroker only

### Phase 2 – Persistence & Order Manager (Next)
- SQLite schema (orders, fills, equity_snapshots, risk_events, system_state)
- OrderManager with full state machine
- Startup reconciliation (even for paper)
- Equity snapshotting

### Phase 3 – Live Broker (CcxtBroker)
- Precision handling (`amount_to_precision`, `price_to_precision`)
- Minimum notional checks
- Idempotent order placement with clientOrderId
- Retries with exponential backoff (tenacity)
- Error mapping to typed exceptions
- Basic protection orders (separate SL + TP first)

### Phase 4 – Safety & Operational Controls
- Kill switch (manual + automatic)
- Stale data detection
- Clock drift check
- Telegram alerting
- Daily loss / max drawdown hard stops (already partially in risk.py)

### Phase 5 – Hardening & First Live Test
- Systemd unit with hardening
- Explicit live activation (`AURA_MODE=live` + `AURA_LIVE=1`)
- Small capital test plan
- Documentation update

---

## 5. Key Design Decisions (Final)

| Topic                    | Decision                                      | Reason |
|--------------------------|-----------------------------------------------|--------|
| Broker abstraction       | AbstractBroker interface                      | Same code path for paper & live |
| Order ID                 | Always generate clientOrderId                 | Idempotency & reconciliation |
| Protection orders (MVP)  | Separate SL + TP first, OCO later             | Simpler, still safe enough |
| Persistence              | SQLite + SQLAlchemy or raw sqlite3            | Zero config, sufficient for single instance |
| Data source of truth     | Exchange                                      | Always reconcile local → exchange |
| Live activation          | Double gate: config + env var                 | Prevent accidents |
| PaperBroker realism      | Fees + slippage + limit/stop logic            | Catch bugs before real money |
| WebSocket                | Start with polling, add later                 | Faster to implement MVP |

---

## 6. Safety Rules for First Live Capital

- Start with ≤ $100 equivalent
- Max risk per trade ≤ 1%
- Daily loss limit ≤ 2%
- Max drawdown kill ≤ 5%
- At least 7 days of clean paper trading with the full OrderManager before going live
- Manual kill switch tested and reachable

---

## 7. Next Concrete Steps

1. Finish SQLite layer + OrderManager (Paper only)
2. Write unit tests for PaperBroker + OrderManager
3. Implement CcxtBroker against Binance (test with very small size or testnet if available)
4. Add safety monitors
5. First controlled live test

---

**Current status:** Phase 1 foundation is in place.  
The project has been backed up as `aura-bot-backup-before-live-20260829-0347.zip`.
