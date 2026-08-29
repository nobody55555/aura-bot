"""
Aura runtime.

Paper mode is the default and fully wired to:
  - Signal engine (analyzer)
  - RiskManager
  - OrderManager + PaperBroker
  - SafetyMonitor (kill-switch, stale data, daily loss, drawdown)
  - Reconciler
  - AlertManager (e-mail, optional)

Live mode requires explicit activation and a working CcxtBroker + credentials.
Live order placement remains gated until you deliberately enable it.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from analyzer import AuraAnalyzer
from config import settings
from risk import RiskManager

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("aura")


def build_components():
    """Wire signal, risk, broker, order manager, safety and alerts."""
    settings.validate()

    # --- Alerts (optional – works even if SMTP not configured) ---
    from live.alerts import AlertManager
    alerts = AlertManager()

    # --- Persistence ---
    from live.db.repository import OrderRepository
    repo = OrderRepository()

    # --- Broker (paper by default) ---
    if settings.mode == "live":
        from live.broker.ccxt_broker import CcxtBroker
        if not settings.api_key or not settings.secret:
            raise SystemExit("Live mode requires BINANCE_API_KEY and BINANCE_SECRET")
        broker = CcxtBroker(
            exchange_id=settings.exchange,
            api_key=settings.api_key,
            secret=settings.secret,
            sandbox=False,
        )
        log.warning("LIVE broker instantiated – real orders possible")
    else:
        from live.broker.paper import PaperBroker
        broker = PaperBroker(initial_balance=10_000.0)
        log.info("PaperBroker active (simulated fills)")

    # --- Safety + Reconciler ---
    from live.safety import SafetyMonitor, Reconciler
    safety = SafetyMonitor(
        repository=repo,
        max_stale_seconds=max(120.0, settings.poll_seconds * 2),
        daily_loss_limit=settings.daily_loss_limit,
        max_drawdown=settings.max_drawdown,
        alert_manager=alerts,
    )
    reconciler = Reconciler(repo, broker, safety=safety)

    # --- Order manager ---
    from live.order_manager import OrderManager
    from live.models import OrderSide, OrderType
    order_manager = OrderManager(broker=broker, repository=repo)

    # --- Classic signal + risk (unchanged research path) ---
    # For paper we still need a data source for OHLCV.
    # Use a lightweight public ccxt instance (no keys required for public data).
    import ccxt
    data_exchange = getattr(ccxt, settings.exchange)({"enableRateLimit": True})
    analyzer = AuraAnalyzer(exchange=data_exchange)
    risk = RiskManager(
        risk_per_trade=settings.risk_per_trade,
        max_position_pct=settings.max_position_pct,
        daily_loss_limit=settings.daily_loss_limit,
        max_drawdown=settings.max_drawdown,
        trailing_atr=settings.trailing_atr,
    )

    return {
        "alerts": alerts,
        "repo": repo,
        "broker": broker,
        "safety": safety,
        "reconciler": reconciler,
        "order_manager": order_manager,
        "analyzer": analyzer,
        "risk": risk,
        "OrderSide": OrderSide,
        "OrderType": OrderType,
    }


def run() -> None:
    if settings.mode == "backtest":
        raise SystemExit(
            "Use `python -m backtest_cli` for backtests; this runtime is for paper/live."
        )

    ctx = build_components()
    alerts = ctx["alerts"]
    safety = ctx["safety"]
    reconciler = ctx["reconciler"]
    order_manager = ctx["order_manager"]
    analyzer = ctx["analyzer"]
    risk = ctx["risk"]
    broker = ctx["broker"]
    OrderSide = ctx["OrderSide"]
    OrderType = ctx["OrderType"]

    log.info(
        "Aura started in %s mode | %s %s | poll=%ss",
        settings.mode,
        settings.symbol,
        settings.timeframe,
        settings.poll_seconds,
    )

    # Startup reconciliation
    try:
        summary = reconciler.reconcile_on_startup(settings.symbol)
        log.info("Startup reconciliation: %s", summary)
    except Exception:
        log.exception("Startup reconciliation failed")

    safety.restore_equity_state()

    while True:
        try:
            # --- Safety gates ---
            safety.require_not_halted()

            # Fetch data
            ohlcv = analyzer.fetch_ohlcv(settings.symbol, settings.timeframe, limit=250)
            if ohlcv is None or len(ohlcv) < 50:
                log.warning("Insufficient OHLCV data")
                time.sleep(settings.poll_seconds)
                continue

            # Mark data as fresh
            try:
                last_ts = ohlcv.index[-1].timestamp()
            except Exception:
                last_ts = time.time()
            safety.update_bar_timestamp(settings.symbol, last_ts)
            safety.check_stale(settings.symbol)

            # Clock drift (live only, if broker supports it)
            if settings.mode == "live" and hasattr(broker, "exchange"):
                try:
                    server_ms = broker.exchange.fetch_time()
                    safety.check_clock_drift(server_ms)
                except Exception:
                    pass

            # Signal
            signal = analyzer.signal(ohlcv)
            last = float(ohlcv["close"].iloc[-1])
            log.info(
                "signal=%s confidence=%.3f regime=%s reasons=%s",
                signal.action,
                signal.confidence,
                signal.regime,
                signal.reasons,
            )

            # Equity / risk update
            try:
                bal = broker.fetch_balance()
                equity = float(bal.get("USDT", {}).get("total") or bal.get("USDT", {}).get("free") or 0)
            except Exception:
                equity = 10_000.0  # fallback for pure paper without prior fills
            safety.update_equity(equity)
            risk.update_equity(equity, datetime.now(timezone.utc).date())

            # Periodic light reconciliation
            if int(time.time()) % max(300, settings.poll_seconds * 5) < settings.poll_seconds:
                try:
                    reconciler.reconcile_open_orders(settings.symbol)
                except Exception:
                    log.exception("Periodic reconciliation failed")

            # --- Order decision (paper fully wired, live still gated) ---
            if signal.action == "buy" and risk.can_trade():
                atr = float(analyzer.features(ohlcv)["atr_14"].iloc[-1])
                stop = last - 2.0 * atr
                size = risk.position_size(equity, last, stop)

                if size > 0:
                    if settings.mode == "paper":
                        log.info(
                            "PAPER → creating entry order: buy %.8f %s @ ~%.2f (stop %.2f)",
                            size, settings.symbol, last, stop,
                        )
                        # Keep PaperBroker price in sync
                        if hasattr(broker, "update_price"):
                            broker.update_price(settings.symbol, last)

                        order = order_manager.create_entry_order(
                            symbol=settings.symbol,
                            side=OrderSide.BUY,
                            amount=size,
                            order_type=OrderType.MARKET,
                            strategy_tag="aura-signal",
                            place_protection=True,
                        )
                        log.info(
                            "Order result: %s status=%s filled=%.6f",
                            order.client_order_id,
                            order.status.value,
                            order.filled,
                        )
                    else:
                        # Live still explicitly blocked until you remove this gate
                        log.error(
                            "Live order placement is still gated. "
                            "Implement final checks / manual enable before removing this guard."
                        )
                        alerts.warning(
                            "Live order blocked",
                            "Signal wanted to buy but live placement is gated in main.py",
                        )

            time.sleep(settings.poll_seconds)

        except KeyboardInterrupt:
            log.info("Aura stopped by user")
            return
        except Exception as e:
            from live.exceptions import SystemHalted, RiskLimitExceeded
            if isinstance(e, (SystemHalted, RiskLimitExceeded)):
                log.critical("Trading halted: %s", e)
                # Stay in loop but do not place orders; wait for manual resume
                time.sleep(settings.poll_seconds)
                continue
            log.exception("Runtime cycle failed")
            time.sleep(min(settings.poll_seconds, 60))


if __name__ == "__main__":
    run()
