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

import argparse
import logging
import math
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from analyzer import AuraAnalyzer
from config import settings
from live.exceptions import RiskLimitExceeded, SystemHalted
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


def _halt_and_raise(safety, reason: str, cause: Exception) -> None:
    """Latch a local halt even when persisting it fails, then abort this cycle."""
    try:
        safety.halt(reason)
    except Exception as persist_error:
        log.critical("Could not persist safety halt (%s): %s", reason, persist_error)
        cause = persist_error
    raise SystemHalted(reason) from cause


def _checked_total_equity(broker, safety) -> float:
    """Read total USDT equity only; free balance is not account equity."""
    try:
        balances = broker.fetch_balance()
        if not isinstance(balances, dict):
            raise ValueError("balance response is not a mapping")
        usdt = balances.get("USDT")
        if not isinstance(usdt, dict):
            raise ValueError("USDT balance is unavailable")
        total = usdt.get("total")
        if total is None:
            raise ValueError("USDT total equity is unknown")
        if isinstance(total, bool):
            raise ValueError("USDT total equity is not numeric")
        equity = float(total)
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError(f"USDT total equity must be finite and positive, got {total!r}")
        return equity
    except Exception as e:
        _halt_and_raise(safety, f"Equity unavailable or invalid: {e}", e)


def _checked_live_exchange_time(broker, safety) -> float:
    """Require and validate the live broker's exchange clock before any signal."""
    try:
        exchange = getattr(broker, "exchange", None)
        if exchange is None:
            raise ValueError("live broker has no exchange attribute")
        fetch_time = getattr(exchange, "fetch_time", None)
        if not callable(fetch_time):
            raise ValueError("live broker exchange has no fetch_time method")
        server_ms = fetch_time()
    except Exception as e:
        _halt_and_raise(safety, f"Cannot verify exchange clock: {e}", e)
    safety.check_clock_drift(server_ms)
    return float(server_ms)


def _process_cycle(ctx, mode: str | None = None) -> None:
    """Run one cycle; all fail-closed market/account checks precede signals."""
    mode = settings.mode if mode is None else mode
    safety = ctx["safety"]
    safety.require_not_halted()

    analyzer = ctx["analyzer"]
    broker = ctx["broker"]
    risk = ctx["risk"]
    ohlcv = analyzer.fetch_ohlcv(settings.symbol, settings.timeframe, limit=250)
    if ohlcv is None or len(ohlcv) < 50:
        log.warning("Insufficient OHLCV data")
        return

    # A closed candle can be old while the feed is current; freshness measures
    # the most recent successful fetch, not the candle's open timestamp.
    safety.update_bar_timestamp(settings.symbol, time.time())
    safety.check_stale(settings.symbol)

    equity = _checked_total_equity(broker, safety)
    try:
        safety.update_equity(equity)
    except (SystemHalted, RiskLimitExceeded):
        raise
    except Exception as e:
        _halt_and_raise(safety, f"Cannot validate or persist account equity: {e}", e)
    risk.update_equity(equity, datetime.now(timezone.utc).date())
    if mode == "live":
        _checked_live_exchange_time(broker, safety)

    # No signal or order decision is reachable until clock and equity checks pass.
    signal = analyzer.signal(ohlcv)
    last = float(ohlcv["close"].iloc[-1])
    log.info(
        "signal=%s confidence=%.3f regime=%s reasons=%s",
        signal.action,
        signal.confidence,
        signal.regime,
        signal.reasons,
    )

    if int(time.time()) % max(300, settings.poll_seconds * 5) < settings.poll_seconds:
        try:
            ctx["reconciler"].reconcile_open_orders(settings.symbol)
        except Exception:
            log.exception("Periodic reconciliation failed")
        safety.require_not_halted()

    if signal.action == "buy" and risk.can_trade():
        atr = float(analyzer.features(ohlcv)["atr_14"].iloc[-1])
        stop = last - 2.0 * atr
        size = risk.position_size(equity, last, stop)

        if size > 0:
            if mode == "paper":
                log.info(
                    "PAPER → creating entry order: buy %.8f %s @ ~%.2f (stop %.2f)",
                    size, settings.symbol, last, stop,
                )
                if hasattr(broker, "update_price"):
                    broker.update_price(settings.symbol, last)

                order = ctx["order_manager"].create_entry_order(
                    symbol=settings.symbol,
                    side=ctx["OrderSide"].BUY,
                    amount=size,
                    order_type=ctx["OrderType"].MARKET,
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
                log.error(
                    "Live order placement is still gated. "
                    "Implement final checks / manual enable before removing this guard."
                )
                ctx["alerts"].warning(
                    "Live order blocked",
                    "Signal wanted to buy but live placement is gated in main.py",
                )


def _operator_resume(
    ctx, reason: str, startup_summary, startup_error, mode: str | None = None
) -> None:
    """Clear a saved halt only after operator request and fresh safety checks."""
    mode = settings.mode if mode is None else mode
    safety = ctx["safety"]
    if not safety.is_halted():
        raise SystemHalted("--resume requested, but no persisted halt is active")
    prior_reason = safety.get_halt_reason()

    def deny(error: Exception) -> None:
        detail = f"Operator resume denied; active halt={prior_reason}; preflight failed: {error}"
        try:
            safety.halt(detail)
        except Exception as persist_error:
            log.critical("Could not persist resume denial: %s", persist_error)
        try:
            safety.repo.record_risk_event("RESUME_DENIED", detail)
        except Exception as persist_error:
            log.critical("Could not persist resume denial event: %s", persist_error)
        raise SystemHalted(detail) from error

    try:
        if startup_error is not None:
            raise ValueError(f"startup reconciliation failed: {startup_error}")
        if startup_summary is None:
            raise ValueError("startup reconciliation produced no result")
        if startup_summary.get("local_only") or startup_summary.get("remote_only"):
            raise ValueError(f"unresolved startup order discrepancies: {startup_summary}")

        ohlcv = ctx["analyzer"].fetch_ohlcv(
            settings.symbol, settings.timeframe, limit=250
        )
        if ohlcv is None or len(ohlcv) < 50:
            raise ValueError("fresh market data is unavailable")
        safety.update_bar_timestamp(settings.symbol, time.time())
        safety.check_stale(settings.symbol)

        equity = _checked_total_equity(ctx["broker"], safety)
        exchange_time = None
        if mode == "live":
            exchange_time = _checked_live_exchange_time(ctx["broker"], safety)
        resume_checks = {
            "equity": equity,
            "exchange_required": mode == "live",
            "operator_approved": True,
        }
        if mode == "live":
            resume_checks["exchange_time_ms"] = exchange_time
        safety.resume(reason, **resume_checks)
    except Exception as e:
        if isinstance(e, SystemHalted) and safety.is_halted():
            try:
                safety.repo.record_risk_event(
                    "RESUME_DENIED",
                    f"Operator approval={reason}; active halt={prior_reason}; failure={e}",
                )
            except Exception as persist_error:
                log.critical("Could not persist resume denial event: %s", persist_error)
            raise
        deny(e)


def run(resume_reason: str | None = None) -> None:
    if settings.mode == "backtest":
        raise SystemExit(
            "Use `python -m backtest_cli` for backtests; this runtime is for paper/live."
        )

    ctx = build_components()
    safety = ctx["safety"]
    reconciler = ctx["reconciler"]

    log.info(
        "Aura started in %s mode | %s %s | poll=%ss",
        settings.mode,
        settings.symbol,
        settings.timeframe,
        settings.poll_seconds,
    )

    # Startup reconciliation; a saved halt is never cleared automatically.
    startup_summary = None
    startup_error = None
    try:
        startup_summary = reconciler.reconcile_on_startup(settings.symbol)
        log.info("Startup reconciliation: %s", startup_summary)
    except Exception as e:
        startup_error = e
        log.exception("Startup reconciliation failed")

    safety.restore_equity_state()

    if resume_reason is not None:
        _operator_resume(ctx, resume_reason, startup_summary, startup_error)
    elif safety.is_halted():
        log.warning("Persisted safety halt remains active: %s", safety.get_halt_reason())

    while True:
        try:
            _process_cycle(ctx)
            time.sleep(settings.poll_seconds)

        except KeyboardInterrupt:
            log.info("Aura stopped by user")
            return
        except Exception as e:
            if isinstance(e, (SystemHalted, RiskLimitExceeded)):
                log.critical("Trading halted: %s", e)
                # Stay fail-closed; an operator must restart with --resume.
                time.sleep(settings.poll_seconds)
                continue
            log.exception("Runtime cycle failed")
            time.sleep(min(settings.poll_seconds, 60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aura trading runtime")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="explicitly request resume of a persisted halt after safety preflight",
    )
    parser.add_argument(
        "--reason",
        help="operator approval/reason required together with --resume",
    )
    args = parser.parse_args()
    if args.reason and not args.resume:
        parser.error("--reason is only valid with --resume")
    if args.resume and (not args.reason or not args.reason.strip()):
        parser.error("--resume requires a non-empty --reason")
    run(resume_reason=args.reason if args.resume else None)
