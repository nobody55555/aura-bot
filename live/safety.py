"""
Safety monitors and operational controls.

- Kill-switch (manual + automatic)
- Stale data detection
- Clock drift check
- Daily loss / max drawdown hard stops (complementing risk.py)
- Simple reconciliation helpers
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from live.db.repository import OrderRepository
from live.exceptions import SystemHalted, RiskLimitExceeded
from live.models import Order, OrderState

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """
    Central safety component.

    Designed to be called:
    - on startup
    - periodically (e.g. every poll cycle)
    - before every new order
    """

    def __init__(
        self,
        repository: OrderRepository,
        max_stale_seconds: float = 120.0,
        max_clock_drift_seconds: float = 3.0,
        daily_loss_limit: float = 0.03,
        max_drawdown: float = 0.10,
        alert_manager: Optional[Any] = None,
    ):
        self.repo = repository
        self.max_stale_seconds = max_stale_seconds
        self.max_clock_drift_seconds = max_clock_drift_seconds
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown = max_drawdown
        self.alerts = alert_manager  # optional AlertManager

        # Runtime state (also mirrored to DB for crash recovery)
        self._last_bar_ts: dict[str, float] = {}          # symbol → unix ts
        self._peak_equity: float = 0.0
        self._day_start_equity: float = 0.0
        self._current_day: Optional[str] = None

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def is_halted(self) -> bool:
        return self.repo.get_system_state("kill_switch", "0") == "1"

    def halt(self, reason: str) -> None:
        self.repo.set_system_state("kill_switch", "1")
        self.repo.record_risk_event("HALT", reason)
        logger.critical("SAFETY HALT: %s", reason)
        if self.alerts is not None:
            try:
                self.alerts.halt(reason)
            except Exception as e:
                logger.error("Failed to send halt alert: %s", e)

    def resume(self, reason: str = "manual") -> None:
        self.repo.set_system_state("kill_switch", "0")
        self.repo.record_risk_event("RESUME", reason)
        logger.warning("SAFETY RESUME: %s", reason)

    def require_not_halted(self) -> None:
        if self.is_halted():
            raise SystemHalted("Kill-switch is active")

    # ------------------------------------------------------------------
    # Stale data
    # ------------------------------------------------------------------

    def update_bar_timestamp(self, symbol: str, bar_ts: Optional[float] = None) -> None:
        """Call this whenever a new bar / ticker arrives."""
        # Use `is None` so that bar_ts=0 is accepted (tests / edge cases).
        self._last_bar_ts[symbol] = time.time() if bar_ts is None else float(bar_ts)

    def check_stale(self, symbol: str) -> None:
        last = self._last_bar_ts.get(symbol)
        if last is None:
            return  # no data yet – not considered stale
        age = time.time() - last
        if age > self.max_stale_seconds:
            self.halt(f"Stale data for {symbol}: {age:.0f}s > {self.max_stale_seconds}s")
            raise SystemHalted(f"Stale data for {symbol}")

    # ------------------------------------------------------------------
    # Clock drift
    # ------------------------------------------------------------------

    def check_clock_drift(self, exchange_time_ms: Optional[int]) -> None:
        """
        Compare local UTC time with exchange server time.
        exchange_time_ms comes from exchange.fetch_time() (milliseconds).
        """
        if exchange_time_ms is None:
            return
        local_ms = int(time.time() * 1000)
        drift_s = abs(local_ms - exchange_time_ms) / 1000.0
        if drift_s > self.max_clock_drift_seconds:
            self.halt(f"Clock drift {drift_s:.2f}s exceeds limit {self.max_clock_drift_seconds}s")
            raise SystemHalted("Clock drift too large")

    # ------------------------------------------------------------------
    # Equity / drawdown / daily loss
    # ------------------------------------------------------------------

    def update_equity(self, equity: float) -> None:
        """Call periodically with current total equity."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self._current_day != today:
            self._current_day = today
            self._day_start_equity = equity
            # Persist day start so we survive restarts
            self.repo.set_system_state("day_start_equity", str(equity))
            self.repo.set_system_state("current_day", today)

        if equity > self._peak_equity:
            self._peak_equity = equity
            self.repo.set_system_state("peak_equity", str(equity))

        # Daily loss
        if self._day_start_equity > 0:
            daily_ret = (equity - self._day_start_equity) / self._day_start_equity
            if daily_ret < -self.daily_loss_limit:
                self.halt(
                    f"Daily loss limit breached: {daily_ret:.2%} < -{self.daily_loss_limit:.2%}"
                )
                raise RiskLimitExceeded("Daily loss limit")

        # Max drawdown from peak
        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
            if dd > self.max_drawdown:
                self.halt(
                    f"Max drawdown breached: {dd:.2%} > {self.max_drawdown:.2%}"
                )
                raise RiskLimitExceeded("Max drawdown")

    def restore_equity_state(self) -> None:
        """Load peak / day-start from DB after restart."""
        peak = self.repo.get_system_state("peak_equity")
        if peak:
            try:
                self._peak_equity = float(peak)
            except ValueError:
                pass
        day_eq = self.repo.get_system_state("day_start_equity")
        if day_eq:
            try:
                self._day_start_equity = float(day_eq)
            except ValueError:
                pass
        self._current_day = self.repo.get_system_state("current_day")


class Reconciler:
    """
    Compare local DB state with broker (exchange or paper).

    Policy: Exchange is the source of truth.
    Local DB is updated to match; discrepancies are logged and can trigger halt.
    """

    def __init__(self, repository: OrderRepository, broker, safety: Optional[SafetyMonitor] = None):
        self.repo = repository
        self.broker = broker
        self.safety = safety

    def reconcile_open_orders(self, symbol: Optional[str] = None) -> dict[str, Any]:
        """
        Fetch open orders from broker and reconcile with local DB.
        Returns a summary of actions taken.
        """
        summary = {
            "local_only": [],
            "remote_only": [],
            "updated": [],
            "matched": 0,
        }

        try:
            remote_orders = self.broker.fetch_open_orders(symbol)
        except Exception as e:
            logger.error("Failed to fetch remote open orders: %s", e)
            if self.safety:
                self.safety.halt(f"Reconciliation failed: cannot fetch open orders ({e})")
            raise

        remote_by_cid = {o.client_order_id: o for o in remote_orders if o.client_order_id}
        local_open = self.repo.get_open_orders(symbol)
        local_by_cid = {o.client_order_id: o for o in local_open}

        # Local orders that are still "open" but missing on exchange
        for cid, local in local_by_cid.items():
            if cid not in remote_by_cid:
                # Could have been filled or canceled while we were down
                try:
                    synced = self.broker.fetch_order(cid, local.symbol)
                    self.repo.save_order(synced)
                    summary["updated"].append(cid)
                    logger.info("Reconciled local-only order %s → %s", cid, synced.status.value)
                except Exception:
                    local.status = OrderState.UNKNOWN
                    local.updated_at = datetime.now(timezone.utc)
                    self.repo.save_order(local)
                    summary["local_only"].append(cid)
                    logger.warning("Order %s exists locally but not on exchange → UNKNOWN", cid)

        # Remote orders we don't know about
        for cid, remote in remote_by_cid.items():
            if cid not in local_by_cid:
                self.repo.save_order(remote)
                summary["remote_only"].append(cid)
                logger.warning("Found unknown remote order %s – imported", cid)
            else:
                # Update local with latest remote state
                self.repo.save_order(remote)
                summary["matched"] += 1

        if summary["local_only"] or summary["remote_only"]:
            logger.warning("Reconciliation discrepancies: %s", summary)
            self.repo.record_risk_event("RECONCILE_DIFF", str(summary))

        return summary

    def reconcile_on_startup(self, symbol: Optional[str] = None) -> dict[str, Any]:
        """Full startup reconciliation."""
        logger.info("Running startup reconciliation…")
        if self.safety:
            self.safety.restore_equity_state()
        return self.reconcile_open_orders(symbol)
