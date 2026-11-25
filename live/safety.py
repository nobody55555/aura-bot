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
import math
import numbers
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from live.db.repository import OrderRepository
from live.exceptions import SystemHalted, RiskLimitExceeded
from live.models import Order, OrderState

logger = logging.getLogger(__name__)

_MIN_PLAUSIBLE_EXCHANGE_TIME_MS = 946684800000  # 2000-01-01 UTC
_MAX_PLAUSIBLE_EXCHANGE_TIME_MS = 4102444800000  # 2100-01-01 UTC
_UNSET = object()


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
        daily_loss_limit: float = 0.02,
        max_drawdown: float = 0.05,
        alert_manager: Optional[Any] = None,
    ):
        self.repo = repository
        if not math.isfinite(max_stale_seconds) or max_stale_seconds <= 0:
            raise ValueError("max_stale_seconds must be finite and positive")
        if not math.isfinite(max_clock_drift_seconds) or max_clock_drift_seconds <= 0:
            raise ValueError("max_clock_drift_seconds must be finite and positive")
        if not math.isfinite(daily_loss_limit) or not 0 < daily_loss_limit <= 0.02:
            raise ValueError("daily_loss_limit must be finite and in (0, 0.02]")
        if not math.isfinite(max_drawdown) or not 0 < max_drawdown <= 0.05:
            raise ValueError("max_drawdown must be finite and in (0, 0.05]")
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
        self._halted = False
        self._halt_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def is_halted(self) -> bool:
        if self._halted:
            return True
        try:
            halted = self.repo.get_system_state("kill_switch", "0") == "1"
        except Exception as e:
            self._halted = True
            self._halt_reason = f"Cannot verify persisted kill-switch state: {e}"
            logger.critical("Fail-closed: %s", self._halt_reason)
            return True
        if halted:
            self._halted = True
            try:
                self._halt_reason = self.repo.get_system_state("kill_switch_reason")
            except Exception:
                self._halt_reason = None
        return halted

    def get_halt_reason(self) -> str:
        if self._halt_reason:
            return self._halt_reason
        try:
            reason = self.repo.get_system_state("kill_switch_reason")
        except Exception as e:
            return f"Reason unavailable because persistence failed: {e}"
        return reason or "Persisted halt reason unavailable"

    def halt(self, reason: str) -> None:
        # Close the in-process gate before touching persistence. If the DB write
        # fails, this process must still be unable to reach signal/order logic.
        self._halted = True
        self._halt_reason = str(reason)
        logger.critical("SAFETY HALT: %s", reason)
        self.repo.set_system_state("kill_switch", "1")
        self.repo.set_system_state("kill_switch_reason", str(reason))
        self.repo.record_risk_event("HALT", str(reason))
        if self.alerts is not None:
            try:
                self.alerts.halt(reason)
            except Exception as e:
                logger.error("Failed to send halt alert: %s", e)

    def resume(
        self,
        reason: str,
        *,
        equity: Any,
        exchange_required: bool,
        exchange_time_ms: Any = _UNSET,
        operator_approved: bool = False,
    ) -> None:
        """Resume only after an explicit operator approval and valid preflight."""
        if not operator_approved:
            raise SystemHalted("Resume requires explicit operator approval")
        if not isinstance(reason, str) or not reason.strip():
            raise SystemHalted("Resume requires a non-empty operator reason")
        if not self.is_halted():
            raise SystemHalted("There is no active halt to resume")

        halt_reason = self.get_halt_reason()
        try:
            self.update_equity(equity)
            if exchange_required:
                self.check_clock_drift(
                    None if exchange_time_ms is _UNSET else exchange_time_ms
                )
            elif exchange_time_ms is not _UNSET:
                self.check_clock_drift(exchange_time_ms)
        except Exception as e:
            denial = f"Resume denied; active halt={halt_reason}; preflight failed: {e}"
            try:
                self.repo.record_risk_event("RESUME_DENIED", denial)
            except Exception as persist_error:
                logger.critical("Could not persist resume denial: %s", persist_error)
            if not self._halted:
                self.halt(denial)
            raise SystemHalted(denial) from e

        approval = (
            f"Operator approval: {reason.strip()}; cleared halt: {halt_reason}; "
            f"equity={float(equity)}; exchange_time_checked={exchange_required or exchange_time_ms is not _UNSET}"
        )
        # Record approval before clearing the persistent gate. A failed audit
        # write therefore leaves the halt active.
        self.repo.record_risk_event("RESUME", approval)
        self.repo.set_system_state("kill_switch", "0")
        self._halted = False
        self._halt_reason = None
        logger.warning("SAFETY RESUME: %s", approval)

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
            self.halt(f"No market data timestamp for {symbol}")
            raise SystemHalted(f"No market data timestamp for {symbol}")
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
        if (
            isinstance(exchange_time_ms, bool)
            or not isinstance(exchange_time_ms, numbers.Real)
        ):
            reason = f"Exchange time is not numeric: {exchange_time_ms!r}"
            self.halt(reason)
            raise SystemHalted(reason)
        try:
            exchange_ms = float(exchange_time_ms)
        except (OverflowError, TypeError, ValueError):
            reason = f"Exchange time is not finite or plausible: {exchange_time_ms!r}"
            self.halt(reason)
            raise SystemHalted(reason)
        if (
            not math.isfinite(exchange_ms)
            or not _MIN_PLAUSIBLE_EXCHANGE_TIME_MS
            <= exchange_ms
            <= _MAX_PLAUSIBLE_EXCHANGE_TIME_MS
        ):
            reason = f"Exchange time is not finite or plausible: {exchange_time_ms!r}"
            self.halt(reason)
            raise SystemHalted(reason)
        local_ms = time.time() * 1000.0
        drift_s = abs(local_ms - exchange_ms) / 1000.0
        if drift_s > self.max_clock_drift_seconds:
            reason = f"Clock drift {drift_s:.2f}s exceeds limit {self.max_clock_drift_seconds}s"
            self.halt(reason)
            raise SystemHalted("Clock drift too large")

    # ------------------------------------------------------------------
    # Equity / drawdown / daily loss
    # ------------------------------------------------------------------

    def update_equity(self, equity: float) -> None:
        """Call periodically with current total equity."""
        try:
            if isinstance(equity, bool):
                raise ValueError("boolean is not an equity value")
            equity = float(equity)
        except (TypeError, ValueError, OverflowError):
            self.halt(f"Invalid or unknown equity value: {equity!r}")
            raise RiskLimitExceeded("Equity unavailable or invalid")
        if not math.isfinite(equity) or equity <= 0:
            self.halt(f"Invalid equity value: {equity!r}")
            raise RiskLimitExceeded("Equity unavailable or invalid")

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
            if daily_ret <= -self.daily_loss_limit:
                self.halt(
                    f"Daily loss limit breached: {daily_ret:.2%} < -{self.daily_loss_limit:.2%}"
                )
                raise RiskLimitExceeded("Daily loss limit")

        # Max drawdown from peak
        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
            if dd >= self.max_drawdown:
                self.halt(
                    f"Max drawdown breached: {dd:.2%} > {self.max_drawdown:.2%}"
                )
                raise RiskLimitExceeded("Max drawdown")

    def restore_equity_state(self) -> None:
        """Load peak / day-start from DB after restart."""
        peak = self.repo.get_system_state("peak_equity")
        day_eq = self.repo.get_system_state("day_start_equity")
        current_day = self.repo.get_system_state("current_day")
        persisted = (peak, day_eq, current_day)
        if any(value is not None for value in persisted) and not all(
            value is not None for value in persisted
        ):
            self.halt("Incomplete persisted equity state")
            return
        if peak is None:
            return  # Fresh database; the first valid equity sample initializes state.

        try:
            peak_value = float(peak)
            day_value = float(day_eq)
            datetime.strptime(current_day, "%Y-%m-%d")
            if not math.isfinite(peak_value) or peak_value <= 0:
                raise ValueError("peak_equity must be finite and positive")
            if not math.isfinite(day_value) or day_value <= 0:
                raise ValueError("day_start_equity must be finite and positive")
        except (TypeError, ValueError) as e:
            self.halt(f"Invalid persisted equity state: {e}")
            return

        self._peak_equity = peak_value
        self._day_start_equity = day_value
        self._current_day = current_day


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
            if self.safety:
                self.safety.halt(f"Unresolved order reconciliation discrepancies: {summary}")

        return summary

    def reconcile_on_startup(self, symbol: Optional[str] = None) -> dict[str, Any]:
        """Full startup reconciliation."""
        logger.info("Running startup reconciliation…")
        if self.safety:
            self.safety.restore_equity_state()
        return self.reconcile_open_orders(symbol)
