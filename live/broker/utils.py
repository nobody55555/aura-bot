"""
Helpers for the live CCXT broker:
- precision rounding
- minimum notional / amount checks
- retry decorator configuration
"""

from __future__ import annotations

from typing import Any, Optional

from live.exceptions import InvalidOrder


def validate_and_round(
    exchange: Any,
    symbol: str,
    amount: float,
    price: Optional[float] = None,
    stop_price: Optional[float] = None,
) -> tuple[float, Optional[float], Optional[float]]:
    """
    Round amount/price according to exchange filters and
    enforce minimum amount / notional limits.

    Returns (rounded_amount, rounded_price, rounded_stop_price).
    Raises InvalidOrder if the order would be rejected by the exchange.
    """
    markets = exchange.markets
    if symbol not in markets:
        # Try to load if not yet cached
        exchange.load_markets()
        markets = exchange.markets

    market = markets.get(symbol)
    if not market:
        raise InvalidOrder(f"Unknown symbol: {symbol}")

    # --- Amount precision & min ---
    amount = float(exchange.amount_to_precision(symbol, amount))
    min_amount = market.get("limits", {}).get("amount", {}).get("min")
    if min_amount is not None and amount < min_amount:
        raise InvalidOrder(
            f"Amount {amount} below minimum {min_amount} for {symbol}"
        )

    # --- Price precision ---
    rounded_price = None
    if price is not None:
        rounded_price = float(exchange.price_to_precision(symbol, price))

    rounded_stop = None
    if stop_price is not None:
        rounded_stop = float(exchange.price_to_precision(symbol, stop_price))

    # --- Minimum notional (cost) ---
    # Use the most relevant price for the check
    check_price = rounded_price or rounded_stop
    if check_price is None:
        # For pure market orders we approximate with last ticker
        try:
            ticker = exchange.fetch_ticker(symbol)
            check_price = float(ticker.get("last") or ticker.get("close") or 0)
        except Exception:
            check_price = 0.0

    min_cost = market.get("limits", {}).get("cost", {}).get("min")
    if min_cost is not None and check_price > 0:
        notional = amount * check_price
        if notional < min_cost:
            raise InvalidOrder(
                f"Notional {notional:.4f} below minimum {min_cost} for {symbol}"
            )

    return amount, rounded_price, rounded_stop


def is_duplicate_client_order_error(exc: Exception) -> bool:
    """
    Detect exchange-specific "duplicate clientOrderId" errors.
    Binance often returns code -2010 or messages containing "duplicate".
    """
    msg = str(exc).lower()
    if "duplicate" in msg and ("client" in msg or "order id" in msg):
        return True
    # Binance specific
    if hasattr(exc, "code") and exc.code in (-2010, -1013):
        return True
    return False
