"""
Typed exceptions for the live trading layer.

These allow the OrderManager to make clear decisions
(retry / abort / halt) without parsing string messages.
"""

from __future__ import annotations


class AuraLiveError(Exception):
    """Base exception for all live trading errors."""


class BrokerError(AuraLiveError):
    """Generic broker / exchange error."""


class InsufficientFunds(BrokerError):
    """Not enough balance to place the order."""


class OrderNotFound(BrokerError):
    """Requested order does not exist on the broker."""


class InvalidOrder(BrokerError):
    """Order rejected by the exchange (precision, limits, etc.)."""


class NetworkError(BrokerError):
    """Transient network / timeout / rate-limit problem."""


class DuplicateOrder(BrokerError):
    """clientOrderId already exists – treat as success if order is present."""


class RiskLimitExceeded(AuraLiveError):
    """Daily loss, max drawdown or position limit hit."""


class SystemHalted(AuraLiveError):
    """Kill-switch is active – no new orders allowed."""
