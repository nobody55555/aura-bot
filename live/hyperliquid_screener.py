"""Pure, offline analysis helpers for binary HIP-4 price markets.

This module accepts recorded API-shaped snapshots and performs no network,
wallet, order, or trading-runtime operations. Numeric API strings are parsed
with :class:`decimal.Decimal` so spread and depth calculations stay exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Sequence

_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWO = Decimal("2")
_EXPIRY_PATTERN = re.compile(r"(?P<date>\d{8})-(?P<time>\d{4})\Z")


@dataclass(frozen=True, slots=True)
class OutcomeMetadata:
    """Typed view of an outcome record and its pipe-delimited description."""

    outcome_id: int
    name: str
    market_class: str | None
    underlying: str | None
    expiry: str | None
    expires_at: datetime | None
    target_price: Decimal | None
    period: str | None
    extra_attributes: tuple[tuple[str, str], ...]
    raw_description: str


@dataclass(frozen=True, slots=True)
class BookMetrics:
    """Top-of-book and top-N notional depth for one outcome side."""

    book_available: bool
    best_bid: Decimal | None
    best_ask: Decimal | None
    midpoint: Decimal | None
    absolute_spread: Decimal | None
    relative_spread: Decimal | None
    bid_depth_notional: Decimal | None
    ask_depth_notional: Decimal | None
    is_empty_book: bool
    is_one_sided: bool
    is_crossed: bool
    depth_levels: int


@dataclass(frozen=True, slots=True)
class SideMetrics:
    """Probability and order-book metrics for side 0 (Yes) or side 1 (No)."""

    side: int
    side_name: str | None
    coin: str | None
    implied_probability: Decimal | None
    probability_source: str | None
    book: BookMetrics


@dataclass(frozen=True, slots=True)
class BinaryMarketMetrics:
    """Combined Yes/No metrics; parity delta is Yes + No - 1."""

    metadata: OutcomeMetadata
    yes: SideMetrics
    no: SideMetrics
    parity_delta: Decimal | None
    executable_overround: Decimal | None

    @property
    def overround(self) -> Decimal | None:
        """Signed Yes/No parity deviation (positive means sum exceeds one)."""
        return self.parity_delta


def _decimal(value: Any, field_name: str) -> Decimal:
    """Parse a decimal string (or Decimal) without routing through float."""
    if isinstance(value, bool) or not isinstance(value, (str, Decimal)):
        raise ValueError(f"{field_name} must be a decimal string or Decimal")
    try:
        result = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} is not a valid decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return result


def _parse_expiry(value: str | None, expiry_timezone: tzinfo) -> datetime | None:
    """Parse supported expiry forms; naive timestamps use the supplied timezone."""
    if not value:
        return None

    match = _EXPIRY_PATTERN.fullmatch(value)
    parsed: datetime | None = None
    if match:
        try:
            parsed = datetime.strptime(value, "%Y%m%d-%H%M")
        except ValueError:
            return None
    else:
        iso_value = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
        try:
            parsed = datetime.fromisoformat(iso_value)
        except ValueError:
            return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=expiry_timezone)
    return parsed


def list_price_binary_records(
    outcomes: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return raw records with an exact pipe-delimited class marker.

    This is deliberately a string-only catalog pre-filter: it checks each
    pipe-delimited key/value pair without constructing parsed metadata.
    """
    selected: list[Mapping[str, Any]] = []
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            continue
        description = outcome.get("description")
        if not isinstance(description, str):
            continue
        for part in description.split("|"):
            key, separator, value = part.partition(":")
            if (
                separator
                and key.strip().casefold() == "class"
                and value.strip().casefold() == "pricebinary"
            ):
                selected.append(outcome)
                break
    return selected


def parse_outcome_metadata(
    outcome: Mapping[str, Any], *, expiry_timezone: tzinfo = timezone.utc
) -> OutcomeMetadata:
    """Parse API outcome metadata into stable fields.

    ``expiry`` is retained verbatim. The common ``YYYYMMDD-HHMM`` form is also
    exposed as ``expires_at``. The common ``YYYYMMDD-HHMM`` format defaults
    to UTC; pass another explicit timezone only when the snapshot convention
    is known to differ.
    """
    if not isinstance(outcome, Mapping):
        raise ValueError("outcome must be a mapping")
    outcome_id = outcome.get("outcome")
    if (
        not isinstance(outcome_id, int)
        or isinstance(outcome_id, bool)
        or outcome_id < 0
    ):
        raise ValueError("outcome must be a non-negative integer")

    name = outcome.get("name", "")
    if not isinstance(name, str):
        raise ValueError("outcome name must be a string")
    description = outcome.get("description", "")
    if description is None:
        description = ""
    if not isinstance(description, str):
        raise ValueError("outcome description must be a string")

    attributes: dict[str, str] = {}
    for part in description.split("|"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"malformed description attribute: {part!r}")
        key, value = (piece.strip() for piece in part.split(":", 1))
        normalized_key = key.casefold()
        if not normalized_key:
            raise ValueError("description attribute key must not be empty")
        if normalized_key in attributes:
            raise ValueError(f"duplicate description attribute: {key!r}")
        attributes[normalized_key] = value

    target_price = None
    if "targetprice" in attributes:
        target_price = _decimal(attributes["targetprice"], "targetPrice")

    expiry = attributes.get("expiry") or None
    standard_fields = {"class", "underlying", "expiry", "targetprice", "period"}
    extras = tuple(
        sorted(
            (key, value)
            for key, value in attributes.items()
            if key not in standard_fields
        )
    )
    return OutcomeMetadata(
        outcome_id=outcome_id,
        name=name,
        market_class=attributes.get("class") or None,
        underlying=attributes.get("underlying") or None,
        expiry=expiry,
        expires_at=_parse_expiry(expiry, expiry_timezone),
        target_price=target_price,
        period=attributes.get("period") or None,
        extra_attributes=extras,
        raw_description=description,
    )


def filter_outcomes(
    outcomes: Iterable[OutcomeMetadata],
    *,
    underlying: str | None = None,
    min_remaining: timedelta | None = None,
    max_remaining: timedelta | None = None,
    now: datetime | None = None,
    include_expired: bool = False,
) -> list[OutcomeMetadata]:
    """Filter parsed metadata by underlying and optional time-to-expiry bounds.

    For expiry filtering, ``now`` must use the same timezone awareness and
    timezone convention as each metadata object's ``expires_at``. Outcomes
    with an unparseable or absent expiry are excluded when checking time.
    Supplying ``now`` alone also checks expiry: expired outcomes and outcomes
    without a parseable expiry are excluded by default, even without remaining-
    time bounds. Expired outcomes can be included with ``include_expired=True``;
    because a non-negative ``min_remaining`` excludes expired outcomes, that
    combination is rejected as contradictory. A negative minimum may still
    impose a lower bound on how far in the past an included expiry can be.
    """
    if underlying is not None and not underlying.strip():
        raise ValueError("underlying must be non-empty when provided")
    if (
        min_remaining is not None
        and max_remaining is not None
        and min_remaining > max_remaining
    ):
        raise ValueError("min_remaining must not exceed max_remaining")
    if not isinstance(include_expired, bool):
        raise ValueError("include_expired must be a bool")
    if include_expired and min_remaining is not None and min_remaining >= timedelta(0):
        raise ValueError(
            "include_expired=True conflicts with non-negative min_remaining"
        )
    uses_expiry = min_remaining is not None or max_remaining is not None
    if uses_expiry and now is None:
        raise ValueError("now is required when filtering by remaining time")
    if now is not None and not isinstance(now, datetime):
        raise ValueError("now must be a datetime")
    checks_expiry = uses_expiry or now is not None
    if checks_expiry and now is None:
        raise ValueError("now is required when checking expiry")

    selected: list[OutcomeMetadata] = []
    for outcome in outcomes:
        if (
            underlying is not None
            and (outcome.underlying or "").casefold() != underlying.strip().casefold()
        ):
            continue
        if checks_expiry:
            expiry = outcome.expires_at
            if expiry is None:
                continue
            if now is None:
                raise ValueError("now is required when checking expiry")
            expiry_aware = expiry.utcoffset() is not None
            now_aware = now.utcoffset() is not None
            if expiry_aware != now_aware:
                raise ValueError(
                    "now and expires_at must both be timezone-aware or both naive"
                )
            remaining = expiry - now
            if not include_expired and remaining <= timedelta(0):
                continue
            if min_remaining is not None and remaining < min_remaining:
                continue
            if max_remaining is not None and remaining > max_remaining:
                continue
        selected.append(outcome)
    return selected


def analyze_book(
    book: Mapping[str, Any] | None, *, depth_levels: int = 5
) -> BookMetrics:
    """Calculate BBO spreads, flags, and top-N notional depth from an L2 snapshot.

    Hyperliquid L2 ``levels`` are interpreted as ``[bids, asks]`` and must be
    ordered best-first (bids descending, asks ascending); unsorted levels are
    rejected rather than silently misreporting top-N depth. ``relative_spread``
    is a ratio, not a percentage (multiply by 100 for percent).
    """
    if (
        not isinstance(depth_levels, int)
        or isinstance(depth_levels, bool)
        or depth_levels < 1
    ):
        raise ValueError("depth_levels must be a positive integer")
    if book is None:
        return BookMetrics(
            book_available=False,
            best_bid=None,
            best_ask=None,
            midpoint=None,
            absolute_spread=None,
            relative_spread=None,
            bid_depth_notional=None,
            ask_depth_notional=None,
            is_empty_book=False,
            is_one_sided=False,
            is_crossed=False,
            depth_levels=depth_levels,
        )

    if not isinstance(book, Mapping):
        raise ValueError("book must be a mapping or None")
    levels = book.get("levels")
    if (
        not isinstance(levels, list)
        or len(levels) != 2
        or any(not isinstance(side, list) for side in levels)
    ):
        raise ValueError("book levels must contain bid and ask lists")

    parsed_sides: list[list[tuple[Decimal, Decimal]]] = []
    for side_name, side_levels in zip(("bids", "asks"), levels):
        parsed: list[tuple[Decimal, Decimal]] = []
        for index, level in enumerate(side_levels):
            if not isinstance(level, Mapping):
                raise ValueError(f"{side_name}[{index}] must be a mapping")
            price = _decimal(level.get("px"), f"{side_name}[{index}].px")
            size = _decimal(level.get("sz"), f"{side_name}[{index}].sz")
            if not _ZERO < price <= _ONE:
                raise ValueError(
                    f"{side_name}[{index}].px must be greater than 0 and at most 1"
                )
            if size < _ZERO:
                raise ValueError(f"{side_name}[{index}].sz must not be negative")
            parsed.append((price, size))
        parsed_sides.append(parsed)

    bids, asks = parsed_sides
    if any(previous[0] <= current[0] for previous, current in zip(bids, bids[1:])):
        raise ValueError("bids must be ordered best-first (descending price)")
    if any(previous[0] >= current[0] for previous, current in zip(asks, asks[1:])):
        raise ValueError("asks must be ordered best-first (ascending price)")
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    midpoint = (
        (best_bid + best_ask) / _TWO
        if best_bid is not None and best_ask is not None
        else None
    )
    spread = (
        best_ask - best_bid if best_bid is not None and best_ask is not None else None
    )
    relative_spread = (
        spread / midpoint if spread is not None and midpoint != _ZERO else None
    )
    bid_depth = sum((price * size for price, size in bids[:depth_levels]), _ZERO)
    ask_depth = sum((price * size for price, size in asks[:depth_levels]), _ZERO)
    has_bids = bool(bids)
    has_asks = bool(asks)
    return BookMetrics(
        book_available=True,
        best_bid=best_bid,
        best_ask=best_ask,
        midpoint=midpoint,
        absolute_spread=spread,
        relative_spread=relative_spread,
        bid_depth_notional=bid_depth,
        ask_depth_notional=ask_depth,
        is_empty_book=not has_bids and not has_asks,
        is_one_sided=has_bids != has_asks,
        is_crossed=spread is not None and spread < _ZERO,
        depth_levels=depth_levels,
    )


def _side_metrics(
    side: int,
    price_record: Mapping[str, Any],
    book: Mapping[str, Any] | None,
    *,
    depth_levels: int,
) -> SideMetrics:
    book_metrics = analyze_book(book, depth_levels=depth_levels)
    raw_mid = price_record.get("midPx")
    if raw_mid is not None:
        probability = _decimal(raw_mid, "midPx")
        if not _ZERO <= probability <= _ONE:
            raise ValueError("midPx must be between 0 and 1 for a binary contract")
        source = "midPx"
    elif book_metrics.midpoint is not None:
        probability = book_metrics.midpoint
        source = "best_bid_ask_mid"
    else:
        probability = None
        source = None

    side_name = price_record.get("side_name")
    coin = price_record.get("coin")
    if side_name is not None and not isinstance(side_name, str):
        raise ValueError("side_name must be a string when provided")
    if coin is not None and not isinstance(coin, str):
        raise ValueError("coin must be a string when provided")
    return SideMetrics(
        side=side,
        side_name=side_name,
        coin=coin,
        implied_probability=probability,
        probability_source=source,
        book=book_metrics,
    )


def analyze_binary_outcome(
    outcome: Mapping[str, Any],
    side_prices: Sequence[Mapping[str, Any]],
    books: Mapping[int, Mapping[str, Any] | None] | None = None,
    *,
    depth_levels: int = 5,
    expiry_timezone: tzinfo = timezone.utc,
) -> BinaryMarketMetrics:
    """Analyze one recorded binary outcome and its two side snapshots.

    ``side_prices`` uses the ``get_outcome_prices()`` record shape from
    ``live.hyperliquid_testnet`` (side 0 = Yes, side 1 = No). ``books`` maps
    those side indices to recorded ``get_outcome_book()`` snapshots. Missing
    books remain distinguishable from available-but-empty books.
    """
    metadata = parse_outcome_metadata(outcome, expiry_timezone=expiry_timezone)
    if (
        metadata.market_class is None
        or metadata.market_class.casefold() != "pricebinary"
    ):
        raise ValueError("analyze_binary_outcome requires a priceBinary outcome")
    if not isinstance(side_prices, Sequence) or isinstance(side_prices, (str, bytes)):
        raise ValueError("side_prices must contain one record for each binary side")
    if len(side_prices) != 2:
        raise ValueError("side_prices must contain exactly two side records")

    records: dict[int, Mapping[str, Any]] = {}
    for record in side_prices:
        if not isinstance(record, Mapping):
            raise ValueError("each side price record must be a mapping")
        side = record.get("side")
        if not isinstance(side, int) or isinstance(side, bool) or side not in (0, 1):
            raise ValueError("side price record side must be 0 or 1")
        if side in records:
            raise ValueError(f"duplicate side price record for side {side}")
        record_outcome = record.get("outcome")
        if record_outcome is not None and record_outcome != metadata.outcome_id:
            raise ValueError("side price record outcome does not match metadata")
        records[side] = record
    if set(records) != {0, 1}:
        raise ValueError("side price records must include side 0 and side 1")

    side_books = {} if books is None else books
    if not isinstance(side_books, Mapping):
        raise ValueError("books must be a mapping of side index to book snapshot")
    if any(
        not isinstance(side, int) or isinstance(side, bool) or side not in (0, 1)
        for side in side_books
    ):
        raise ValueError("books keys must be side 0 or side 1")
    yes = _side_metrics(0, records[0], side_books.get(0), depth_levels=depth_levels)
    no = _side_metrics(1, records[1], side_books.get(1), depth_levels=depth_levels)
    parity_delta = (
        yes.implied_probability + no.implied_probability - _ONE
        if yes.implied_probability is not None and no.implied_probability is not None
        else None
    )
    executable_overround = (
        yes.book.best_ask + no.book.best_ask - _ONE
        if yes.book.best_ask is not None and no.book.best_ask is not None
        else None
    )
    return BinaryMarketMetrics(
        metadata=metadata,
        yes=yes,
        no=no,
        parity_delta=parity_delta,
        executable_overround=executable_overround,
    )
