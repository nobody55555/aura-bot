"""Deterministic offline tests for the HIP-4 paper screener."""

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from live.hyperliquid_screener import (
    analyze_binary_outcome,
    analyze_book,
    filter_outcomes,
    list_price_binary_records,
    parse_outcome_metadata,
)


BTC_OUTCOME = {
    "outcome": 123,
    "name": "Recurring BTC",
    "description": (
        "class:priceBinary|underlying:BTC|expiry:20261123-0600|"
        "targetPrice:90000.00|period:1d|source:recorded"
    ),
    "sideSpecs": [{"name": "Yes"}, {"name": "No"}],
}

YES_PRICES = {
    "outcome": 123,
    "side": 0,
    "side_name": "Yes",
    "coin": "#1230",
    "midPx": "0.52",
}
NO_PRICES = {
    "outcome": 123,
    "side": 1,
    "side_name": "No",
    "coin": "#1231",
    "midPx": None,
}

YES_BOOK = {
    "coin": "#1230",
    "time": 1795352737058,
    "levels": [
        [
            {"px": "0.49", "sz": "2.0", "n": 1},
            {"px": "0.48", "sz": "3.0", "n": 1},
            {"px": "0.47", "sz": "100", "n": 1},
        ],
        [
            {"px": "0.51", "sz": "4.0", "n": 1},
            {"px": "0.52", "sz": "5.0", "n": 1},
        ],
    ],
}
NO_BOOK = {
    "coin": "#1231",
    "time": 1795352737058,
    "levels": [
        [{"px": "0.48", "sz": "10", "n": 1}],
        [{"px": "0.50", "sz": "8", "n": 1}],
    ],
}


def test_list_price_binary_records_screens_mixed_raw_catalog_without_parsing():
    binary = {
        **BTC_OUTCOME,
        "description": "  CLASS:PRICEBINARY|underlying:BTC  ",
    }
    nonbinary_free_text = {
        **BTC_OUTCOME,
        "outcome": 124,
        "description": "Free-form description without structured attributes.",
    }
    missing_description = {"outcome": 125, "name": "No description"}
    null_description = {"outcome": 126, "name": "Null description", "description": None}

    assert list_price_binary_records(
        [binary, nonbinary_free_text, missing_description, null_description, None, "x"]
    ) == [binary]


def test_list_price_binary_records_trims_and_casefolds_both_attribute_parts():
    spaced_class = {
        **BTC_OUTCOME,
        "description": " class : priceBinary |underlying:BTC",
    }

    assert list_price_binary_records([spaced_class]) == [spaced_class]


@pytest.mark.parametrize(
    "description",
    [
        "subclass:pricebinary",
        "class:pricebinaryExtended",
        "Free text mentions class: in the middle, but not the pricebinary value.",
    ],
)
def test_list_price_binary_records_rejects_substrings_and_non_attribute_text(
    description,
):
    nonbinary = {**BTC_OUTCOME, "description": description}

    assert list_price_binary_records([nonbinary]) == []


def test_parse_metadata_fields_and_exact_target_price():
    metadata = parse_outcome_metadata(BTC_OUTCOME)

    assert metadata.outcome_id == 123
    assert metadata.name == "Recurring BTC"
    assert metadata.market_class == "priceBinary"
    assert metadata.underlying == "BTC"
    assert metadata.expiry == "20261123-0600"
    assert metadata.expires_at == datetime(2026, 11, 23, 6, 0, tzinfo=timezone.utc)
    assert metadata.target_price == Decimal("90000.00")
    assert metadata.period == "1d"
    assert metadata.extra_attributes == (("source", "recorded"),)


def test_metadata_expiry_can_use_an_explicit_timezone_override():
    offset = timezone(timedelta(hours=2))
    metadata = parse_outcome_metadata(BTC_OUTCOME, expiry_timezone=offset)

    assert metadata.expires_at == datetime(2026, 11, 23, 6, 0, tzinfo=offset)


@pytest.mark.parametrize(
    ("expiry", "message"),
    [
        ("20261025-0230", "ambiguous"),
        ("20260329-0230", "does not exist"),
    ],
)
def test_naive_europe_zurich_expiry_rejects_ambiguous_or_nonexistent_dst_time(
    expiry, message
):
    outcome = {
        **BTC_OUTCOME,
        "description": f"class:priceBinary|underlying:BTC|expiry:{expiry}",
    }

    with pytest.raises(ValueError, match=message):
        parse_outcome_metadata(outcome, expiry_timezone=ZoneInfo("Europe/Zurich"))


def test_explicit_offset_expiry_remains_valid_during_dst_fallback():
    outcome = {
        **BTC_OUTCOME,
        "description": ("class:priceBinary|expiry:2026-10-25T02:30:00+02:00"),
    }

    metadata = parse_outcome_metadata(
        outcome, expiry_timezone=ZoneInfo("Europe/Zurich")
    )

    assert metadata.expires_at.astimezone(UTC) == datetime(
        2026, 10, 25, 0, 30, tzinfo=UTC
    )


def test_iso_expiry_accepts_lowercase_z_as_utc():
    metadata = parse_outcome_metadata(
        {
            **BTC_OUTCOME,
            "description": "class:priceBinary|expiry:2026-11-23T06:00:00z",
        }
    )

    assert metadata.expires_at == datetime(2026, 11, 23, 6, 0, tzinfo=timezone.utc)


def test_filter_by_underlying_and_remaining_time():
    btc = parse_outcome_metadata(BTC_OUTCOME, expiry_timezone=timezone.utc)
    eth = parse_outcome_metadata(
        {
            **BTC_OUTCOME,
            "outcome": 124,
            "description": "class:priceBinary|underlying:ETH|expiry:20261123-0600",
        },
        expiry_timezone=timezone.utc,
    )
    now = datetime(2026, 11, 23, 5, 0, tzinfo=timezone.utc)

    selected = filter_outcomes(
        [btc, eth],
        underlying=" btc ",
        min_remaining=timedelta(minutes=30),
        max_remaining=timedelta(hours=2),
        now=now,
    )

    assert selected == [btc]


def test_expiry_filter_requires_now_and_matching_timezone_awareness():
    utc_expiry = parse_outcome_metadata(BTC_OUTCOME)
    with pytest.raises(ValueError, match="now is required"):
        filter_outcomes([utc_expiry], min_remaining=timedelta(0))
    with pytest.raises(ValueError, match="must both be timezone-aware"):
        filter_outcomes(
            [utc_expiry],
            min_remaining=timedelta(0),
            now=datetime(2026, 11, 23, 5, 0),
        )


def test_max_remaining_excludes_expired_outcomes_unless_explicitly_included():
    expired = parse_outcome_metadata(
        {
            **BTC_OUTCOME,
            "description": "class:priceBinary|underlying:BTC|expiry:20261123-0400",
        }
    )
    now = datetime(2026, 11, 23, 5, 0, tzinfo=timezone.utc)

    assert filter_outcomes([expired], max_remaining=timedelta(hours=2), now=now) == []
    assert filter_outcomes(
        [expired],
        max_remaining=timedelta(hours=2),
        now=now,
        include_expired=True,
    ) == [expired]
    with pytest.raises(ValueError, match="include_expired must be a bool"):
        filter_outcomes(
            [expired], max_remaining=timedelta(hours=2), now=now, include_expired=1
        )


def test_now_alone_checks_expiry_and_drops_unparseable_expiries():
    future = parse_outcome_metadata(BTC_OUTCOME)
    expired = parse_outcome_metadata(
        {
            **BTC_OUTCOME,
            "outcome": 124,
            "description": "class:priceBinary|underlying:BTC|expiry:20261123-0400",
        }
    )
    unparseable = parse_outcome_metadata(
        {
            **BTC_OUTCOME,
            "outcome": 125,
            "description": "class:priceBinary|underlying:BTC|expiry:later-today",
        }
    )
    now = datetime(2026, 11, 23, 5, 0, tzinfo=timezone.utc)

    assert filter_outcomes([future, expired, unparseable], now=now) == [future]
    assert filter_outcomes(
        [future, expired, unparseable], now=now, include_expired=True
    ) == [future, expired]


@pytest.mark.parametrize("minimum", [timedelta(0), timedelta(minutes=5)])
def test_including_expired_rejects_nonnegative_minimum(minimum):
    outcome = parse_outcome_metadata(BTC_OUTCOME)
    now = datetime(2026, 11, 23, 5, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="conflicts with non-negative min_remaining"):
        filter_outcomes(
            [outcome],
            min_remaining=minimum,
            now=now,
            include_expired=True,
        )


def test_including_expired_allows_negative_minimum_as_a_lower_bound():
    expired = parse_outcome_metadata(
        {
            **BTC_OUTCOME,
            "description": "class:priceBinary|underlying:BTC|expiry:20261123-0400",
        }
    )
    now = datetime(2026, 11, 23, 5, 0, tzinfo=timezone.utc)

    assert filter_outcomes(
        [expired],
        min_remaining=-timedelta(hours=2),
        now=now,
        include_expired=True,
    ) == [expired]


def test_description_rejects_duplicates_malformed_fields_and_bad_target_price():
    with pytest.raises(ValueError, match="duplicate"):
        parse_outcome_metadata(
            {**BTC_OUTCOME, "description": "underlying:BTC|Underlying:ETH"}
        )
    with pytest.raises(ValueError, match="malformed"):
        parse_outcome_metadata({**BTC_OUTCOME, "description": "underlying:BTC|broken"})
    with pytest.raises(ValueError, match="targetPrice"):
        parse_outcome_metadata(
            {**BTC_OUTCOME, "description": "targetPrice:not-a-number"}
        )


def test_book_calculates_decimal_spread_and_top_n_notional_depth():
    metrics = analyze_book(YES_BOOK, depth_levels=2)

    assert metrics.book_available is True
    assert metrics.best_bid == Decimal("0.49")
    assert metrics.best_ask == Decimal("0.51")
    assert metrics.midpoint == Decimal("0.50")
    assert metrics.absolute_spread == Decimal("0.02")
    assert metrics.relative_spread == Decimal("0.04")
    assert metrics.bid_depth_notional == Decimal("2.42")
    assert metrics.ask_depth_notional == Decimal("4.64")
    assert metrics.is_empty_book is False
    assert metrics.is_one_sided is False
    assert metrics.is_crossed is False


def test_empty_one_sided_and_missing_books_are_distinguished():
    empty = analyze_book({"levels": [[], []]})
    one_sided = analyze_book({"levels": [[], [{"px": "0.39", "sz": "12.0", "n": 2}]]})
    missing = analyze_book(None)

    assert empty.is_empty_book is True
    assert empty.is_one_sided is False
    assert empty.bid_depth_notional == Decimal("0")
    assert empty.ask_depth_notional == Decimal("0")

    assert one_sided.is_empty_book is False
    assert one_sided.is_one_sided is True
    assert one_sided.best_bid is None
    assert one_sided.best_ask == Decimal("0.39")
    assert one_sided.ask_depth_notional == Decimal("4.6800")

    assert missing.book_available is False
    assert missing.is_empty_book is False
    assert missing.is_one_sided is False
    assert missing.bid_depth_notional is None
    assert missing.ask_depth_notional is None


def test_binary_probabilities_prefer_midpx_and_fall_back_to_book_midpoint():
    market = analyze_binary_outcome(
        BTC_OUTCOME,
        [YES_PRICES, NO_PRICES],
        books={0: YES_BOOK, 1: NO_BOOK},
        depth_levels=2,
    )

    assert market.yes.implied_probability == Decimal("0.52")
    assert market.yes.probability_source == "midPx"
    assert market.no.implied_probability == Decimal("0.49")
    assert market.no.probability_source == "best_bid_ask_mid"
    assert market.parity_delta == Decimal("0.01")
    assert market.overround == Decimal("0.01")
    assert market.executable_overround == Decimal("0.01")
    assert market.metadata.underlying == "BTC"


def test_executable_overround_uses_best_asks_not_mid_probabilities():
    no_prices = {**NO_PRICES, "midPx": "0.49"}
    no_book = {
        "levels": [
            [{"px": "0.48", "sz": "8", "n": 1}],
            [{"px": "0.55", "sz": "8", "n": 1}],
        ]
    }
    market = analyze_binary_outcome(
        BTC_OUTCOME,
        [YES_PRICES, no_prices],
        books={0: YES_BOOK, 1: no_book},
    )

    assert market.parity_delta == Decimal("0.01")
    assert market.executable_overround == Decimal("0.06")


def test_executable_overround_is_unknown_if_either_ask_is_missing():
    no_bid_only = {"levels": [[{"px": "0.48", "sz": "8", "n": 1}], []]}
    market = analyze_binary_outcome(
        BTC_OUTCOME,
        [YES_PRICES, NO_PRICES],
        books={0: YES_BOOK, 1: no_bid_only},
    )

    assert market.executable_overround is None


def test_probability_uses_exact_api_string_even_for_tight_spread():
    book = {
        "levels": [
            [{"px": "0.499", "sz": "1", "n": 1}],
            [{"px": "0.501", "sz": "1", "n": 1}],
        ]
    }
    metrics = analyze_book(book)

    assert metrics.absolute_spread == Decimal("0.002")
    assert metrics.midpoint == Decimal("0.500")
    assert metrics.relative_spread == Decimal("0.004")


def test_binary_market_rejects_wrong_class_mismatched_outcome_and_bad_sides():
    with pytest.raises(ValueError, match="priceBinary"):
        analyze_binary_outcome(
            {**BTC_OUTCOME, "description": "class:other"}, [YES_PRICES, NO_PRICES]
        )
    with pytest.raises(ValueError, match="priceBinary"):
        analyze_binary_outcome(
            {**BTC_OUTCOME, "description": "underlying:BTC|expiry:20261123-0600"},
            [YES_PRICES, NO_PRICES],
        )
    with pytest.raises(ValueError, match="does not match"):
        analyze_binary_outcome(BTC_OUTCOME, [YES_PRICES, {**NO_PRICES, "outcome": 999}])
    with pytest.raises(ValueError, match="exactly two"):
        analyze_binary_outcome(BTC_OUTCOME, [YES_PRICES])


def test_books_must_be_a_mapping():
    with pytest.raises(ValueError, match="books must be a mapping"):
        analyze_binary_outcome(BTC_OUTCOME, [YES_PRICES, NO_PRICES], books=[0, 1])


def test_books_value_type_is_rejected_transitively():
    with pytest.raises(ValueError, match="book must be a mapping or None"):
        analyze_binary_outcome(
            BTC_OUTCOME, [YES_PRICES, NO_PRICES], books={0: "not a mapping"}
        )


def test_books_allow_none_but_reject_non_mapping_sibling():
    market = analyze_binary_outcome(
        BTC_OUTCOME, [YES_PRICES, NO_PRICES], books={0: None}
    )
    assert market.yes.book.book_available is False

    with pytest.raises(ValueError, match="book must be a mapping or None"):
        analyze_binary_outcome(
            BTC_OUTCOME,
            [YES_PRICES, NO_PRICES],
            books={0: None, 1: "kein mapping"},
        )


def test_book_rejects_float_prices_nonfinite_and_invalid_levels():
    with pytest.raises(ValueError, match="decimal string or Decimal"):
        analyze_book({"levels": [[{"px": 0.5, "sz": "1"}], []]})
    with pytest.raises(ValueError, match="finite"):
        analyze_book({"levels": [[{"px": "NaN", "sz": "1"}], []]})
    with pytest.raises(ValueError, match="must be positive"):
        analyze_book({"levels": [[{"px": "0.5", "sz": "-1"}], []]})
    with pytest.raises(ValueError, match="depth_levels"):
        analyze_book({"levels": [[], []]}, depth_levels=0)
    with pytest.raises(ValueError, match="book must be a mapping or None"):
        analyze_book("not a mapping")


@pytest.mark.parametrize("side_index", [0, 1], ids=["bid", "ask"])
def test_book_rejects_zero_size_levels_on_both_sides(side_index):
    levels = [[], []]
    levels[side_index] = [{"px": "0.5", "sz": "0"}]

    with pytest.raises(ValueError, match="must be positive"):
        analyze_book({"levels": levels})


@pytest.mark.parametrize(
    "levels",
    [
        [[{"px": "0.48", "sz": "1"}, {"px": "0.49", "sz": "1"}], []],
        [[{"px": "0.49", "sz": "1"}, {"px": "0.49", "sz": "1"}], []],
        [[], [{"px": "0.52", "sz": "1"}, {"px": "0.51", "sz": "1"}]],
        [[], [{"px": "0.51", "sz": "1"}, {"px": "0.51", "sz": "1"}]],
    ],
)
def test_book_rejects_levels_not_ordered_best_first(levels):
    with pytest.raises(ValueError, match="ordered best-first"):
        analyze_book({"levels": levels})


def test_crossed_book_is_flagged_and_spread_remains_signed():
    metrics = analyze_book(
        {
            "levels": [
                [{"px": "0.51", "sz": "1", "n": 1}],
                [{"px": "0.49", "sz": "1", "n": 1}],
            ]
        }
    )

    assert metrics.absolute_spread == Decimal("-0.02")
    assert metrics.is_crossed is True


def test_zero_and_one_are_valid_midpx_but_zero_orderbook_price_is_rejected():
    market = analyze_binary_outcome(
        BTC_OUTCOME,
        [
            {**YES_PRICES, "midPx": "0.00"},
            {**NO_PRICES, "midPx": "1.00"},
        ],
    )

    assert market.yes.implied_probability == Decimal("0.00")
    assert market.no.implied_probability == Decimal("1.00")
    assert market.parity_delta == Decimal("0.00")

    with pytest.raises(ValueError, match="greater than 0 and at most 1"):
        analyze_book({"levels": [[{"px": "0.00", "sz": "1"}], []]})


def test_expired_or_unparseable_expiry_is_excluded_from_time_filter():
    expired = parse_outcome_metadata(
        {
            **BTC_OUTCOME,
            "description": "class:priceBinary|underlying:BTC|expiry:20261123-0400",
        },
        expiry_timezone=timezone.utc,
    )
    unparseable = parse_outcome_metadata(
        {
            **BTC_OUTCOME,
            "outcome": 125,
            "description": "class:priceBinary|underlying:BTC|expiry:later-today",
        },
        expiry_timezone=timezone.utc,
    )
    now = datetime(2026, 11, 23, 5, 0, tzinfo=timezone.utc)

    assert (
        filter_outcomes([expired, unparseable], min_remaining=timedelta(0), now=now)
        == []
    )


def test_missing_px_or_sz_is_rejected_instead_of_being_guessed():
    with pytest.raises(ValueError, match="bids\\[0\\]\\.px"):
        analyze_book({"levels": [[{"sz": "1"}], []]})
    with pytest.raises(ValueError, match="asks\\[0\\]\\.sz"):
        analyze_book({"levels": [[], [{"px": "0.5"}]]})


def test_missing_prices_leave_probability_and_parity_unknown():
    market = analyze_binary_outcome(
        BTC_OUTCOME,
        [{**YES_PRICES, "midPx": None}, NO_PRICES],
        books={0: {"levels": [[], []]}, 1: {"levels": [[], []]}},
    )

    assert market.yes.implied_probability is None
    assert market.no.implied_probability is None
    assert market.parity_delta is None


def test_negative_level_price_and_nonfinite_probability_are_rejected():
    with pytest.raises(ValueError, match="greater than 0 and at most 1"):
        analyze_book({"levels": [[{"px": "-0.01", "sz": "1"}], []]})
    with pytest.raises(ValueError, match="finite"):
        analyze_binary_outcome(
            BTC_OUTCOME,
            [{**YES_PRICES, "midPx": "Infinity"}, NO_PRICES],
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        analyze_binary_outcome(
            BTC_OUTCOME,
            [{**YES_PRICES, "midPx": "-0.01"}, NO_PRICES],
        )
