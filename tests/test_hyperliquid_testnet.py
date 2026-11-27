"""Tests for the fail-closed, read-only Hyperliquid Testnet client."""

from __future__ import annotations

from http.client import HTTPException

import pytest

import live.hyperliquid_testnet as hl

SAMPLE_OUTCOMES = [
    {
        "outcome": 123,
        "name": "Recurring",
        "description": "class:priceBinary|underlying:BTC|expiry:20261123-0600|targetPrice:90000|period:1d",
        "sideSpecs": [{"name": "Yes"}, {"name": "No"}],
    },
    {
        "outcome": 124,
        "name": "Recurring",
        "description": "class:priceBinary|underlying:ETH|expiry:20261123-0600|targetPrice:3000|period:1d",
        "sideSpecs": [{"name": "Yes"}, {"name": "No"}],
    },
]


@pytest.fixture(autouse=True)
def _set_testnet_environment_for_tests(monkeypatch):
    monkeypatch.setenv("HL_ENV", "testnet")


def test_default_endpoints_are_official_testnet_endpoints():
    client = hl.HyperliquidTestnetClient()
    assert hl.TESTNET_REST_URL == hl.hyperliquid_constants.TESTNET_API_URL
    assert client._rest_url == "https://api.hyperliquid-testnet.xyz"
    assert client._websocket_url == "wss://api.hyperliquid-testnet.xyz/ws"


@pytest.mark.parametrize(
    "rest_url",
    [
        None,
        "https://api.hyperliquid.xyz",
        "https://hyperliquid-testnet.xyz",
        "http://api.hyperliquid-testnet.xyz",
        "https://api.hyperliquid-testnet.xyz:443",
        "https://api.hyperliquid-testnet.xyz/",
        "https://api.hyperliquid-testnet.xyz/info",
    ],
)
def test_mainnet_or_noncanonical_rest_endpoint_is_rejected(rest_url):
    with pytest.raises(hl.TestnetOnlyError):
        hl.HyperliquidTestnetClient(rest_url=rest_url)


@pytest.mark.parametrize("websocket_url", [None, "wss://api.hyperliquid.xyz/ws"])
def test_mainnet_or_noncanonical_websocket_endpoint_is_rejected(websocket_url):
    with pytest.raises(hl.TestnetOnlyError):
        hl.HyperliquidTestnetClient(websocket_url=websocket_url)


def test_non_testnet_environment_fails_closed(monkeypatch):
    monkeypatch.setenv("HL_ENV", "mainnet")
    with pytest.raises(hl.TestnetOnlyError, match="HL_ENV"):
        hl.HyperliquidTestnetClient()


def test_missing_environment_fails_closed(monkeypatch):
    monkeypatch.delenv("HL_ENV")
    with pytest.raises(hl.TestnetOnlyError, match="explicitly set"):
        hl.HyperliquidTestnetClient()


def test_environment_is_revalidated_before_every_request(monkeypatch):
    client = hl.HyperliquidTestnetClient()
    calls = []
    monkeypatch.setattr(
        hl,
        "_post_info_json",
        lambda url, payload, timeout: (
            calls.append((url, payload, timeout)) or {"outcomes": []}
        ),
    )
    monkeypatch.setenv("HL_ENV", "mainnet")

    with pytest.raises(hl.TestnetOnlyError, match="HL_ENV"):
        client.get_outcome_metadata()
    assert calls == []


def test_mutated_rest_url_is_rejected_before_transport(monkeypatch):
    client = hl.HyperliquidTestnetClient()
    calls = []
    monkeypatch.setattr(
        hl,
        "_post_info_json",
        lambda url, payload, timeout: calls.append(url) or {"outcomes": []},
    )
    client._rest_url = "https://api.hyperliquid.xyz"

    with pytest.raises(hl.TestnetOnlyError):
        client.get_outcome_metadata()
    assert calls == []


def test_changed_sdk_testnet_constant_fails_closed(monkeypatch):
    client = hl.HyperliquidTestnetClient()
    monkeypatch.setattr(
        hl.hyperliquid_constants,
        "TESTNET_API_URL",
        "https://api.hyperliquid.xyz",
    )

    with pytest.raises(hl.TestnetOnlyError):
        client.get_outcome_metadata()


def test_outcome_metadata_uses_only_the_read_only_info_query(monkeypatch):
    calls = []

    def fake_post(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {"outcomes": SAMPLE_OUTCOMES}

    monkeypatch.setattr(hl, "_post_info_json", fake_post)
    client = hl.HyperliquidTestnetClient(timeout=3)

    assert client.get_outcome_metadata() == SAMPLE_OUTCOMES
    assert calls == [
        (
            "https://api.hyperliquid-testnet.xyz/info",
            {"type": "outcomeMeta"},
            3.0,
        )
    ]


def test_find_outcomes_filters_locally_by_underlying_and_text(monkeypatch):
    client = hl.HyperliquidTestnetClient()
    calls = []
    monkeypatch.setattr(
        client,
        "get_outcome_metadata",
        lambda: calls.append("metadata") or SAMPLE_OUTCOMES,
    )

    found = client.find_outcomes(underlying="btc", text="targetPrice:90000")
    assert found == [SAMPLE_OUTCOMES[0]]
    assert calls == ["metadata"]


def test_find_outcomes_normalizes_metadata_and_filter_whitespace(monkeypatch):
    outcome = {
        "name": "Daily BTC outcome",
        "description": "class:priceBinary | underlying: BTC | targetPrice:90000",
    }
    client = hl.HyperliquidTestnetClient()
    monkeypatch.setattr(client, "get_outcome_metadata", lambda: [outcome])

    assert client.find_outcomes(underlying=" btc ", text="  BTC  ") == [outcome]


def test_http_protocol_exception_is_wrapped_and_connection_closed(monkeypatch):
    state = {"closed": False}

    class BrokenConnection:
        def __init__(self, host, port, timeout):
            assert host == "api.hyperliquid-testnet.xyz"
            assert port == 443

        def request(self, *args, **kwargs):
            raise HTTPException("connection ended unexpectedly")

        def close(self):
            state["closed"] = True

    monkeypatch.setattr(hl, "HTTPSConnection", BrokenConnection)
    with pytest.raises(hl.HyperliquidTestnetError, match="Could not reach"):
        hl._post_info_json(
            "https://api.hyperliquid-testnet.xyz/info", {"type": "outcomeMeta"}, 1
        )
    assert state["closed"]


def test_oversized_response_is_rejected_and_connection_closed(monkeypatch):
    state = {"closed": False}

    class OversizedResponse:
        status = 200

        def read(self, size):
            assert size == hl._MAX_RESPONSE_BYTES + 1
            return b"x" * size

    class OversizedConnection:
        def __init__(self, host, port, timeout):
            assert host == "api.hyperliquid-testnet.xyz"

        def request(self, *args, **kwargs):
            return None

        def getresponse(self):
            return OversizedResponse()

        def close(self):
            state["closed"] = True

    monkeypatch.setattr(hl, "HTTPSConnection", OversizedConnection)
    with pytest.raises(hl.HyperliquidTestnetError, match="size limit"):
        hl._post_info_json(
            "https://api.hyperliquid-testnet.xyz/info", {"type": "outcomeMeta"}, 1
        )
    assert state["closed"]


def test_malformed_info_response_is_rejected(monkeypatch):
    monkeypatch.setattr(hl, "_post_info_json", lambda *args: {"markets": []})
    client = hl.HyperliquidTestnetClient()

    with pytest.raises(hl.HyperliquidTestnetError, match="outcomes list"):
        client.get_outcome_metadata()


def test_non_dict_outcome_record_is_rejected(monkeypatch):
    monkeypatch.setattr(hl, "_post_info_json", lambda *args: {"outcomes": ["bad"]})
    client = hl.HyperliquidTestnetClient()

    with pytest.raises(hl.HyperliquidTestnetError, match="malformed"):
        client.get_outcome_metadata()


def test_invalid_timeout_is_rejected():
    with pytest.raises(ValueError, match="timeout"):
        hl.HyperliquidTestnetClient(timeout=float("inf"))


def test_local_query_rejects_empty_filters():
    client = hl.HyperliquidTestnetClient()
    with pytest.raises(ValueError, match="underlying"):
        client.find_outcomes(underlying=" ")
    with pytest.raises(ValueError, match="text"):
        client.find_outcomes(text="")


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "clearinghouseState", "user": "0xabc"},
        {"type": "openOrders", "user": "0xabc"},
        {"type": "spotClearinghouseState", "user": "0xabc"},
        {"type": "outcomeMeta", "user": "0xabc"},
        {"type": "spotMetaAndAssetCtxs", "coin": "#1230"},
        {"type": "l2Book"},
        {"type": "l2Book", "coin": "#1230", "nSigFigs": 3},
        {"type": "l2Book", "coin": ""},
        {"type": "l2Book", "coin": "#1230\n"},
        {"type": "l2Book", "coin": "x" * 129},
    ],
)
def test_info_payloads_are_allowlisted_and_rejected_before_transport(
    monkeypatch, payload
):
    calls = []
    monkeypatch.setattr(
        hl, "HTTPSConnection", lambda *args, **kwargs: calls.append(args)
    )

    with pytest.raises(hl.TestnetOnlyError):
        hl._post_info_json("https://api.hyperliquid-testnet.xyz/info", payload, 1)
    assert calls == []


def test_all_approved_info_payload_schemas_reach_only_testnet(monkeypatch):
    import json

    calls = []

    class Response:
        status = 200

        def read(self, size):
            return b"{}"

    class Connection:
        def __init__(self, host, port, timeout):
            assert (host, port) == ("api.hyperliquid-testnet.xyz", 443)

        def request(self, method, path, *, body, headers):
            calls.append((method, path, json.loads(body)))

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(hl, "HTTPSConnection", Connection)
    endpoint = "https://api.hyperliquid-testnet.xyz/info"
    for payload in (
        {"type": "outcomeMeta"},
        {"type": "spotMetaAndAssetCtxs"},
        {"type": "l2Book", "coin": "#1230"},
    ):
        hl._post_info_json(endpoint, payload, 1)

    assert calls == [
        ("POST", "/info", {"type": "outcomeMeta"}),
        ("POST", "/info", {"type": "spotMetaAndAssetCtxs"}),
        ("POST", "/info", {"type": "l2Book", "coin": "#1230"}),
    ]


def test_get_outcome_prices_joins_contexts_by_documented_outcome_coin(monkeypatch):
    client = hl.HyperliquidTestnetClient()
    monkeypatch.setattr(client, "get_outcome_metadata", lambda: SAMPLE_OUTCOMES)
    calls = []

    def fake_post(url, payload, timeout):
        calls.append((url, payload, timeout))
        return [
            {"tokens": [], "universe": []},
            [
                {
                    "coin": "#1230",
                    "markPx": "0.62",
                    "midPx": "0.63",
                    "prevDayPx": "0.58",
                    "dayNtlVlm": "12.5",
                    "dayBaseVlm": "20.0",
                },
                {"coin": "#1231", "markPx": "0.38", "midPx": None},
                {"coin": "#1240", "markPx": "0.51", "midPx": "0.52"},
                {"coin": "#1241", "markPx": "0.49", "midPx": "0.48"},
                {"coin": "BTC", "markPx": "90000"},
            ],
        ]

    monkeypatch.setattr(hl, "_post_info_json", fake_post)
    prices = client.get_outcome_prices()

    assert calls == [
        (
            "https://api.hyperliquid-testnet.xyz/info",
            {"type": "spotMetaAndAssetCtxs"},
            10.0,
        )
    ]
    assert [(price["outcome"], price["side"], price["coin"]) for price in prices] == [
        (123, 0, "#1230"),
        (123, 1, "#1231"),
        (124, 0, "#1240"),
        (124, 1, "#1241"),
    ]
    assert prices[0]["side_name"] == "Yes"
    assert prices[0]["markPx"] == "0.62"
    assert prices[0]["midPx"] == "0.63"
    assert prices[0]["prevDayPx"] == "0.58"
    assert prices[0]["dayNtlVlm"] == "12.5"
    assert prices[0]["dayBaseVlm"] == "20.0"
    assert prices[1]["context_available"] is True
    assert prices[1]["midPx"] is None


def test_get_outcome_prices_marks_missing_context_without_guessing(monkeypatch):
    client = hl.HyperliquidTestnetClient()
    monkeypatch.setattr(client, "get_outcome_metadata", lambda: SAMPLE_OUTCOMES[:1])
    monkeypatch.setattr(
        hl,
        "_post_info_json",
        lambda *args: [{"tokens": [], "universe": []}, [{"coin": "BTC"}]],
    )

    prices = client.get_outcome_prices()

    assert len(prices) == 2
    assert all(price["context_available"] is False for price in prices)
    assert all(price["markPx"] is None for price in prices)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("markPx", "NaN"),
        ("midPx", "Infinity"),
        ("prevDayPx", "1.01"),
        ("midPx", "-0.01"),
        ("dayNtlVlm", "-0.1"),
        ("dayBaseVlm", "NaN"),
        ("markPx", True),
        ("midPx", {"unexpected": "object"}),
    ],
)
def test_get_outcome_prices_rejects_invalid_context_values(monkeypatch, field, value):
    client = hl.HyperliquidTestnetClient()
    monkeypatch.setattr(client, "get_outcome_metadata", lambda: SAMPLE_OUTCOMES[:1])
    monkeypatch.setattr(
        hl,
        "_post_info_json",
        lambda *args: [
            {"tokens": [], "universe": []},
            [{"coin": "#1230", field: value}],
        ],
    )

    with pytest.raises(hl.HyperliquidTestnetError, match=field):
        client.get_outcome_prices()


def test_get_outcome_prices_allows_binary_price_bounds_zero_volumes_and_missing_values(
    monkeypatch,
):
    client = hl.HyperliquidTestnetClient()
    monkeypatch.setattr(client, "get_outcome_metadata", lambda: SAMPLE_OUTCOMES[:1])
    monkeypatch.setattr(
        hl,
        "_post_info_json",
        lambda *args: [
            {"tokens": [], "universe": []},
            [
                {
                    "coin": "#1230",
                    "markPx": "0",
                    "midPx": "1",
                    "prevDayPx": None,
                    "dayNtlVlm": "0",
                    "dayBaseVlm": "0.0",
                }
            ],
        ],
    )

    prices = client.get_outcome_prices()

    assert prices[0]["markPx"] == "0"
    assert prices[0]["midPx"] == "1"
    assert prices[0]["prevDayPx"] is None
    assert prices[0]["dayNtlVlm"] == "0"
    assert prices[0]["dayBaseVlm"] == "0.0"
    assert prices[1]["context_available"] is False


@pytest.mark.parametrize(
    "spot_response",
    [
        {"tokens": [], "universe": []},
        [{"tokens": [], "universe": []}],
        [{"tokens": [], "universe": []}, {"coin": "#1230"}],
        [{"tokens": [], "universe": []}, ["malformed"]],
        [{"universe": []}, []],
        [{"tokens": []}, []],
    ],
)
def test_get_outcome_prices_rejects_malformed_spot_response(monkeypatch, spot_response):
    client = hl.HyperliquidTestnetClient()
    monkeypatch.setattr(client, "get_outcome_metadata", lambda: SAMPLE_OUTCOMES[:1])
    monkeypatch.setattr(hl, "_post_info_json", lambda *args: spot_response)

    with pytest.raises(
        hl.HyperliquidTestnetError, match="spot metadata or asset contexts"
    ):
        client.get_outcome_prices()


def test_get_outcome_prices_rejects_invalid_outcome_encoding_metadata(monkeypatch):
    client = hl.HyperliquidTestnetClient()
    monkeypatch.setattr(
        client,
        "get_outcome_metadata",
        lambda: [{"outcome": True, "sideSpecs": [{"name": "Yes"}, {"name": "No"}]}],
    )
    monkeypatch.setattr(
        hl,
        "_post_info_json",
        lambda *args: [{"tokens": [], "universe": []}, []],
    )

    with pytest.raises(hl.HyperliquidTestnetError, match="invalid or duplicate"):
        client.get_outcome_prices()


def test_get_outcome_book_uses_encoded_coin_and_validates_response(monkeypatch):
    client = hl.HyperliquidTestnetClient()
    monkeypatch.setattr(client, "get_outcome_metadata", lambda: SAMPLE_OUTCOMES)
    calls = []

    def fake_post(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {
            "coin": "#1231",
            "time": 1795352737058,
            "levels": [[], [{"px": "0.39", "sz": "12.0", "n": 2}]],
        }

    monkeypatch.setattr(hl, "_post_info_json", fake_post)
    book = client.get_outcome_book(123, 1)

    assert calls == [
        (
            "https://api.hyperliquid-testnet.xyz/info",
            {"type": "l2Book", "coin": "#1231"},
            10.0,
        )
    ]
    assert book["coin"] == "#1231"
    assert book["levels"][1][0]["px"] == "0.39"


def test_get_outcome_book_rejects_unknown_outcome_before_book_request(monkeypatch):
    client = hl.HyperliquidTestnetClient()
    monkeypatch.setattr(client, "get_outcome_metadata", lambda: SAMPLE_OUTCOMES)
    calls = []
    monkeypatch.setattr(hl, "_post_info_json", lambda *args: calls.append(args))

    with pytest.raises(hl.HyperliquidTestnetError, match="listed Testnet outcome"):
        client.get_outcome_book(999, 0)
    assert calls == []


def test_get_outcome_book_rejects_invalid_side_before_request(monkeypatch):
    client = hl.HyperliquidTestnetClient()
    calls = []
    monkeypatch.setattr(client, "get_outcome_metadata", lambda: calls.append("meta"))

    with pytest.raises(ValueError, match="side must be 0 or 1"):
        client.get_outcome_book(123, 2)
    assert calls == []


@pytest.mark.parametrize(
    "book",
    [
        {"coin": "#1231", "time": 1, "levels": [[], []]},
        {"coin": "#1230", "time": 1, "levels": [[]]},
        {"coin": "#1230", "time": True, "levels": [[], []]},
        {"coin": "#1230", "time": 1, "levels": [[{"px": 0.5, "sz": "1", "n": 1}], []]},
        {
            "coin": "#1230",
            "time": 1,
            "levels": [[{"px": "0.5", "sz": "1", "n": True}], []],
        },
    ],
)
def test_get_outcome_book_rejects_malformed_book_responses(monkeypatch, book):
    client = hl.HyperliquidTestnetClient()
    monkeypatch.setattr(client, "get_outcome_metadata", lambda: SAMPLE_OUTCOMES)
    monkeypatch.setattr(hl, "_post_info_json", lambda *args: book)

    with pytest.raises(hl.HyperliquidTestnetError, match="l2Book"):
        client.get_outcome_book(123, 0)
