"""Deterministic Semantic Scholar retry/fallback coverage adapted from PR #71."""

from unittest.mock import Mock

import pytest
import requests

from paper_search_mcp.academic_platforms import semantic


def response(status, headers=None):
    return Mock(status_code=status, headers=headers or {})


class FakeClock:
    """time.monotonic and time.sleep for the retry loop: sleeping advances the clock."""

    def __init__(self):
        self.now = 0.0
        self.sleep = Mock(side_effect=self._advance)

    def _advance(self, seconds):
        self.now += seconds

    def monotonic(self):
        return self.now

    def waits(self):
        return [call.args[0] for call in self.sleep.call_args_list]


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(semantic.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(semantic.time, "sleep", fake.sleep)
    return fake


@pytest.fixture
def searcher(monkeypatch, clock):
    instance = semantic.SemanticSearcher()
    monkeypatch.setattr(instance, "get_api_key", lambda: None)
    return instance


@pytest.mark.parametrize("exception_path", [False, True])
def test_keyless_rate_limit_backs_off_then_retries_every_8s_until_answered(
    searcher, monkeypatch, clock, exception_path
):
    # Keyless requests share one pool with every other keyless user, so getting
    # through is luck: keep asking until it answers.
    limited = response(429)
    if exception_path:
        refused = response(200)
        refused.raise_for_status.side_effect = requests.HTTPError(response=limited)
    else:
        refused = limited
    success = response(200)
    get = Mock(side_effect=[refused] * 7 + [success])
    monkeypatch.setattr(searcher.session, "get", get)
    assert searcher.request_api("paper/search", {"query": "test"}) is success
    assert get.call_count == 8
    assert clock.waits() == [1, 2, 4, 8, 8, 8, 8]


def test_keyless_retries_give_up_after_five_minutes(searcher, monkeypatch, clock):
    get = Mock(return_value=response(429))
    monkeypatch.setattr(searcher.session, "get", get)
    result = searcher.request_api("paper/search", {})
    assert result["error"] == "rate_limited"
    assert result["status_code"] == 429
    waits = clock.waits()
    assert waits[:4] == [1, 2, 4, 8]
    assert set(waits[4:]) == {8}
    assert 300 - 8 < sum(waits) <= 300
    assert get.call_count == len(waits) + 1


def test_authenticated_rate_limit_retries_with_retry_after(searcher, monkeypatch, clock):
    monkeypatch.setattr(searcher, "get_api_key", lambda: "test-key")
    success = response(200)
    get = Mock(side_effect=[response(429, {"Retry-After": "4"}), success])
    monkeypatch.setattr(searcher.session, "get", get)
    assert searcher.request_api("paper/search", {}) is success
    assert get.call_count == 2
    assert all(call.kwargs["headers"] == {"x-api-key": "test-key"} for call in get.call_args_list)
    assert clock.waits() == [4]


def test_authenticated_retries_are_bounded(searcher, monkeypatch, clock):
    monkeypatch.setattr(searcher, "get_api_key", lambda: "test-key")
    get = Mock(return_value=response(429))
    monkeypatch.setattr(searcher.session, "get", get)
    assert searcher.request_api("paper/search", {})["error"] == "rate_limited"
    assert get.call_count == 3
    assert clock.waits() == [2, 4]


@pytest.mark.parametrize("fallback_status", [200, 429])
def test_rejected_key_falls_back_to_anonymous_access(searcher, monkeypatch, fallback_status):
    monkeypatch.setattr(searcher, "get_api_key", lambda: "rejected-key")
    fallback = response(fallback_status)
    get = Mock(side_effect=lambda *a, **k: response(403) if get.call_count == 1 else fallback)
    monkeypatch.setattr(searcher.session, "get", get)
    result = searcher.request_api("paper/search", {})
    if fallback_status == 200:
        assert result is fallback
        assert get.call_count == 2
    else:
        assert result["error"] == "rate_limited"
    assert get.call_args_list[0].kwargs["headers"] == {"x-api-key": "rejected-key"}
    assert all(call.kwargs["headers"] == {} for call in get.call_args_list[1:])


def test_search_surfaces_rate_limit_as_error(searcher, monkeypatch):
    monkeypatch.setattr(searcher.session, "get", Mock(return_value=response(429)))
    with pytest.raises(semantic.SemanticScholarRequestError, match="rate_limited, HTTP 429"):
        searcher.search("test")
