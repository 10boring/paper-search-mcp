"""Deterministic Semantic Scholar retry/fallback coverage adapted from PR #71."""

from unittest.mock import Mock

import pytest
import requests

from paper_search_mcp.academic_platforms import semantic


def response(status, headers=None):
    return Mock(status_code=status, headers=headers or {})


@pytest.fixture
def searcher(monkeypatch):
    instance = semantic.SemanticSearcher()
    monkeypatch.setattr(instance, "get_api_key", lambda: None)
    return instance


@pytest.mark.parametrize("exception_path", [False, True])
def test_anonymous_rate_limit_waits_and_retries(searcher, monkeypatch, exception_path):
    # Keyless requests share one pool with every other keyless user, so a 429
    # is usually gone a little later; giving up at once loses the source.
    limited = response(429, {"Retry-After": "7"})
    if exception_path:
        first = response(200)
        first.raise_for_status.side_effect = requests.HTTPError(response=limited)
    else:
        first = limited
    success = response(200)
    get = Mock(side_effect=[first, success])
    sleep = Mock()
    monkeypatch.setattr(searcher.session, "get", get)
    monkeypatch.setattr(semantic.time, "sleep", sleep)
    assert searcher.request_api("paper/search", {"query": "test"}) is success
    assert get.call_count == 2
    sleep.assert_called_once_with(7)


def test_anonymous_retries_are_bounded_and_back_off(searcher, monkeypatch):
    get = Mock(return_value=response(429))
    sleep = Mock()
    monkeypatch.setattr(searcher.session, "get", get)
    monkeypatch.setattr(semantic.time, "sleep", sleep)
    result = searcher.request_api("paper/search", {})
    assert result["error"] == "rate_limited"
    assert result["status_code"] == 429
    assert get.call_count == 6
    assert [call.args[0] for call in sleep.call_args_list] == [5, 10, 20, 30, 30]


def test_a_long_retry_after_is_capped(searcher, monkeypatch):
    success = response(200)
    get = Mock(side_effect=[response(429, {"Retry-After": "600"}), success])
    sleep = Mock()
    monkeypatch.setattr(searcher.session, "get", get)
    monkeypatch.setattr(semantic.time, "sleep", sleep)
    assert searcher.request_api("paper/search", {}) is success
    sleep.assert_called_once_with(30)


def test_authenticated_rate_limit_retries_with_retry_after(searcher, monkeypatch):
    monkeypatch.setattr(searcher, "get_api_key", lambda: "test-key")
    success = response(200)
    get = Mock(side_effect=[response(429, {"Retry-After": "4"}), success])
    sleep = Mock()
    monkeypatch.setattr(searcher.session, "get", get)
    monkeypatch.setattr(semantic.time, "sleep", sleep)
    assert searcher.request_api("paper/search", {}) is success
    assert get.call_count == 2
    assert all(call.kwargs["headers"] == {"x-api-key": "test-key"} for call in get.call_args_list)
    sleep.assert_called_once_with(4)


def test_authenticated_retries_are_bounded(searcher, monkeypatch):
    monkeypatch.setattr(searcher, "get_api_key", lambda: "test-key")
    get = Mock(return_value=response(429))
    sleep = Mock()
    monkeypatch.setattr(searcher.session, "get", get)
    monkeypatch.setattr(semantic.time, "sleep", sleep)
    assert searcher.request_api("paper/search", {})["error"] == "rate_limited"
    assert get.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [2, 4]


@pytest.mark.parametrize("fallback_status", [200, 429])
def test_rejected_key_falls_back_to_anonymous_access(searcher, monkeypatch, fallback_status):
    monkeypatch.setattr(searcher, "get_api_key", lambda: "rejected-key")
    get = Mock(side_effect=[response(403)] + [response(fallback_status)] * 5)
    sleep = Mock()
    monkeypatch.setattr(searcher.session, "get", get)
    monkeypatch.setattr(semantic.time, "sleep", sleep)
    result = searcher.request_api("paper/search", {})
    if fallback_status == 200:
        assert result.status_code == 200
        assert get.call_count == 2
    else:
        assert result["error"] == "rate_limited"
    assert get.call_args_list[0].kwargs["headers"] == {"x-api-key": "rejected-key"}
    assert all(call.kwargs["headers"] == {} for call in get.call_args_list[1:])


def test_search_surfaces_rate_limit_as_error(searcher, monkeypatch):
    monkeypatch.setattr(searcher.session, "get", Mock(return_value=response(429)))
    monkeypatch.setattr(semantic.time, "sleep", Mock())
    with pytest.raises(semantic.SemanticScholarRequestError, match="rate_limited, HTTP 429"):
        searcher.search("test")
