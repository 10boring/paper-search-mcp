from unittest.mock import Mock

import pytest

from paper_search_mcp.academic_platforms.medrxiv import MedRxivSearcher


def _searcher(reported_category):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "messages": [{"status": "ok", "category": reported_category}],
        "collection": [{
            "doi": "10.1101/2026.01.01.000001",
            "title": "Test paper",
            "authors": "A One; B Two",
            "abstract": "Abstract",
            "date": "2026-01-01",
            "version": "1",
            "category": "epidemiology",
        }],
    }
    searcher = MedRxivSearcher()
    searcher.session.get = Mock(return_value=response)
    return searcher


def test_a_query_that_is_no_category_is_refused():
    # The API has no keyword search: it answers an unknown category with every
    # recent preprint and reports the category as "all".
    with pytest.raises(ValueError, match="category"):
        _searcher("all").search("machine learning", max_results=5)


def test_a_known_category_is_searched():
    assert len(_searcher("epidemiology").search("epidemiology", max_results=1)) == 1
