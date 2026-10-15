from unittest.mock import Mock

from paper_search_mcp.academic_platforms.hal import HALSearcher
from paper_search_mcp.academic_platforms.zenodo import ZenodoSearcher


def _sent_params(searcher, **kwargs):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {}
    searcher.session.get = Mock(return_value=response)
    searcher.search("llm inference", max_results=5, **kwargs)
    return searcher.session.get.call_args.kwargs["params"]


def test_zenodo_filters_a_range_of_years():
    q = _sent_params(ZenodoSearcher(), year="2025-2026")["q"]
    assert "publication_date:[2025-01-01 TO 2026-12-31]" in q


def test_zenodo_filters_one_year():
    q = _sent_params(ZenodoSearcher(), year="2026")["q"]
    assert "publication_date:[2026-01-01 TO 2026-12-31]" in q


def test_hal_filters_a_range_of_years():
    fq = _sent_params(HALSearcher(), year="2025-2026")["fq"]
    assert "publicationDateY_i:[2025 TO 2026]" in fq


def test_hal_filters_one_year():
    fq = _sent_params(HALSearcher(), year="2026")["fq"]
    assert "publicationDateY_i:2026" in fq
