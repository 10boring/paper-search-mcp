import asyncio
from unittest.mock import AsyncMock, patch

from paper_search_mcp import server

DATED = "arxiv,pubmed,pmc,core,europepmc,openalex,crossref,doaj,dblp,openaire,zenodo,hal"


def _sent(year):
    """What search_papers hands each source's searcher for this year."""
    calls = {}

    async def capture(searcher, query, max_results, **kwargs):
        calls[type(searcher).__name__] = (query, kwargs)
        return []

    with patch.object(server, "async_search", AsyncMock(side_effect=capture)):
        asyncio.run(server.search_papers("llm inference", max_results_per_source=5, sources=DATED, year=year))
    return calls


def test_the_period_reaches_every_source_that_can_filter_by_date():
    # Without it a source returns its best papers of any year, and the ones
    # outside the period are thrown away afterwards, leaving few or none.
    sent = _sent("2025-2026")
    assert "submittedDate:[202501010000 TO 202612312359]" in sent["ArxivSearcher"][0]
    assert "2025:2026[dp]" in sent["PubMedSearcher"][0]
    assert "2025:2026[pdat]" in sent["PMCSearcher"][0]
    assert "yearPublished>=2025 AND yearPublished<=2026" in sent["CORESearcher"][0]
    assert "PUB_YEAR:[2025 TO 2026]" in sent["EuropePMCSearcher"][0]
    assert sent["OpenAlexSearcher"][1] == {"filter": "publication_year:2025-2026"}
    assert sent["CrossRefSearcher"][1] == {"filter": "from-pub-date:2025-01-01,until-pub-date:2026-12-31"}
    for name in ("DOAJSearcher", "DBLPSearcher", "OpenAiresearcher", "ZenodoSearcher", "HALSearcher"):
        assert sent[name][1] == {"year": "2025-2026"}, name


def test_one_year_is_a_period_of_one_year():
    sent = _sent("2026")
    assert sent["OpenAlexSearcher"][1] == {"filter": "publication_year:2026-2026"}
    assert "2026:2026[dp]" in sent["PubMedSearcher"][0]


def test_without_a_period_the_query_goes_out_unchanged():
    sent = _sent("")
    assert all(query == "llm inference" for query, _ in sent.values())
    assert not any({"filter", "year"} & kwargs.keys() for _, kwargs in sent.values())
