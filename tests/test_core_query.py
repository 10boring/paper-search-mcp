from paper_search_mcp.academic_platforms.core import CORESearcher


def test_a_plain_query_requires_every_word():
    # CORE reads bare words as OR, which matches tens of millions of works.
    # Stopwords are left out, since one in an AND query can match nothing.
    assert (
        CORESearcher._build_query("large language model inference on consumer hardware")
        == "large AND language AND model AND inference AND consumer AND hardware"
    )
    assert CORESearcher._build_query("  transformers  ") == "transformers"


def test_a_structured_query_is_left_as_written():
    assert CORESearcher._build_query('"machine learning" OR robots') == '"machine learning" OR robots'
    assert CORESearcher._build_query("title:transformers") == "title:transformers"
    assert CORESearcher._build_query("deep AND learning") == "deep AND learning"


def test_a_query_of_only_stopwords_is_kept():
    assert CORESearcher._build_query("on the") == "on the"
