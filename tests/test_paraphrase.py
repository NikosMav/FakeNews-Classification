"""Unit tests for deterministic title paraphrases (no MiniLM / no ISOT)."""

from __future__ import annotations

import pandas as pd

from evidence_retrieval.eval.queries import (
    build_paraphrase_queries,
    paraphrase_title,
)


def test_paraphrase_differs_from_title():
    title = "Federal Reserve raises interest rates amid inflation concerns"
    para = paraphrase_title(title)
    assert para
    assert para.casefold() != title.casefold()


def test_paraphrase_deterministic():
    title = "WATCH: Trump Supporter Slams Media Over Climate Deal (VIDEO)"
    assert paraphrase_title(title) == paraphrase_title(title)


def test_paraphrase_strips_clickbait_and_outlet_suffix():
    title = "OOPS! Puerto Rico needs restructuring to avoid cascading defaults: Treasury"
    para = paraphrase_title(title)
    assert "OOPS" not in para.upper()
    assert not para.lower().endswith("treasury")
    assert para.casefold() != title.casefold()


def test_paraphrase_empty_and_short():
    assert "happened" in paraphrase_title("").lower()
    short = paraphrase_title("Hi")
    assert short.casefold() != "hi"


def test_build_paraphrase_queries_preserves_gold_article_id(tiny_articles: pd.DataFrame):
    from evidence_retrieval.chunking import articles_to_chunks

    chunks = articles_to_chunks(tiny_articles, chunk_words=40, overlap=5, fields=("body",))
    queries = build_paraphrase_queries(chunks, max_queries=10, random_state=7)
    assert queries
    for q in queries:
        assert q.query_id.startswith("para-")
        assert q.query_text.casefold() != q.title.casefold()
        assert q.gold_article_ids == [q.article_id]
        assert q.gold_chunk_ids
        assert all(cid.startswith(f"{q.article_id}:") for cid in q.gold_chunk_ids)
