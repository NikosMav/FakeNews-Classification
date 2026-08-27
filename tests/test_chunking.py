"""Unit tests for word-window chunking (no models, no ISOT)."""

from __future__ import annotations

import pandas as pd

from evidence_retrieval.chunking import articles_to_chunks, chunk_text


def test_chunk_text_empty_and_short():
    assert chunk_text("") == []
    assert chunk_text("one two three", chunk_words=120) == ["one two three"]


def test_chunk_text_overlapping_windows():
    words = " ".join(f"w{i}" for i in range(50))
    chunks = chunk_text(words, chunk_words=20, overlap=5, min_tail_words=5)
    assert len(chunks) >= 2
    assert chunks[0].split()[0] == "w0"
    # Overlap: second window should start before the first ends.
    first = chunks[0].split()
    second = chunks[1].split()
    assert second[0] in first


def test_chunk_text_merges_short_tail():
    words = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_text(words, chunk_words=20, overlap=0, min_tail_words=10)
    # 25 words with step 20 → last 5-word tail should merge into previous.
    assert len(chunks) == 1
    assert len(chunks[0].split()) == 25


def test_articles_to_chunks_body_metadata(tiny_articles: pd.DataFrame):
    chunks = articles_to_chunks(tiny_articles, chunk_words=40, overlap=5, fields=("body",))
    assert not chunks.empty
    assert set(chunks["field"]) == {"body"}
    assert {"article_id", "title", "label_name", "passage", "chunk_id"}.issubset(chunks.columns)
    # Every chunk_id is article_id:index
    for _, row in chunks.iterrows():
        aid, idx = str(row["chunk_id"]).split(":")
        assert int(aid) == int(row["article_id"])
        assert int(idx) == int(row["chunk_index"])


def test_articles_to_chunks_title_only(tiny_articles: pd.DataFrame):
    chunks = articles_to_chunks(tiny_articles, fields=("title",))
    assert len(chunks) == len(tiny_articles)
    assert set(chunks["field"]) == {"title"}
    assert list(chunks["passage"]) == list(tiny_articles["title"])
