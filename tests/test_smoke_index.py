"""Smoke: build → query plumbing without ISOT or MiniLM downloads."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evidence_retrieval.encoders import SparseEncoder
from evidence_retrieval.index import IndexConfig, PassageIndex
from tests.conftest import FakeDenseEncoder


def test_build_query_smoke_in_memory(tiny_articles: pd.DataFrame, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("evidence_retrieval.index.DenseEncoder", FakeDenseEncoder)

    # Tiny corpus: lower min_df so TF-IDF has a vocabulary.
    real_sparse_init = SparseEncoder.__init__

    def _sparse_init(self, *args, **kwargs):
        kwargs.setdefault("min_df", 1)
        kwargs.setdefault("max_features", 5000)
        real_sparse_init(self, *args, **kwargs)

    monkeypatch.setattr(SparseEncoder, "__init__", _sparse_init)

    config = IndexConfig(
        n_articles=len(tiny_articles),
        chunk_words=40,
        overlap=5,
        fields=("body",),
        dense_model="fake-minilm",
        random_state=7,
        embed_batch_size=8,
    )
    index = PassageIndex.build(
        articles=tiny_articles,
        config=config,
        show_progress=False,
    )
    assert len(index.chunks) >= 3
    assert index.dense_embeddings is not None
    assert index.dense_embeddings.shape[0] == len(index.chunks)

    hits = index.query(
        "Federal Reserve raises interest rates",
        top_k=3,
        method="hybrid",
    )
    assert len(hits) == 3
    assert hits[0].rank == 1
    assert hits[0].title
    assert hits[0].label_name in {"true", "fake"}
    assert hits[0].passage
    assert hits[0].score > 0

    # Self-ish query should prefer Fed/rates articles over the school lunch piece.
    top_ids = {h.article_id for h in hits}
    assert 101 in top_ids or 202 in top_ids or 404 in top_ids
    assert all(h.method == "hybrid" for h in hits)


def test_build_save_load_query_roundtrip(
    tiny_articles: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr("evidence_retrieval.index.DenseEncoder", FakeDenseEncoder)

    real_sparse_init = SparseEncoder.__init__

    def _sparse_init(self, *args, **kwargs):
        kwargs.setdefault("min_df", 1)
        kwargs.setdefault("max_features", 5000)
        real_sparse_init(self, *args, **kwargs)

    monkeypatch.setattr(SparseEncoder, "__init__", _sparse_init)

    config = IndexConfig(
        n_articles=len(tiny_articles),
        chunk_words=40,
        overlap=5,
        fields=("body",),
        dense_model="fake-minilm",
        random_state=7,
    )
    index = PassageIndex.build(articles=tiny_articles, config=config, show_progress=False)
    out = index.save(tmp_path / "fixture_index")
    assert (out / "chunks.parquet").exists()
    assert (out / "config.json").exists()

    loaded = PassageIndex.load(out)
    # Avoid MiniLM on load: swap dense encoder for the fake before querying.
    loaded.dense_encoder = FakeDenseEncoder()
    hits = loaded.query("inflation and interest rates", top_k=2, method="tfidf")
    assert len(hits) == 2
    assert hits[0].chunk_id
