"""Unit tests for ranking metrics (no models, no ISOT)."""

from __future__ import annotations

from evidence_retrieval.eval.metrics import (
    aggregate_metrics,
    dcg_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    recall_at_k_binary,
)


def test_recall_at_k_and_hit_rate():
    relevant = {"a", "b"}
    ranked = ["x", "a", "y", "b"]
    assert recall_at_k(relevant, ranked, k=2) == 0.5
    assert recall_at_k(relevant, ranked, k=4) == 1.0
    assert recall_at_k_binary(relevant, ranked, k=1) == 0.0
    assert recall_at_k_binary(relevant, ranked, k=2) == 1.0
    assert recall_at_k(set(), ranked, k=3) == 0.0


def test_mrr():
    assert mrr({"b"}, ["a", "b", "c"]) == 0.5
    assert mrr({"z"}, ["a", "b", "c"]) == 0.0
    assert mrr(set(), ["a"]) == 0.0


def test_ndcg_perfect_and_partial():
    relevant = {"a", "b"}
    perfect = ["a", "b", "c"]
    partial = ["c", "a", "b"]
    assert ndcg_at_k(relevant, perfect, k=2) == 1.0
    assert 0.0 < ndcg_at_k(relevant, partial, k=3) < 1.0
    assert ndcg_at_k(set(), perfect, k=2) == 0.0


def test_dcg_at_k_known_value():
    # rel=[1,0] → (2^1-1)/log2(2) + 0 = 1.0
    assert dcg_at_k([1.0, 0.0], k=2) == 1.0


def test_aggregate_metrics_mean():
    rows = [
        {"hit@5": 1.0, "mrr": 1.0},
        {"hit@5": 0.0, "mrr": 0.5},
    ]
    agg = aggregate_metrics(rows)
    assert agg["hit@5"] == 0.5
    assert agg["mrr"] == 0.75
    assert aggregate_metrics([]) == {}
