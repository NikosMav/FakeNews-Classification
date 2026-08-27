"""Unit tests for Reciprocal Rank Fusion (pure logic, no models)."""

from __future__ import annotations

from evidence_retrieval.index import reciprocal_rank_fusion


def test_rrf_prefers_items_high_in_both_lists():
    sparse = [(10, 0.9), (20, 0.8), (30, 0.1)]
    dense = [(20, 0.95), (10, 0.5), (40, 0.2)]
    fused = reciprocal_rank_fusion([sparse, dense], k_rrf=60, fetch=4)
    ids = [i for i, _ in fused]
    # 10 and 20 appear in both → should outrank 30 and 40.
    assert ids[0] in {10, 20}
    assert ids[1] in {10, 20}
    assert set(ids) == {10, 20, 30, 40}


def test_rrf_score_formula():
    # Single list: score(idx) = 1/(k+rank)
    ranked = [(7, 1.0), (8, 0.5)]
    fused = reciprocal_rank_fusion([ranked], k_rrf=60)
    assert fused[0] == (7, 1.0 / 61)
    assert fused[1] == (8, 1.0 / 62)


def test_rrf_fetch_truncates():
    sparse = [(i, 1.0) for i in range(10)]
    dense = [(i, 1.0) for i in range(9, -1, -1)]
    fused = reciprocal_rank_fusion([sparse, dense], k_rrf=60, fetch=3)
    assert len(fused) == 3


def test_rrf_ignores_input_scores():
    # Different absolute scores, same ranks → identical RRF scores.
    a = reciprocal_rank_fusion([[(1, 99.0), (2, 1.0)]], k_rrf=10)
    b = reciprocal_rank_fusion([[(1, 0.01), (2, 0.009)]], k_rrf=10)
    assert a == b
