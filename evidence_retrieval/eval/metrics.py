"""Ranking metrics: Recall@k, MRR, nDCG@k."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(relevant: set[str], ranked: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(ranked[:k])
    return len(relevant & top) / len(relevant)


def recall_at_k_binary(relevant: set[str], ranked: Sequence[str], k: int) -> float:
    """1.0 if any relevant item appears in top-k (hit rate / Success@k)."""
    if not relevant:
        return 0.0
    return 1.0 if relevant & set(ranked[:k]) else 0.0


def mrr(relevant: set[str], ranked: Sequence[str]) -> float:
    if not relevant:
        return 0.0
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        total += (2**rel - 1) / math.log2(i + 1)
    return total


def ndcg_at_k(relevant: set[str], ranked: Sequence[str], k: int) -> float:
    """Binary relevance nDCG@k."""
    if not relevant:
        return 0.0
    gains = [1.0 if item in relevant else 0.0 for item in ranked[:k]]
    dcg = dcg_at_k(gains, k)
    ideal = dcg_at_k([1.0] * min(len(relevant), k), k)
    if ideal == 0.0:
        return 0.0
    return dcg / ideal


def aggregate_metrics(
    per_query: Iterable[dict[str, float]],
) -> dict[str, float]:
    rows = list(per_query)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(sum(r[k] for r in rows) / len(rows)) for k in keys}
