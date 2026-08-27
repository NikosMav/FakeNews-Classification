"""Eval package exports."""

from evidence_retrieval.eval.metrics import (
    aggregate_metrics,
    mrr,
    ndcg_at_k,
    recall_at_k,
    recall_at_k_binary,
)
from evidence_retrieval.eval.queries import (
    EvalQuery,
    build_paraphrase_queries,
    build_title_queries,
    paraphrase_queries_to_frame,
    paraphrase_title,
    queries_to_frame,
)

__all__ = [
    "EvalQuery",
    "aggregate_metrics",
    "build_paraphrase_queries",
    "build_title_queries",
    "mrr",
    "ndcg_at_k",
    "paraphrase_queries_to_frame",
    "paraphrase_title",
    "queries_to_frame",
    "recall_at_k",
    "recall_at_k_binary",
]
