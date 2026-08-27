"""Run retrieval evaluation and ablations; write results CSVs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from evidence_retrieval.data import load_isot, stratified_sample
from evidence_retrieval.eval.metrics import (
    aggregate_metrics,
    mrr,
    ndcg_at_k,
    recall_at_k,
    recall_at_k_binary,
)
from evidence_retrieval.eval.queries import build_title_queries
from evidence_retrieval.index import IndexConfig, Method, PassageIndex

KS = (1, 5, 10)


def _evaluate_method(
    index: PassageIndex,
    queries,
    method: Method,
    ks: Sequence[int] = KS,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Return mean metrics + per-query detail frame."""
    per_query = []
    detail_rows = []
    max_k = max(ks)

    for q in queries:
        hits = index.query(
            q.query_text,
            top_k=max_k,
            method=method,
            # Keep self in the pool — gold passages are from this article.
            exclude_article_ids=None,
        )
        ranked_chunks = [h.chunk_id for h in hits]
        ranked_articles = [str(h.article_id) for h in hits]
        gold_chunks = set(q.gold_chunk_ids)
        gold_articles = {str(a) for a in q.gold_article_ids}

        row: dict[str, float] = {}
        for k in ks:
            row[f"passage_recall@{k}"] = recall_at_k(gold_chunks, ranked_chunks, k)
            row[f"article_hit@{k}"] = recall_at_k_binary(
                gold_articles, ranked_articles, k
            )
            row[f"ndcg@{k}"] = ndcg_at_k(gold_chunks, ranked_chunks, k)
        row["mrr"] = mrr(gold_chunks, ranked_chunks)
        row["article_mrr"] = mrr(gold_articles, ranked_articles)
        per_query.append(row)

        # Hard-negative / leakage signals among top-5
        top5 = hits[:5]
        opposite = sum(1 for h in top5 if h.label_name != q.label_name)
        same_subject_other = sum(
            1
            for h in top5
            if h.subject == q.subject and h.article_id != q.article_id
        )
        detail_rows.append(
            {
                "query_id": q.query_id,
                "method": method,
                "label_name": q.label_name,
                "subject": q.subject,
                "title": q.title,
                "top1_article_id": top5[0].article_id if top5 else None,
                "top1_label": top5[0].label_name if top5 else None,
                "top1_score": top5[0].score if top5 else None,
                "top1_title": top5[0].title if top5 else None,
                "gold_in_top5": row["article_hit@5"],
                "opposite_label_in_top5": opposite,
                "same_subject_other_in_top5": same_subject_other,
                "mrr": row["mrr"],
                "article_mrr": row["article_mrr"],
            }
        )

    means = aggregate_metrics(per_query)
    means["method"] = method  # type: ignore[assignment]
    means["n_queries"] = float(len(queries))
    return means, pd.DataFrame(detail_rows)


def run_main_comparison(
    data_dir: Path | str = "data",
    n_articles: int = 4000,
    chunk_words: int = 120,
    overlap: int = 20,
    max_queries: int = 300,
    random_state: int = 7,
    methods: Sequence[Method] = ("tfidf", "dense", "hybrid"),
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, PassageIndex]:
    """Build default body index; evaluate TF-IDF vs dense vs hybrid on SAME queries."""
    config = IndexConfig(
        n_articles=n_articles,
        chunk_words=chunk_words,
        overlap=overlap,
        fields=("body",),
        random_state=random_state,
    )
    index = PassageIndex.build(
        data_dir=data_dir, config=config, show_progress=show_progress
    )
    queries = build_title_queries(
        index.chunks, max_queries=max_queries, random_state=random_state
    )

    summary_rows = []
    details = []
    for method in methods:
        means, detail = _evaluate_method(index, queries, method)
        summary_rows.append(means)
        details.append(detail)

    summary = pd.DataFrame(summary_rows)
    detail = pd.concat(details, ignore_index=True)
    return summary, detail, index


def run_ablations(
    data_dir: Path | str = "data",
    n_articles: int = 2500,
    max_queries: int = 200,
    random_state: int = 7,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Chunk-size, field, and method ablations on a shared article sample."""
    df = load_isot(data_dir)
    articles = stratified_sample(df, n_articles=n_articles, random_state=random_state)

    experiments = [
        # chunk size (body, hybrid)
        {"name": "chunk60_hybrid", "chunk_words": 60, "fields": ("body",), "method": "hybrid"},
        {"name": "chunk120_hybrid", "chunk_words": 120, "fields": ("body",), "method": "hybrid"},
        {"name": "chunk240_hybrid", "chunk_words": 240, "fields": ("body",), "method": "hybrid"},
        # title vs body (dense)
        {"name": "title_only_dense", "chunk_words": 120, "fields": ("title",), "method": "dense"},
        {"name": "body_dense", "chunk_words": 120, "fields": ("body",), "method": "dense"},
        {"name": "title_body_dense", "chunk_words": 120, "fields": ("title_body",), "method": "dense"},
        # methods at fixed chunk120 body
        {"name": "body120_tfidf", "chunk_words": 120, "fields": ("body",), "method": "tfidf"},
        {"name": "body120_dense", "chunk_words": 120, "fields": ("body",), "method": "dense"},
        {"name": "body120_hybrid", "chunk_words": 120, "fields": ("body",), "method": "hybrid"},
    ]

    # Cache indexes by (chunk_words, fields)
    index_cache: dict[tuple, PassageIndex] = {}
    rows = []

    for exp in experiments:
        key = (exp["chunk_words"], exp["fields"])
        if key not in index_cache:
            cfg = IndexConfig(
                n_articles=n_articles,
                chunk_words=int(exp["chunk_words"]),
                overlap=20,
                fields=tuple(exp["fields"]),
                random_state=random_state,
            )
            index_cache[key] = PassageIndex.build(
                data_dir=data_dir,
                config=cfg,
                articles=articles,
                show_progress=show_progress,
            )
        index = index_cache[key]
        queries = build_title_queries(
            index.chunks, max_queries=max_queries, random_state=random_state
        )
        # Title-only index: gold = title chunk of same article
        if exp["fields"] == ("title",):
            # rebuild gold as the title chunk ids
            from evidence_retrieval.eval.queries import EvalQuery

            fixed = []
            for q in queries:
                gold = index.chunks.loc[
                    index.chunks["article_id"] == q.article_id, "chunk_id"
                ].astype(str).tolist()
                if not gold:
                    continue
                fixed.append(
                    EvalQuery(
                        query_id=q.query_id,
                        article_id=q.article_id,
                        query_text=q.query_text,
                        gold_chunk_ids=gold,
                        gold_article_ids=[q.article_id],
                        label_name=q.label_name,
                        subject=q.subject,
                        title=q.title,
                    )
                )
            queries = fixed

        means, _ = _evaluate_method(index, queries, exp["method"])  # type: ignore[arg-type]
        rows.append(
            {
                "experiment": exp["name"],
                "chunk_words": exp["chunk_words"],
                "fields": ",".join(exp["fields"]),
                "method": exp["method"],
                "n_queries": means["n_queries"],
                "article_hit@1": means["article_hit@1"],
                "article_hit@5": means["article_hit@5"],
                "article_hit@10": means["article_hit@10"],
                "passage_recall@5": means["passage_recall@5"],
                "ndcg@5": means["ndcg@5"],
                "ndcg@10": means["ndcg@10"],
                "mrr": means["mrr"],
                "article_mrr": means["article_mrr"],
            }
        )

    return pd.DataFrame(rows)


def write_qualitative_failures(
    detail: pd.DataFrame,
    out_path: Path | str,
    n_examples: int = 8,
) -> Path:
    """Document failure modes: misses, opposite-label leakage, same-subject collapse."""
    out_path = Path(out_path)
    # Prefer hybrid rows if present
    df = detail[detail["method"] == "hybrid"] if (detail["method"] == "hybrid").any() else detail

    misses = df[df["gold_in_top5"] < 1.0].head(n_examples)
    leakage = df[df["opposite_label_in_top5"] >= 2].head(n_examples)
    style = df[df["same_subject_other_in_top5"] >= 2].head(n_examples)

    lines = [
        "# Qualitative failure cases (auto-sampled from eval detail)",
        "",
        "These examples come from the title→passage evaluation detail dump.",
        "They illustrate limits of nearest-neighbor retrieval over ISOT — not claim verdicts.",
        "",
        "## Misses (gold article absent from top-5)",
        "",
    ]
    if misses.empty:
        lines.append("_No misses in the sampled hybrid detail (strong self-retrieval)._")
    else:
        for _, r in misses.iterrows():
            lines.append(
                f"- **Query title:** {r['title']!r}  \n"
                f"  Gold bucket: `{r['label_name']}` / subject `{r['subject']}`.  \n"
                f"  Top-1 returned: {r['top1_title']!r} "
                f"(bucket `{r['top1_label']}`, score={r['top1_score']:.3f})."
            )

    lines += ["", "## Opposite-label neighbors in top-5 (source-bucket leakage)", ""]
    if leakage.empty:
        lines.append("_Few/no queries with ≥2 opposite-label neighbors in top-5._")
    else:
        for _, r in leakage.iterrows():
            lines.append(
                f"- **Query:** {r['title']!r} (`{r['label_name']}`) → "
                f"{int(r['opposite_label_in_top5'])} opposite-label hits in top-5; "
                f"top-1 `{r['top1_label']}`: {r['top1_title']!r}."
            )
        lines.append(
            "\nRetrieved opposite-bucket neighbors show topical/style overlap across "
            "source classes. They do **not** prove or refute the query claim."
        )

    lines += ["", "## Same-subject / outlet-style collapse", ""]
    if style.empty:
        lines.append("_Few/no queries with ≥2 other same-subject articles in top-5._")
    else:
        for _, r in style.iterrows():
            lines.append(
                f"- **Query:** {r['title']!r} (subject `{r['subject']}`) → "
                f"{int(r['same_subject_other_in_top5'])} other same-subject articles in top-5."
            )
        lines.append(
            "\nSame-subject neighbors often share wire diction or political framing; "
            "the ranker can collapse to outlet/topic style rather than the specific claim."
        )

    lines += [
        "",
        "## Takeaway",
        "",
        "High self-retrieval scores mean the index can find an article's own passages "
        "from its title. That is necessary but not sufficient for fact-checking. "
        "ISOT labels remain source buckets.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def save_eval_bundle(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    ablations: pd.DataFrame,
    results_dir: Path | str = "results",
    index: PassageIndex | None = None,
    index_dir: Path | str = "data/retrieval_index/default",
    queries_path: Path | str | None = None,
) -> dict[str, Path]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "metrics": results_dir / "retrieval_metrics.csv",
        "detail": results_dir / "retrieval_eval_detail.csv",
        "ablations": results_dir / "retrieval_ablations.csv",
        "qualitative": results_dir / "qualitative_failures.md",
    }
    # Stable column order for README tables
    metric_cols = [
        "method",
        "n_queries",
        "article_hit@1",
        "article_hit@5",
        "article_hit@10",
        "passage_recall@1",
        "passage_recall@5",
        "passage_recall@10",
        "ndcg@1",
        "ndcg@5",
        "ndcg@10",
        "mrr",
        "article_mrr",
    ]
    summary = summary[[c for c in metric_cols if c in summary.columns]]
    summary.to_csv(paths["metrics"], index=False)
    detail.to_csv(paths["detail"], index=False)
    ablations.to_csv(paths["ablations"], index=False)
    write_qualitative_failures(detail, paths["qualitative"])

    if index is not None:
        index.save(index_dir)
        paths["index"] = Path(index_dir)

    if queries_path is None:
        queries_path = results_dir / "eval_queries.csv"
    # Rebuild query list from detail titles is lossy; caller may pass separately.
    paths["queries"] = Path(queries_path)

    meta = {
        "protocol": (
            "Title-as-claim self-retrieval: query=article title; "
            "gold=indexed chunks with the same article_id. "
            "Not human fact-check labels."
        ),
        "metrics_file": str(paths["metrics"]),
        "ablations_file": str(paths["ablations"]),
    }
    meta_path = results_dir / "retrieval_eval_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["meta"] = meta_path
    return paths
