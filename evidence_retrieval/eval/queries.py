"""Labeled query construction for retrieval evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class EvalQuery:
    query_id: str
    article_id: int
    query_text: str
    gold_chunk_ids: list[str]
    gold_article_ids: list[int]
    label_name: str
    subject: str
    title: str


def build_title_queries(
    chunks: pd.DataFrame,
    max_queries: int = 300,
    min_title_words: int = 4,
    random_state: int = 7,
) -> list[EvalQuery]:
    """Title-as-claim → gold = same article's indexed body passages.

    Justified synthetic protocol when human qrels are unavailable. See package docs.
    """
    body_chunks = chunks[chunks["field"].isin(["body", "title_body"])]
    if body_chunks.empty:
        body_chunks = chunks

    article_ids = body_chunks.groupby("article_id").size().reset_index(name="n_chunks")
    meta = chunks.drop_duplicates("article_id")[
        ["article_id", "title", "label_name", "subject"]
    ]
    candidates = article_ids.merge(meta, on="article_id")
    candidates["title_words"] = candidates["title"].str.split().str.len().fillna(0)
    candidates = candidates[candidates["title_words"] >= min_title_words]
    candidates = candidates[candidates["title"].str.strip().astype(bool)]

    if candidates.empty:
        raise RuntimeError("No eligible query articles — check chunk fields/titles.")

    n = min(max_queries, len(candidates))
    sampled = candidates.sample(n=n, random_state=random_state).reset_index(drop=True)
    gold_map = body_chunks.groupby("article_id")["chunk_id"].apply(list).to_dict()

    queries: list[EvalQuery] = []
    for _, row in sampled.iterrows():
        aid = int(row["article_id"])
        gold = [str(c) for c in gold_map.get(aid, [])]
        if not gold:
            continue
        queries.append(
            EvalQuery(
                query_id=f"title-{aid}",
                article_id=aid,
                query_text=str(row["title"]).strip(),
                gold_chunk_ids=gold,
                gold_article_ids=[aid],
                label_name=str(row["label_name"]),
                subject=str(row["subject"]),
                title=str(row["title"]),
            )
        )
    return queries


def queries_to_frame(queries: list[EvalQuery]) -> pd.DataFrame:
    rows = []
    for q in queries:
        rows.append(
            {
                "query_id": q.query_id,
                "article_id": q.article_id,
                "query_text": q.query_text,
                "gold_chunk_ids": "|".join(q.gold_chunk_ids),
                "gold_article_ids": "|".join(str(x) for x in q.gold_article_ids),
                "label_name": q.label_name,
                "subject": q.subject,
                "title": q.title,
                "n_gold_chunks": len(q.gold_chunk_ids),
            }
        )
    return pd.DataFrame(rows)
