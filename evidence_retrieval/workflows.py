"""Optional: classify-then-retrieve vs retrieve-first on held-out indexed docs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from evidence_retrieval.data import load_isot
from evidence_retrieval.index import PassageIndex


@dataclass
class WorkflowRow:
    title: str
    gold_label: str
    classify_label: str
    retrieve_first_vote: str
    neighbor_true: int
    neighbor_fake: int
    top_neighbor_title: str
    top_neighbor_score: float


def compare_workflows(
    index: PassageIndex,
    data_dir: str | Path = "data",
    n_demo: int = 8,
    top_k: int = 5,
    train_cap: int = 8000,
    random_state: int = 7,
) -> pd.DataFrame:
    """Contrast document classification with passage neighborhood votes.

    Neither workflow is a fact-checker. Labels remain source buckets.
    """
    df = load_isot(data_dir)
    indexed_ids = set(index.chunks["article_id"].unique().tolist())
    train_pool = df[~df["article_id"].isin(indexed_ids)].copy()
    train_pool["body"] = (train_pool["title"] + " " + train_pool["text"]).str.strip()
    train_cap = min(train_cap, len(train_pool))
    train_pool = train_pool.sample(n=train_cap, random_state=random_state)

    X_train, _, y_train, _ = train_test_split(
        train_pool["body"],
        train_pool["label"],
        test_size=0.2,
        random_state=random_state,
        stratify=train_pool["label"],
    )
    tfidf = TfidfVectorizer(max_features=40_000, ngram_range=(1, 2), min_df=2)
    X_vec = tfidf.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=random_state)
    clf.fit(X_vec, y_train)

    meta = index.chunks.drop_duplicates("article_id")
    demo = meta.sample(n=min(n_demo, len(meta)), random_state=random_state)
    full = df.set_index("article_id")
    rows = []
    for _, doc in demo.iterrows():
        aid = int(doc["article_id"])
        art = full.loc[aid]
        text = f"{art['title']}. {art['text']}"
        pred = int(clf.predict(tfidf.transform([text]))[0])
        pred_name = "true" if pred == 1 else "fake"

        hits = index.query(
            text[:1500],
            top_k=top_k,
            method="hybrid",
            exclude_article_ids={aid},
        )
        vote_true = sum(1 for h in hits if h.label_name == "true")
        vote_fake = sum(1 for h in hits if h.label_name == "fake")
        if vote_true > vote_fake:
            vote = "true"
        elif vote_fake > vote_true:
            vote = "fake"
        else:
            vote = "tie"

        rows.append(
            WorkflowRow(
                title=str(art["title"])[:100],
                gold_label=str(art["label_name"]),
                classify_label=pred_name,
                retrieve_first_vote=vote,
                neighbor_true=vote_true,
                neighbor_fake=vote_fake,
                top_neighbor_title=(hits[0].title[:80] if hits else ""),
                top_neighbor_score=(hits[0].score if hits else float("nan")),
            )
        )

    return pd.DataFrame([r.__dict__ for r in rows])
