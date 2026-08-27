#!/usr/bin/env python3
"""Build evidence_retrieval.ipynb — Chapter 2: nearest-neighbor evidence retrieval over ISOT."""

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


def build() -> nbf.NotebookNode:
    cells = []

    cells.append(
        md(
            """
# Evidence Retrieval over ISOT (Chapter 2)

**Author:** [Nikos Mavrapidis](https://github.com/NikosMav)
**Companion to:** `fake_news_classification.ipynb` (Chapter 1 — supervised classification)

This notebook adds an **evidence-retrieval** path on the same ISOT corpus:

1. Chunk articles into passages
2. Embed passages with a local CPU-friendly sentence model
3. Given a query / claim / article, return the nearest passages with source metadata
   (`title`, `label`, similarity `score`, …)

### Honest framing (read this)

| This is | This is **not** |
| --- | --- |
| Nearest-neighbor passage retrieval over a fixed local corpus | A production fact-checker |
| A retrieval demo adjacent to RAG (the **R**) | RAG-over-the-web / live search |
| Useful for “show me related evidence from ISOT” | Proof that a claim is true or false |

ISOT labels are **source buckets** (Reuters-style “true” vs. unreliable-outlet “fake”),
not claim-level verdicts. A retrieved “fake” neighbor does **not** prove a claim is false;
a “true” neighbor does **not** prove it is true. Similarity ≠ verification.

Generation (the **G** in RAG) is out of scope here except for optionally quoting the
retrieved passage text as extractive evidence.
"""
        )
    )

    cells.append(
        md(
            """
## Setup

Same dataset as Chapter 1. If CSVs are missing:

```bash
python scripts/download_data.py
```

Default embedding model: `sentence-transformers/all-MiniLM-L6-v2` (local, CPU-friendly, no paid API).
"""
        )
    )

    cells.append(
        code(
            """
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from sentence_transformers import SentenceTransformer

%matplotlib inline
sns.set_theme(style="whitegrid")
pd.set_option("display.max_colwidth", 160)

DATA_DIR = Path("data")
TRUE_PATH = DATA_DIR / "True.csv"
FAKE_PATH = DATA_DIR / "Fake.csv"
INDEX_DIR = DATA_DIR / "retrieval_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 7
# CPU-friendly defaults: stratified sample of articles, short passages.
N_ARTICLES = 4000
CHUNK_WORDS = 120
CHUNK_OVERLAP = 20
TOP_K = 5
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64

if not TRUE_PATH.exists() or not FAKE_PATH.exists():
    raise FileNotFoundError(
        "Missing data/True.csv and/or data/Fake.csv. "
        "Run: python scripts/download_data.py"
    )

print("Data files found:")
print(f"  {TRUE_PATH} ({TRUE_PATH.stat().st_size:,} bytes)")
print(f"  {FAKE_PATH} ({FAKE_PATH.stat().st_size:,} bytes)")
"""
        )
    )

    cells.append(
        md(
            """
## Load corpus

Reuse the same ISOT files and binary labels as Chapter 1 (`1` = true / Reuters bucket,
`0` = fake / unreliable-outlet bucket).
"""
        )
    )

    cells.append(
        code(
            """
df_true = pd.read_csv(TRUE_PATH)
df_fake = pd.read_csv(FAKE_PATH)
df_true = df_true.copy()
df_fake = df_fake.copy()
df_true["label"] = 1
df_fake["label"] = 0
df_true["label_name"] = "true"
df_fake["label_name"] = "fake"

df_all = pd.concat([df_true, df_fake], ignore_index=True)
df_all["article_id"] = np.arange(len(df_all))
df_all["title"] = df_all["title"].fillna("").astype(str)
df_all["text"] = df_all["text"].fillna("").astype(str)
df_all["subject"] = df_all.get("subject", pd.Series([""] * len(df_all))).fillna("").astype(str)
df_all["date"] = df_all.get("date", pd.Series([""] * len(df_all))).fillna("").astype(str)

print(df_all["label_name"].value_counts().to_string())
print("Total articles:", len(df_all))
display(df_all.head(2))
"""
        )
    )

    cells.append(
        md(
            """
## Sample an indexable subset

Full ISOT is ~45k articles. For a clone-and-run CPU demo we index a **stratified sample**
(`N_ARTICLES`, default 4000). Raise it locally if you want denser coverage; the pipeline
is the same.
"""
        )
    )

    cells.append(
        code(
            """
n_per_class = N_ARTICLES // 2
sampled_parts = []
for label_value, group in df_all.groupby("label"):
    take = min(n_per_class, len(group))
    sampled_parts.append(group.sample(n=take, random_state=RANDOM_STATE))

corpus = pd.concat(sampled_parts, ignore_index=True).sample(
    frac=1.0, random_state=RANDOM_STATE
).reset_index(drop=True)

print(f"Indexed articles: {len(corpus)} "
      f"(true={int((corpus.label == 1).sum())}, fake={int((corpus.label == 0).sum())})")
"""
        )
    )

    cells.append(
        md(
            """
## Chunk articles into passages

Retrieval works better over **passages** than whole articles: a claim usually aligns with
a local span, not an entire wire story. We split on whitespace into overlapping word
windows (`CHUNK_WORDS` / `CHUNK_OVERLAP`).
"""
        )
    )

    cells.append(
        code(
            """
def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = re.findall(r"\\S+", text)
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = max(chunk_words - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_words]
        if len(piece) < max(20, chunk_words // 4) and chunks:
            # fold a tiny trailing remnant into the previous chunk
            chunks[-1] = chunks[-1] + " " + " ".join(piece)
            break
        chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return chunks


rows = []
for _, art in corpus.iterrows():
    body = art["text"].strip()
    title = art["title"].strip()
    # Prefer body chunks; if body is empty, fall back to the title as a single passage.
    passages = chunk_text(body) if body else ([title] if title else [])
    for i, passage in enumerate(passages):
        rows.append(
            {
                "chunk_id": f"{art['article_id']}:{i}",
                "article_id": int(art["article_id"]),
                "chunk_index": i,
                "title": title,
                "label": int(art["label"]),
                "label_name": art["label_name"],
                "subject": art["subject"],
                "date": art["date"],
                "passage": passage,
            }
        )

chunks_df = pd.DataFrame(rows)
print(f"Passages: {len(chunks_df):,} from {chunks_df['article_id'].nunique():,} articles")
print(f"Mean passage words: {chunks_df['passage'].str.split().str.len().mean():.1f}")
display(chunks_df.head(3))
"""
        )
    )

    cells.append(
        md(
            """
## Embed passages (local sentence-transformers)

`all-MiniLM-L6-v2` runs on CPU and needs no API key. Embeddings are L2-normalized so
cosine similarity is a plain dot product. Results are cached under `data/retrieval_index/`
so re-runs skip the encode step when inputs match.
"""
        )
    )

    cells.append(
        code(
            """
def cache_key() -> str:
    payload = {
        "model": EMBED_MODEL_NAME,
        "n_articles": N_ARTICLES,
        "chunk_words": CHUNK_WORDS,
        "overlap": CHUNK_OVERLAP,
        "random_state": RANDOM_STATE,
        "article_ids": chunks_df["article_id"].tolist(),
        "n_chunks": len(chunks_df),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return digest


key = cache_key()
emb_path = INDEX_DIR / f"embeddings_{key}.npy"
meta_path = INDEX_DIR / f"chunks_{key}.parquet"

if emb_path.exists() and meta_path.exists():
    print(f"Loading cached embeddings: {emb_path.name}")
    embeddings = np.load(emb_path)
    chunks_df = pd.read_parquet(meta_path)
else:
    print(f"Loading model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = chunks_df["passage"].tolist()
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    print(f"Embedded {len(texts):,} passages in {elapsed:.1f}s "
          f"({len(texts) / max(elapsed, 1e-6):.1f} passages/s)")
    np.save(emb_path, embeddings)
    chunks_df.to_parquet(meta_path, index=False)
    print(f"Cached → {emb_path.name}, {meta_path.name}")

embeddings = normalize(embeddings.astype(np.float32), norm="l2")
print("Embedding matrix:", embeddings.shape)
"""
        )
    )

    cells.append(
        md(
            """
## Build a nearest-neighbor index

`sklearn.neighbors.NearestNeighbors` with cosine metric keeps the demo dependency-light
(no FAISS required). For larger corpora you would swap in FAISS / Annoy / a vector DB;
the retrieval API stays the same.
"""
        )
    )

    cells.append(
        code(
            """
nn_index = NearestNeighbors(
    n_neighbors=min(50, len(chunks_df)),
    metric="cosine",
    algorithm="brute",
)
nn_index.fit(embeddings)

# Keep the encoder around for queries (reload if we used cache-only path above).
if "model" not in globals() or model is None:
    model = SentenceTransformer(EMBED_MODEL_NAME)


def retrieve(
    query: str,
    top_k: int = TOP_K,
    exclude_article_ids: set[int] | None = None,
) -> pd.DataFrame:
    \"\"\"Return top-k nearest passages for a free-text query/claim/article.\"\"\"
    q = (query or "").strip()
    if not q:
        raise ValueError("query must be non-empty")

    q_emb = model.encode(
        [q],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Over-fetch so we can drop self-matches / excluded articles and still fill top_k.
    fetch = min(len(chunks_df), max(top_k * 10, top_k + 20))
    distances, indices = nn_index.kneighbors(q_emb, n_neighbors=fetch)
    distances, indices = distances[0], indices[0]

    exclude = exclude_article_ids or set()
    rows = []
    for dist, idx in zip(distances, indices):
        row = chunks_df.iloc[int(idx)]
        if int(row["article_id"]) in exclude:
            continue
        # sklearn cosine distance = 1 - cosine similarity
        score = float(1.0 - dist)
        rows.append(
            {
                "rank": len(rows) + 1,
                "score": score,
                "title": row["title"],
                "label_name": row["label_name"],
                "label": int(row["label"]),
                "subject": row["subject"],
                "date": row["date"],
                "article_id": int(row["article_id"]),
                "chunk_id": row["chunk_id"],
                "passage": row["passage"],
            }
        )
        if len(rows) >= top_k:
            break

    return pd.DataFrame(rows)


def show_hits(hits: pd.DataFrame, query: str) -> None:
    print(f"Query: {query[:240]}{'…' if len(query) > 240 else ''}")
    print(f"Hits: {len(hits)}")
    display(
        hits[
            ["rank", "score", "label_name", "title", "subject", "article_id", "passage"]
        ]
    )


print("Index ready.")
"""
        )
    )

    cells.append(
        md(
            """
## Demo: retrieve evidence for a claim / query

Try a few free-text claims. Inspect **score**, **title**, **label_name**, and the
**passage** — treat labels as source metadata, not truth verdicts.
"""
        )
    )

    cells.append(
        code(
            """
demo_queries = [
    "US president meets foreign leaders to discuss trade sanctions and diplomacy",
    "Celebrity conspiracy claims about a secret government plot",
    "Federal Reserve interest rate decision and stock market reaction",
]

for q in demo_queries:
    hits = retrieve(q, top_k=TOP_K)
    show_hits(hits, q)
    print("-" * 80)
"""
        )
    )

    cells.append(
        md(
            """
## Label mix among neighbors (sanity check, not a verdict)

For each query, look at how many retrieved neighbors come from the `true` vs `fake`
buckets. This is a **descriptive** view of the neighborhood — not a classifier and not
a fact-check.
"""
        )
    )

    cells.append(
        code(
            """
summary_rows = []
for q in demo_queries:
    hits = retrieve(q, top_k=TOP_K)
    summary_rows.append(
        {
            "query": q[:80] + ("…" if len(q) > 80 else ""),
            "n_true_neighbors": int((hits["label_name"] == "true").sum()),
            "n_fake_neighbors": int((hits["label_name"] == "fake").sum()),
            "mean_score": float(hits["score"].mean()) if len(hits) else float("nan"),
            "top_title": hits.iloc[0]["title"] if len(hits) else "",
        }
    )

neighbor_summary = pd.DataFrame(summary_rows)
display(neighbor_summary)

fig, ax = plt.subplots(figsize=(8, 3.5))
x = np.arange(len(neighbor_summary))
ax.bar(x - 0.18, neighbor_summary["n_true_neighbors"], width=0.36, label="true bucket")
ax.bar(x + 0.18, neighbor_summary["n_fake_neighbors"], width=0.36, label="fake bucket")
ax.set_xticks(x)
ax.set_xticklabels([f"Q{i+1}" for i in x])
ax.set_ylabel("Neighbor count")
ax.set_title("Source-bucket mix among top-k retrieved passages")
ax.legend()
plt.tight_layout()
plt.show()
"""
        )
    )

    cells.append(
        md(
            """
## Optional: classify then retrieve vs retrieve first

Two workflows on the same held-out articles (from the indexed sample):

1. **Classify → retrieve:** train a quick TF-IDF + logistic regression on *non-indexed*
   articles, predict a label for a held-out indexed article, then retrieve nearest
   passages (excluding the article itself).
2. **Retrieve first:** skip the classifier; retrieve neighbors and report the majority
   source-bucket among them as a naive “neighborhood vote.”

Neither workflow is a fact-checker. The point is to contrast **document classification**
(Chapter 1 skill) with **passage retrieval** (this chapter).
"""
        )
    )

    cells.append(
        code(
            """
# Train a lightweight classifier on articles NOT in the retrieval index sample.
indexed_ids = set(corpus["article_id"].tolist())
train_pool = df_all[~df_all["article_id"].isin(indexed_ids)].copy()
train_pool["body"] = (train_pool["title"] + " " + train_pool["text"]).str.strip()

# Cap training size for CPU friendliness while keeping both classes.
train_cap = min(8000, len(train_pool))
train_pool = train_pool.sample(n=train_cap, random_state=RANDOM_STATE)

X_train, X_val, y_train, y_val = train_test_split(
    train_pool["body"],
    train_pool["label"],
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=train_pool["label"],
)

tfidf = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2)
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)

clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE)
clf.fit(X_train_vec, y_train)
val_pred = clf.predict(X_val_vec)
print(
    f"Side classifier (TF-IDF + LR) on non-indexed pool — "
    f"val accuracy={accuracy_score(y_val, val_pred):.4f}, "
    f"F1={f1_score(y_val, val_pred):.4f}"
)

# Pick a few held-out indexed articles as demo documents.
demo_docs = corpus.sample(n=4, random_state=RANDOM_STATE).reset_index(drop=True)
comparison_rows = []

for _, doc in demo_docs.iterrows():
    doc_text = f"{doc['title']}. {doc['text']}"
    pred = int(clf.predict(tfidf.transform([doc_text]))[0])
    pred_name = "true" if pred == 1 else "fake"

    # Classify → retrieve neighbors (exclude self).
    hits_after = retrieve(
        doc_text[:1500],
        top_k=TOP_K,
        exclude_article_ids={int(doc["article_id"])},
    )
    # Retrieve-first neighborhood vote.
    vote_true = int((hits_after["label_name"] == "true").sum())
    vote_fake = int((hits_after["label_name"] == "fake").sum())
    neighborhood_vote = (
        "true" if vote_true > vote_fake else ("fake" if vote_fake > vote_true else "tie")
    )

    comparison_rows.append(
        {
            "title": doc["title"][:90],
            "gold_label": doc["label_name"],
            "classify_then_label": pred_name,
            "retrieve_first_vote": neighborhood_vote,
            "neighbor_true": vote_true,
            "neighbor_fake": vote_fake,
            "top_neighbor_title": hits_after.iloc[0]["title"][:70] if len(hits_after) else "",
            "top_neighbor_score": float(hits_after.iloc[0]["score"]) if len(hits_after) else float("nan"),
        }
    )

    print("=" * 80)
    print(f"Article: {doc['title'][:100]}")
    print(f"Gold source-bucket: {doc['label_name']} | Classify→label: {pred_name} | "
          f"Retrieve-first vote: {neighborhood_vote}")
    show_hits(hits_after, doc_text[:200])

comparison_df = pd.DataFrame(comparison_rows)
print("\\nSide-by-side summary:")
display(comparison_df)
"""
        )
    )

    cells.append(
        md(
            """
## Tiny extractive “answer” (optional, not generative RAG)

The cheapest grounded response is to **quote** the top retrieved passage. No LLM, no
hallucinated synthesis — just the corpus span that scored highest.
"""
        )
    )

    cells.append(
        code(
            """
def extractive_quote(query: str, top_k: int = 1) -> str:
    hits = retrieve(query, top_k=top_k)
    if hits.empty:
        return "No passages retrieved."
    top = hits.iloc[0]
    return (
        f"Query: {query}\\n\\n"
        f"Top passage (score={top['score']:.3f}, source-bucket={top['label_name']}, "
        f"title={top['title']!r}):\\n\\n"
        f"\"{top['passage']}\""
    )


print(extractive_quote(demo_queries[0], top_k=1))
"""
        )
    )

    cells.append(
        md(
            """
## Limitations

1. **Not a fact-checker.** Nearest neighbors show *related* ISOT passages. Source-bucket
   labels are not claim-level truth.
2. **Closed corpus.** Index is a stratified ISOT sample — not the open web, not news APIs.
3. **Bi-encoder only.** `all-MiniLM-L6-v2` + cosine NN is a solid baseline; no cross-encoder
   re-ranker, hybrid BM25, or learned retrieval metrics (nDCG / recall@k with judgments).
4. **Style leakage still exists.** Neighbors may match writing style / outlet cues rather
   than propositional content.
5. **Generation out of scope.** Quoting a passage is extractive evidence display, not a
   generative RAG answer.

## Relation to Chapter 1 and to RAG

| Chapter 1 — Classification | Chapter 2 — Retrieval (this notebook) | Full RAG |
| --- | --- | --- |
| Document → class label | Query → ranked passages + metadata | Query → retrieve → generate answer |
| Needs labeled training set | Needs an indexable corpus | Needs index + generator (+ usually eval) |
| Metrics: accuracy, F1, ROC-AUC | Metrics (in production): recall@k, nDCG | + groundedness / answer quality |

Shared foundations: text → vectors → careful evaluation. Classification accuracy on ISOT
does **not** imply retrieval quality; retrieving a “fake”-labeled neighbor does **not**
refute a claim.
"""
        )
    )

    cells.append(
        md(
            """
## How to re-run

```bash
python scripts/download_data.py
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook evidence_retrieval.ipynb
```

Rebuild this notebook from source:

```bash
python scripts/build_retrieval_notebook.py
```
"""
        )
    )

    nb = new_notebook(cells=cells)
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


def main():
    out = Path(__file__).resolve().parents[1] / "evidence_retrieval.ipynb"
    nb = build()
    nbf.write(nb, out)
    print(f"Wrote {out} with {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
