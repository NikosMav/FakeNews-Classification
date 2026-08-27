#!/usr/bin/env python3
"""Rebuild evidence_retrieval.ipynb as a thin walkthrough over the Python package."""

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


def build() -> nbf.NotebookNode:
    cells = [
        md(
            """
# Evidence Retrieval Walkthrough

Thin notebook over the `evidence_retrieval` package. For metrics, ablations, and the
full case study, see the root **README** and `python scripts/run_retrieval_eval.py`.

**Not a fact-checker.** ISOT labels are source buckets; nearest neighbors ≠ verification.
"""
        ),
        md("## Build or load an index"),
        code(
            """
from pathlib import Path
from evidence_retrieval.index import IndexConfig, PassageIndex

INDEX_DIR = Path("data/retrieval_index/default")

if (INDEX_DIR / "chunks.parquet").exists():
    index = PassageIndex.load(INDEX_DIR)
    print(f"Loaded index: {len(index.chunks):,} passages")
else:
    config = IndexConfig(n_articles=2000, chunk_words=120, fields=("body",))
    index = PassageIndex.build(data_dir="data", config=config, show_progress=True)
    index.save(INDEX_DIR)
    print(f"Built + saved index → {INDEX_DIR}")
"""
        ),
        md("## Query by claim / article text"),
        code(
            """
query = "Federal Reserve interest rate decision and markets"
hits = index.query_df(query, top_k=5, method="hybrid")
display(hits[["rank", "score", "label_name", "title", "passage"]])
"""
        ),
        md(
            """
## Optional: classify-then-retrieve vs retrieve-first

Uses a TF-IDF classifier trained off the indexed sample, then contrasts predicted
source-bucket vs neighborhood vote among retrieved passages. Still not a fact-check.
"""
        ),
        code(
            """
from evidence_retrieval.workflows import compare_workflows

comparison = compare_workflows(index, data_dir="data", n_demo=6)
display(comparison)
"""
        ),
        md(
            """
## Reproduce the README metrics table

```bash
python scripts/run_retrieval_eval.py
```
"""
        ),
    ]
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
