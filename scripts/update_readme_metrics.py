#!/usr/bin/env python3
"""Insert results/retrieval_metrics.csv into the README metrics table markers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
METRICS = ROOT / "results" / "retrieval_metrics.csv"
START = "<!-- METRICS_TABLE_START -->"
END = "<!-- METRICS_TABLE_END -->"


def fmt(x: float) -> str:
    return f"{x:.4f}"


def build_table(df: pd.DataFrame) -> str:
    cols = [
        ("method", "Method", None),
        ("article_hit@1", "Article Hit@1", fmt),
        ("article_hit@5", "Article Hit@5", fmt),
        ("article_hit@10", "Article Hit@10", fmt),
        ("passage_recall@5", "Passage Recall@5", fmt),
        ("ndcg@5", "nDCG@5", fmt),
        ("ndcg@10", "nDCG@10", fmt),
        ("mrr", "MRR", fmt),
    ]
    header = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "| " + " | ".join("---" if c[2] is None else "---:" for c in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for key, _, formatter in cols:
            val = row[key]
            cells.append(str(val) if formatter is None else formatter(float(val)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    if not METRICS.exists():
        raise SystemExit(f"Missing {METRICS}; run scripts/run_retrieval_eval.py first")
    df = pd.read_csv(METRICS)
    table = build_table(df)
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit("README missing METRICS_TABLE markers")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    new = before + START + "\n" + table + "\n" + END + after
    # Drop the placeholder sentence if present
    new = new.replace(
        "**Placeholder until the eval run commits real numbers** — if you are reading a checkout\n"
        "before that run finishes, regenerate locally.\n\n",
        "",
    )
    README.write_text(new, encoding="utf-8")
    print(f"Updated README metrics table from {METRICS}")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
