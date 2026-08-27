#!/usr/bin/env python3
"""Paraphrase-title retrieval eval — writes results/paraphrase_*.csv only.

Does NOT modify results/retrieval_metrics.csv (title self-retrieval table stays intact).

Usage:
  python scripts/download_data.py
  python -m evidence_retrieval build   # or reuse data/retrieval_index/default
  python scripts/run_paraphrase_eval.py

Writes:
  results/paraphrase_metrics.csv
  results/paraphrase_queries.csv
  results/paraphrase_eval_detail.csv
  results/paraphrase_eval_meta.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_retrieval.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-articles", type=int, default=4000)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--index",
        default="data/retrieval_index/default",
        help="Reuse this index if present; otherwise build",
    )
    parser.add_argument("--no-save-index", action="store_true")
    args, unknown = parser.parse_known_args()

    argv = [
        "eval",
        "--paraphrase-only",
        "--data-dir",
        str(args.data_dir),
        "--results-dir",
        str(args.results_dir),
        "--n-articles",
        str(args.n_articles),
        "--max-queries",
        str(args.max_queries),
        "--index",
        str(args.index),
        "--index-out",
        str(args.index),
    ]
    if args.quiet:
        argv.append("--quiet")
    if args.no_save_index:
        argv.append("--no-save-index")
    argv.extend(unknown)
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
