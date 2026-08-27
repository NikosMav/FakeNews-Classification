"""CLI: build index, query passages, run evaluation.

Examples:
  python -m evidence_retrieval build --data-dir data --out data/retrieval_index/default
  python -m evidence_retrieval query "Federal Reserve raises interest rates" --top-k 5
  python -m evidence_retrieval eval --data-dir data --results-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evidence_retrieval.index import IndexConfig, PassageIndex


def _add_build_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("data/retrieval_index/default"))
    p.add_argument("--n-articles", type=int, default=4000)
    p.add_argument("--chunk-words", type=int, default=120)
    p.add_argument("--overlap", type=int, default=20)
    p.add_argument(
        "--fields",
        default="body",
        help="Comma-separated: body | title | title_body",
    )
    p.add_argument("--random-state", type=int, default=7)


def cmd_build(args: argparse.Namespace) -> int:
    fields = tuple(f.strip() for f in args.fields.split(",") if f.strip())
    config = IndexConfig(
        n_articles=args.n_articles,
        chunk_words=args.chunk_words,
        overlap=args.overlap,
        fields=fields,
        random_state=args.random_state,
    )
    print(
        f"Building index: n_articles={config.n_articles}, "
        f"chunk_words={config.chunk_words}, fields={config.fields}"
    )
    index = PassageIndex.build(
        data_dir=args.data_dir, config=config, show_progress=True
    )
    out = index.save(args.out)
    print(
        f"Saved {len(index.chunks):,} passages "
        f"({index.chunks['article_id'].nunique():,} articles) → {out}"
    )
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    index = PassageIndex.load(args.index)
    hits = index.query_df(
        args.text, top_k=args.top_k, method=args.method
    )
    if args.json:
        print(hits.to_json(orient="records", force_ascii=False, indent=2))
    else:
        if hits.empty:
            print("No hits.")
            return 0
        for _, row in hits.iterrows():
            print(
                f"#{int(row['rank'])}  score={row['score']:.4f}  "
                f"label={row['label_name']}  article_id={row['article_id']}"
            )
            print(f"  title: {row['title']}")
            passage = row["passage"]
            if len(passage) > 280:
                passage = passage[:277] + "..."
            print(f"  passage: {passage}")
            print()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from evidence_retrieval.eval.queries import queries_to_frame
    from evidence_retrieval.eval.run import (
        run_ablations,
        run_main_comparison,
        save_eval_bundle,
    )
    from evidence_retrieval.eval.queries import build_title_queries

    print("Running main comparison (TF-IDF vs dense vs hybrid)...")
    summary, detail, index = run_main_comparison(
        data_dir=args.data_dir,
        n_articles=args.n_articles,
        chunk_words=args.chunk_words,
        max_queries=args.max_queries,
        random_state=args.random_state,
        show_progress=not args.quiet,
    )
    print(summary.to_string(index=False))

    import pandas as pd

    if not args.skip_ablations:
        print("\nRunning ablations (chunk size / fields / method)...")
        ablations = run_ablations(
            data_dir=args.data_dir,
            n_articles=min(args.n_articles, args.ablation_articles),
            max_queries=min(args.max_queries, args.ablation_queries),
            random_state=args.random_state,
            show_progress=not args.quiet,
        )
        print(ablations.to_string(index=False))
    else:
        ablations = pd.DataFrame()

    queries = build_title_queries(
        index.chunks, max_queries=args.max_queries, random_state=args.random_state
    )
    qpath = Path(args.results_dir) / "eval_queries.csv"
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    queries_to_frame(queries).to_csv(qpath, index=False)

    paths = save_eval_bundle(
        summary=summary,
        detail=detail,
        ablations=ablations,
        results_dir=args.results_dir,
        index=index if args.save_index else None,
        index_dir=args.index_out,
        queries_path=qpath,
    )
    print("\nWrote:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evidence_retrieval",
        description=(
            "Evidence retrieval over ISOT: chunk, embed, query, evaluate. "
            "Not a fact-checker — labels are source buckets."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Chunk + embed + save an index")
    _add_build_args(p_build)
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="Retrieve ranked passages for a claim/text")
    p_query.add_argument("text", help="Query / claim / article text")
    p_query.add_argument(
        "--index",
        type=Path,
        default=Path("data/retrieval_index/default"),
    )
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument(
        "--method",
        choices=["tfidf", "dense", "hybrid"],
        default="hybrid",
    )
    p_query.add_argument("--json", action="store_true")
    p_query.set_defaults(func=cmd_query)

    p_eval = sub.add_parser(
        "eval",
        help="Reproducible eval: metrics + ablations → results/*.csv",
    )
    p_eval.add_argument("--data-dir", type=Path, default=Path("data"))
    p_eval.add_argument("--results-dir", type=Path, default=Path("results"))
    p_eval.add_argument("--n-articles", type=int, default=4000)
    p_eval.add_argument("--chunk-words", type=int, default=120)
    p_eval.add_argument("--max-queries", type=int, default=300)
    p_eval.add_argument("--ablation-articles", type=int, default=2500)
    p_eval.add_argument("--ablation-queries", type=int, default=200)
    p_eval.add_argument("--random-state", type=int, default=7)
    p_eval.add_argument("--skip-ablations", action="store_true")
    p_eval.add_argument("--save-index", action="store_true", default=True)
    p_eval.add_argument("--no-save-index", action="store_false", dest="save_index")
    p_eval.add_argument(
        "--index-out",
        type=Path,
        default=Path("data/retrieval_index/default"),
    )
    p_eval.add_argument("--quiet", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
