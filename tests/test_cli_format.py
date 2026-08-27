"""CLI formatting tests — fixture output for README (no model download)."""

from __future__ import annotations

from evidence_retrieval.cli import build_parser, format_query_hits
from evidence_retrieval.index import Hit


def _sample_hits() -> list[Hit]:
    long_passage = (
        "The Federal Reserve raised interest rates by a quarter point on Wednesday "
        "as officials cited persistent inflation. Markets reacted calmly to the "
        "decision after weeks of speculation about policy tightening. Economists "
        "said further rate increases remain possible if inflation stays elevated "
        "through the next quarter and beyond the summer outlook window."
    )
    return [
        Hit(
            rank=1,
            score=0.0328,
            chunk_id="101:0",
            article_id=101,
            title="Federal Reserve raises interest rates amid inflation concerns",
            label=1,
            label_name="true",
            subject="politicsNews",
            date="2017-01-15",
            passage=long_passage,
            method="hybrid",
        ),
        Hit(
            rank=2,
            score=0.0311,
            chunk_id="202:0",
            article_id=202,
            title="Outrage as secret cabal controls interest rates says anonymous blog",
            label=0,
            label_name="fake",
            subject="News",
            date="2017-02-01",
            passage=(
                "An anonymous blog claimed a secret cabal controls interest rates and the "
                "Federal Reserve. The post offered no evidence."
            ),
            method="hybrid",
        ),
        Hit(
            rank=3,
            score=0.0164,
            chunk_id="404:0",
            article_id=404,
            title="Tech shares climb after inflation data cools market fears",
            label=1,
            label_name="true",
            subject="politicsNews",
            date="2017-04-02",
            passage=(
                "Tech shares climbed after inflation data cooled market fears. Investors "
                "watched the Federal Reserve for clues on interest rates."
            ),
            method="hybrid",
        ),
    ]


def test_format_query_hits_includes_title_label_score_passage():
    text = format_query_hits(
        _sample_hits(),
        query_text="Federal Reserve raises interest rates",
        method="hybrid",
        passage_chars=120,
    )
    assert 'query: "Federal Reserve raises interest rates"' in text
    assert "method: hybrid" in text
    assert "source bucket" in text.lower() or "ISOT source bucket" in text
    assert "#1  score=0.0328  label=true  article_id=101" in text
    assert "title: Federal Reserve raises interest rates amid inflation concerns" in text
    assert "passage:" in text
    assert "..." in text  # long passage truncated
    assert "#2  score=0.0311  label=fake" in text
    assert "fact-check" in text.lower() or "not a fact-check" in text


def test_format_query_hits_empty():
    assert "No hits." in format_query_hits([])


def test_cli_help_mentions_not_fact_checker():
    parser = build_parser()
    help_text = parser.format_help()
    assert "NOT a fact-checker" in help_text or "not a fact-checker" in help_text.lower()
    assert "build" in help_text
    assert "query" in help_text
    assert "eval" in help_text


def test_cli_help_is_windows_console_safe():
    """Default Windows PowerShell commonly uses cp1252."""
    build_parser().format_help().encode("cp1252")


def test_fixture_example_matches_formatter():
    """Committed README example is generated from this formatter + fixture hits."""
    from pathlib import Path

    example_path = Path(__file__).resolve().parents[1] / "examples" / "query_output_fixture.txt"
    generated = format_query_hits(
        _sample_hits(),
        query_text="Federal Reserve raises interest rates",
        method="hybrid",
        passage_chars=280,
    )
    assert example_path.exists(), "examples/query_output_fixture.txt must be committed"
    assert example_path.read_text(encoding="utf-8") == generated
