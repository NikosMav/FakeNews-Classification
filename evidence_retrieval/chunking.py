"""Article → overlapping word-window passages."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import pandas as pd

_WORD_RE = re.compile(r"\S+")


def chunk_text(
    text: str,
    chunk_words: int = 120,
    overlap: int = 20,
    min_tail_words: int | None = None,
) -> list[str]:
    """Split text into overlapping word windows."""
    words = _WORD_RE.findall(text or "")
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]

    step = max(chunk_words - overlap, 1)
    min_tail = min_tail_words if min_tail_words is not None else max(20, chunk_words // 4)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_words]
        if len(piece) < min_tail and chunks:
            chunks[-1] = chunks[-1] + " " + " ".join(piece)
            break
        chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return chunks


def articles_to_chunks(
    articles: pd.DataFrame,
    chunk_words: int = 120,
    overlap: int = 20,
    fields: Sequence[str] = ("body",),
) -> pd.DataFrame:
    """Chunk articles into passages with source metadata.

    ``fields`` controls what is indexed (one mode at a time is typical):
    - ``body``: article text windows only
    - ``title``: title as a single passage per article
    - ``title_body``: ``"{title}. {body_chunk}"`` for each body window
    """
    field_mode = _resolve_field_mode(fields)
    rows: list[dict] = []

    for _, art in articles.iterrows():
        title = str(art.get("title", "")).strip()
        body = str(art.get("text", "")).strip()
        article_id = int(art["article_id"])
        base_meta = {
            "article_id": article_id,
            "title": title,
            "label": int(art["label"]),
            "label_name": art["label_name"],
            "subject": str(art.get("subject", "")),
            "date": str(art.get("date", "")),
        }

        if field_mode == "title":
            passage = title or body[:500]
            if not passage:
                continue
            rows.append(
                {
                    **base_meta,
                    "chunk_id": f"{article_id}:0",
                    "chunk_index": 0,
                    "field": "title",
                    "passage": passage,
                }
            )
            continue

        passages = chunk_text(body, chunk_words=chunk_words, overlap=overlap)
        if not passages:
            fallback = title or body
            if not fallback:
                continue
            passages = [fallback]

        for i, passage in enumerate(passages):
            if field_mode == "title_body" and title:
                text = f"{title}. {passage}"
                field_name = "title_body"
            else:
                text = passage
                field_name = "body"
            rows.append(
                {
                    **base_meta,
                    "chunk_id": f"{article_id}:{i}",
                    "chunk_index": i,
                    "field": field_name,
                    "passage": text,
                }
            )

    return pd.DataFrame(rows)


def _resolve_field_mode(fields: Iterable[str]) -> str:
    field_set = {f.strip() for f in fields}
    if "title_body" in field_set:
        return "title_body"
    if field_set == {"title"} or "title" in field_set and "body" not in field_set:
        return "title"
    return "body"
