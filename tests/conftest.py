"""Shared fixtures for evidence_retrieval unit tests.

No MiniLM / sentence-transformers downloads. No ISOT True.csv / Fake.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import normalize


@pytest.fixture
def tiny_articles() -> pd.DataFrame:
    """A few synthetic articles with overlapping vocabulary (no ISOT)."""
    rows = [
        {
            "article_id": 101,
            "title": "Federal Reserve raises interest rates amid inflation concerns",
            "text": (
                "The Federal Reserve raised interest rates by a quarter point on Wednesday "
                "as officials cited persistent inflation. Markets reacted calmly to the "
                "decision after weeks of speculation about policy tightening. Economists "
                "said further rate increases remain possible if inflation stays elevated."
            ),
            "label": 1,
            "label_name": "true",
            "subject": "politicsNews",
            "date": "2017-01-15",
        },
        {
            "article_id": 202,
            "title": "Outrage as secret cabal controls interest rates says anonymous blog",
            "text": (
                "An anonymous blog claimed a secret cabal controls interest rates and the "
                "Federal Reserve. The post offered no evidence and recycled conspiracy "
                "talking points about inflation and markets. Fact checkers later noted "
                "the claims contradict public Federal Reserve meeting records."
            ),
            "label": 0,
            "label_name": "fake",
            "subject": "News",
            "date": "2017-02-01",
        },
        {
            "article_id": 303,
            "title": "Local school board approves new lunch menu for spring semester",
            "text": (
                "The local school board approved a new lunch menu for the spring semester. "
                "Students will see more vegetables and fewer fried options. Parents attended "
                "the meeting and asked about allergy labeling and cafeteria staffing."
            ),
            "label": 1,
            "label_name": "true",
            "subject": "politicsNews",
            "date": "2017-03-10",
        },
        {
            "article_id": 404,
            "title": "Tech shares climb after inflation data cools market fears",
            "text": (
                "Tech shares climbed after inflation data cooled market fears. Investors "
                "watched the Federal Reserve for clues on interest rates. Analysts said "
                "calm markets reflected lower odds of aggressive policy tightening."
            ),
            "label": 1,
            "label_name": "true",
            "subject": "politicsNews",
            "date": "2017-04-02",
        },
    ]
    return pd.DataFrame(rows)


class FakeDenseEncoder:
    """Deterministic bag-of-words hash embeddings — no torch / MiniLM."""

    def __init__(self, model_name: str = "fake-minilm", batch_size: int = 64, dim: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self.dim = dim

    def encode(self, texts, show_progress: bool = False) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in str(text).lower().split():
                vec[hash(token) % self.dim] += 1.0
            vectors.append(vec)
        arr = np.vstack(vectors)
        # Avoid zero rows for empty strings.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return normalize(arr / norms, norm="l2").astype(np.float32)
