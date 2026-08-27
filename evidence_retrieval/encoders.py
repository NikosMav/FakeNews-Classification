"""Sparse TF-IDF and dense sentence-transformer encoders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SparseEncoder:
    """Fit-once TF-IDF encoder over passage text."""

    max_features: int = 50_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    vectorizer: TfidfVectorizer | None = field(default=None, repr=False)

    def fit(self, texts: Sequence[str]) -> "SparseEncoder":
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
        )
        self.vectorizer.fit(list(texts))
        return self

    def encode(self, texts: Sequence[str]):
        if self.vectorizer is None:
            raise RuntimeError("SparseEncoder.fit() must be called before encode().")
        return self.vectorizer.transform(list(texts))


@dataclass
class DenseEncoder:
    """Local sentence-transformers encoder (CPU-friendly by default)."""

    model_name: str = DEFAULT_DENSE_MODEL
    batch_size: int = 64
    _model: object | None = field(default=None, repr=False)

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(
        self,
        texts: Sequence[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        model = self._ensure_model()
        emb = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return normalize(np.asarray(emb, dtype=np.float32), norm="l2")
