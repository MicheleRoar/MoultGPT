# llm/retrieval/embedder.py
"""
Embedding backends for the retrieval module — deliberately two, at opposite
ends of the "local vs remote" trade-off the rest of llm/ already makes
(see backend/providers.py's module docstring for the original version of
this trade-off, applied there to chat completions rather than embeddings).

TfidfEmbedder
    Sparse, local, zero new dependencies — scikit-learn is already a
    llm/requirements.txt dependency (pipeline/summarization.py uses it for
    TF-IDF + KMeans clustering). Cosine similarity over TF-IDF vectors is
    lexical retrieval, not true semantic retrieval, but it's a real, fast,
    dependency-free baseline that needs no API key.

MistralRemoteEmbedder
    Dense, semantic, calls Mistral's embeddings API (`mistral-embed`) —
    reuses the same MISTRAL_API_KEY already used for chat completions, no
    new secret to configure. A local sentence-transformers encoder would
    need torch, which llm/requirements.txt deliberately excludes (see that
    file's header comment) — this keeps the same "remote API, no local
    GPU/weights" design instead of quietly reintroducing torch through the
    back door for embeddings only.

Both implement the same tiny interface — `.fit(texts)` (a real fit for
TfidfEmbedder, a no-op for the remote one, since that model is pretrained)
and `.encode(texts) -> np.ndarray [n, dim]` — so VectorIndex and the CLIs
don't need to know which one they're using.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import requests


class ProviderError(RuntimeError):
    """Raised when the remote embedding call fails or returns something unusable."""


class TfidfEmbedder:
    """Local, sparse, zero-new-dependency baseline embedder."""

    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            # Fit-on-first-use fallback so a single encode() call still works
            # without a separate fit() step — at the cost of the vocabulary
            # only reflecting whatever text is passed in *this* call. Callers
            # that need query and corpus to share a vocabulary (i.e. any real
            # retrieval use) should call fit() on the corpus explicitly first.
            self.fit(texts)
        return self._vectorizer.transform(texts).toarray().astype("float32")


class MistralRemoteEmbedder:
    """Dense, semantic embedder via Mistral's remote embeddings API."""

    name = "mistral-embed"
    MODEL_ID = "mistral-embed"
    API_URL = "https://api.mistral.ai/v1/embeddings"
    TIMEOUT_SEC = 60

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")

    def fit(self, texts: List[str]) -> None:
        pass  # nothing to fit — the remote model is already trained

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.api_key:
            raise ProviderError(
                "MISTRAL_API_KEY is not set — required for MistralRemoteEmbedder. "
                "Use --embedder tfidf for a dependency/key-free baseline instead."
            )
        resp = requests.post(
            self.API_URL,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.MODEL_ID, "input": texts},
            timeout=self.TIMEOUT_SEC,
        )
        if resp.status_code != 200:
            raise ProviderError(f"Mistral embeddings error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        try:
            # Sort by "index" — batch APIs aren't guaranteed to echo results
            # back in request order, and silently zipping a shuffled result
            # to the wrong chunk would be a very quiet correctness bug.
            vectors = [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]
        except (KeyError, IndexError) as e:
            raise ProviderError(f"Unexpected Mistral embeddings response shape: {data}") from e
        return np.array(vectors, dtype="float32")


EMBEDDERS = {
    "tfidf": TfidfEmbedder,
    "mistral-embed": MistralRemoteEmbedder,
}


def get_embedder(name: str):
    if name not in EMBEDDERS:
        raise ValueError(f"Unknown embedder '{name}'. Choices: {list(EMBEDDERS)}")
    return EMBEDDERS[name]()
