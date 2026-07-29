# llm/retrieval/index.py
"""
A minimal vector index: cosine similarity over an in-memory numpy array,
persisted as a .npz (vectors) + .json (metadata + chunk text) pair.

This is deliberately not FAISS or a real vector database — the corpus
sizes this module is meant for (sentences from a handful of papers, not
millions of documents) don't need one, and skipping it keeps this module
dependency-free (numpy is already required by llm/requirements.txt). If
MoultGPT's retrieval corpus ever grows past a few thousand chunks,
swapping VectorIndex.search()'s brute-force dot product for FAISS/pgvector
/etc. is a contained change — embedder.py and the CLIs are unaffected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class Chunk:
    text: str
    paper_id: str
    meta: Dict[str, Any] = field(default_factory=dict)


class VectorIndex:
    def __init__(self, embedder_name: str):
        self.embedder_name = embedder_name
        self.vectors: np.ndarray = np.zeros((0, 0), dtype="float32")
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"chunks/vectors length mismatch: {len(chunks)} vs {vectors.shape[0]}")
        # L2-normalize once at build time so search() is a plain dot product
        # instead of recomputing norms on every query.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = vectors / norms
        self.chunks = chunks

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Chunk, float]]:
        """Return the top-k (chunk, cosine_similarity) pairs for query_vector."""
        if self.vectors.shape[0] == 0:
            return []
        q = np.asarray(query_vector, dtype="float32").reshape(-1)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        scores = self.vectors @ q
        k = min(k, len(self.chunks))
        top_idx = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p.with_suffix(".npz"), vectors=self.vectors)
        meta = {
            "embedder_name": self.embedder_name,
            "chunks": [{"text": c.text, "paper_id": c.paper_id, "meta": c.meta} for c in self.chunks],
        }
        p.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        p = Path(path)
        data = np.load(str(p.with_suffix(".npz")))
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        idx = cls(embedder_name=meta["embedder_name"])
        idx.vectors = data["vectors"]
        idx.chunks = [Chunk(text=c["text"], paper_id=c["paper_id"], meta=c.get("meta", {})) for c in meta["chunks"]]
        return idx
