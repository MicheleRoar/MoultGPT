#!/usr/bin/env python3
# llm/retrieval/retrieve.py
"""
Query a VectorIndex built by build_index.py.

Usage
-----
    python retrieve.py --index index/demo_corpus --embedder tfidf \\
        --query "How many days does the moult cycle last?" --k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

RETRIEVAL_ROOT = Path(__file__).resolve().parent
LLM_ROOT = RETRIEVAL_ROOT.parent
for p in (str(RETRIEVAL_ROOT), str(LLM_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from embedder import get_embedder  # type: ignore  # noqa: E402
from index import Chunk, VectorIndex  # type: ignore  # noqa: E402


def retrieve_top_k(query: str, index: VectorIndex, embedder, k: int = 5) -> List[Tuple[Chunk, float]]:
    """Embed `query` with `embedder` and return the top-k (Chunk, similarity) pairs from `index`."""
    query_vector = embedder.encode([query])[0]
    return index.search(query_vector, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, help="Index path prefix (as passed to build_index.py --out)")
    parser.add_argument("--embedder", choices=["tfidf", "mistral-embed"], default="tfidf",
                         help="Must match the embedder the index was built with.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    index = VectorIndex.load(args.index)
    if index.embedder_name != args.embedder:
        print(f"[WARN] Index was built with '{index.embedder_name}', querying with '{args.embedder}' — "
              f"results will likely be meaningless (different vector spaces).")

    embedder = get_embedder(args.embedder)
    if args.embedder == "tfidf":
        # TfidfEmbedder has no persisted vocabulary (see index.py's module
        # docstring on why this isn't FAISS/a real vector DB) — re-fit
        # deterministically on the index's own chunk texts so the query is
        # embedded into the SAME vocabulary space the stored vectors live
        # in. A fresh TfidfVectorizer fit only on the one-sentence query
        # would share almost no vocabulary with the corpus at all.
        embedder.fit([c.text for c in index.chunks])

    results = retrieve_top_k(args.query, index, embedder, k=args.k)
    if not results:
        print("(empty index)")
        return
    for chunk, score in results:
        print(f"[{score:.3f}] ({chunk.paper_id}) {chunk.text}")


if __name__ == "__main__":
    main()
