#!/usr/bin/env python3
# llm/retrieval/build_index.py
"""
Build a VectorIndex from one or more papers (already-parsed TEI XML, or
plain text) — sentence-level chunks, embedded with the chosen backend.

Usage
-----
    # From already-parsed TEI XML (no GROBID needed — tei_to_text() just
    # reads the file):
    python build_index.py \\
        --tei ../finetuning/papers/2.tei.xml ../finetuning/papers/3.tei.xml \\
        --embedder tfidf --out index/demo_corpus

    # From raw text files, with the semantic (remote) embedder:
    python build_index.py --text some_paper.txt --embedder mistral-embed --out index/demo_corpus

Requires no new dependencies beyond what's already in llm/requirements.txt
(scikit-learn for --embedder tfidf; requests + MISTRAL_API_KEY env var for
--embedder mistral-embed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

RETRIEVAL_ROOT = Path(__file__).resolve().parent
LLM_ROOT = RETRIEVAL_ROOT.parent
for p in (str(RETRIEVAL_ROOT), str(LLM_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.parser import tei_to_text  # type: ignore  # noqa: E402
from pipeline.summarization import _simple_sentence_split  # type: ignore  # noqa: E402

from embedder import get_embedder  # type: ignore  # noqa: E402
from index import Chunk, VectorIndex  # type: ignore  # noqa: E402


def load_paper_text(path: Path) -> str:
    """TEI XML (GROBID output) or plain text -> a single text blob."""
    if path.name.endswith(".tei.xml") or path.suffix == ".xml":
        return tei_to_text(str(path))
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_paper(text: str, paper_id: str, min_chars: int = 40) -> List[Chunk]:
    """
    Split into sentence-level chunks — reuses the same sentence splitter
    pipeline/summarization.py's ontology-gated extraction uses, so the two
    context-selection strategies compared in evaluate_retrieval.py operate
    on the exact same candidate sentences, not two different tokenizations.
    """
    sentences = [s.strip() for s in _simple_sentence_split(text) if s.strip()]
    return [Chunk(text=s, paper_id=paper_id, meta={}) for s in sentences if len(s) >= min_chars]


def build_corpus_chunks(paper_paths: List[Path], min_chars: int = 40) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for p in paper_paths:
        text = load_paper_text(p)
        if not text.strip():
            print(f"[WARN] No text extracted from {p}, skipping.")
            continue
        chunks = chunk_paper(text, paper_id=p.stem.replace(".tei", ""), min_chars=min_chars)
        print(f"[INFO] {p.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tei", nargs="*", default=[], help="TEI XML file(s) (already-parsed GROBID output)")
    parser.add_argument("--text", nargs="*", default=[], help="Plain .txt file(s)")
    parser.add_argument("--embedder", choices=["tfidf", "mistral-embed"], default="tfidf")
    parser.add_argument("--out", required=True, help="Output path prefix (writes <out>.npz + <out>.json)")
    parser.add_argument("--min_chars", type=int, default=40,
                         help="Drop chunks shorter than this (strips headers, figure captions, stray fragments)")
    args = parser.parse_args()

    paper_paths = [Path(p) for p in args.tei] + [Path(p) for p in args.text]
    if not paper_paths:
        parser.error("Provide at least one --tei or --text file.")

    all_chunks = build_corpus_chunks(paper_paths, min_chars=args.min_chars)
    if not all_chunks:
        parser.error("No chunks extracted from any input file.")

    print(f"[INFO] Embedding {len(all_chunks)} chunks with '{args.embedder}'...")
    embedder = get_embedder(args.embedder)
    texts = [c.text for c in all_chunks]
    embedder.fit(texts)
    vectors = embedder.encode(texts)

    index = VectorIndex(embedder_name=args.embedder)
    index.build(all_chunks, vectors)
    index.save(args.out)
    print(f"[INFO] Saved index ({vectors.shape[0]} vectors, dim={vectors.shape[1]}) to {args.out}.npz/.json")


if __name__ == "__main__":
    main()
