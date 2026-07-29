#!/usr/bin/env python3
# llm/retrieval/evaluate_retrieval.py
"""
Compares three ways of picking "the sentences worth showing the LLM" on
MoultGPT's own paper corpus (llm/finetuning/papers/*.tei.xml — real
extracted text, no GROBID service or network access needed, via
pipeline.parser.tei_to_text):

  1. ontology   — the method already in production (pipeline/summarization.py):
                  score every sentence against the moulting OWL ontology,
                  keep the ones that pass a threshold, cluster with TF-IDF +
                  KMeans down to a fixed-size summary. Query-agnostic: the
                  same summary is produced regardless of what's being asked.
  2. tfidf       — this module's sparse retrieval (embedder.py's
                  TfidfEmbedder): embed every sentence once, then for each
                  query return the top-k by cosine similarity. Query-aware,
                  local, no API key.
  3. mistral-embed — this module's dense retrieval (MistralRemoteEmbedder).
                  Only runs if MISTRAL_API_KEY is set; reported as
                  "not run" otherwise rather than silently skipped, so the
                  results file always shows what WAS and WASN'T measured
                  (same "don't fake a green run" principle as
                  .github/workflows/ci.yml).

Metric: for a handful of hand-written queries with a small set of expected
keywords each (chosen by actually reading the source papers — see the
QUERIES list below, and the paper excerpts in this file's git history /
the eval_results.md this script produces), a method "hits" a query if ANY
sentence it selected contains every keyword in that query's set. This is a
cheap, explainable proxy for relevance, not a real gold-labelled relevance
judgment — n=5 queries over 3 papers is nowhere near enough to draw a
general conclusion from, and that limitation is stated in the output file,
not hidden. What this script IS good for: a concrete, reproducible,
actually-executed comparison instead of just describing what retrieval
"would" do.

Usage
-----
    python evaluate_retrieval.py --out eval_results.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

RETRIEVAL_ROOT = Path(__file__).resolve().parent
LLM_ROOT = RETRIEVAL_ROOT.parent
for p in (str(RETRIEVAL_ROOT), str(LLM_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.parser import tei_to_text  # type: ignore  # noqa: E402
from pipeline.summarization import extract_relevant_sentences  # type: ignore  # noqa: E402
from domain.moulting_ontology_gate import MoultingOntologyGate  # type: ignore  # noqa: E402

from build_index import build_corpus_chunks  # type: ignore  # noqa: E402
from embedder import get_embedder, ProviderError  # type: ignore  # noqa: E402
from index import VectorIndex  # type: ignore  # noqa: E402
from retrieve import retrieve_top_k  # type: ignore  # noqa: E402

DEFAULT_PAPERS = [
    LLM_ROOT / "finetuning" / "papers" / "2.tei.xml",  # crustacean growth/molt/reproduction trade-offs
    LLM_ROOT / "finetuning" / "papers" / "3.tei.xml",  # Porcellio scaber (isopod) moult cycle substages
    LLM_ROOT / "finetuning" / "papers" / "4.tei.xml",  # Armadillo officinalis (isopod) moult cycle + exuviae
]
DEFAULT_OWL_PATH = LLM_ROOT / "data" / "moultdb_moulting_ontology_v3_8.owl"

# Each query's keyword set was chosen by reading the actual extracted text
# of the three papers above (all three are moult-cycle-focused isopod/
# crustacean physiology papers) — not guessed abstractly. "hit" = at least
# one selected/retrieved sentence contains every keyword (case-insensitive
# substring match).
QUERIES: List[Tuple[str, List[str]]] = [
    ("How many days does the moult cycle last?", ["duration", "day"]),
    ("What are the substages of the premoult stage?", ["premoult", "substage"]),
    ("What happens to the exuviae after moulting?", ["exuvi"]),
    ("How does molting frequency relate to reproduction?", ["molt", "reproduct"]),
    ("What is the growth increment at ecdysis?", ["ecdysis", "growth"]),
]

TOP_K = 5
ONTOLOGY_SENTENCES_PER_PAPER = 10  # -> pool of up to 30 sentences across the 3 papers, comparable in scale to top_k=5 x 5 queries


def keyword_hit(sentence: str, keywords: List[str]) -> bool:
    s = sentence.lower()
    return all(kw.lower() in s for kw in keywords)


def find_hit(sentences: List[str], keywords: List[str]) -> str | None:
    for s in sentences:
        if keyword_hit(s, keywords):
            return s
    return None


def run_ontology_method(paper_paths: List[Path]) -> List[str]:
    """Query-agnostic: one fixed pool of sentences, built once from all papers."""
    gate = MoultingOntologyGate(owl_path=str(DEFAULT_OWL_PATH))
    pool: List[str] = []
    for p in paper_paths:
        text = tei_to_text(str(p))
        summary = extract_relevant_sentences(text, gate, num_sentences=ONTOLOGY_SENTENCES_PER_PAPER)
        pool.extend(s for s in summary.split("\n") if s.strip())
    return pool


def run_retrieval_method(paper_paths: List[Path], embedder_name: str) -> Dict[str, List[str]]:
    """Query-aware: build one index over all papers, retrieve top-k per query."""
    chunks = build_corpus_chunks(paper_paths)
    embedder = get_embedder(embedder_name)
    texts = [c.text for c in chunks]
    embedder.fit(texts)
    vectors = embedder.encode(texts)

    index = VectorIndex(embedder_name=embedder_name)
    index.build(chunks, vectors)

    results: Dict[str, List[str]] = {}
    for query, _keywords in QUERIES:
        top = retrieve_top_k(query, index, embedder, k=TOP_K)
        results[query] = [chunk.text for chunk, _score in top]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(RETRIEVAL_ROOT / "eval_results.md"))
    args = parser.parse_args()

    paper_paths = DEFAULT_PAPERS
    for p in paper_paths:
        if not p.exists():
            parser.error(f"Missing expected corpus file: {p}")

    lines: List[str] = []
    lines.append("# Retrieval evaluation — ontology-gated extraction vs. embedding retrieval\n")
    lines.append(
        "Generated by `evaluate_retrieval.py`. Corpus: `llm/finetuning/papers/{2,3,4}.tei.xml` "
        "(real extracted paper text, via `pipeline.parser.tei_to_text` — no GROBID service or "
        "network access needed to reproduce this). See that script's module docstring for the "
        "full methodology and its stated limitations before reading the numbers below as more "
        "than they are (n=5 queries, keyword-hit is a proxy for relevance, not gold-labelled "
        "human judgment).\n"
    )

    method_results: Dict[str, Dict] = {}

    print("[INFO] Running ontology-gated extraction (query-agnostic pool)...")
    t0 = time.time()
    ontology_pool = run_ontology_method(paper_paths)
    ontology_time = time.time() - t0
    print(f"[INFO] ontology: {len(ontology_pool)} pooled sentences in {ontology_time:.2f}s")

    print("[INFO] Running tfidf retrieval (query-aware)...")
    t0 = time.time()
    tfidf_results = run_retrieval_method(paper_paths, "tfidf")
    tfidf_time = time.time() - t0
    print(f"[INFO] tfidf: retrieved for {len(tfidf_results)} queries in {tfidf_time:.2f}s")

    mistral_results = None
    mistral_error = None
    try:
        print("[INFO] Attempting mistral-embed retrieval (requires MISTRAL_API_KEY)...")
        mistral_results = run_retrieval_method(paper_paths, "mistral-embed")
    except ProviderError as e:
        mistral_error = str(e)
        print(f"[INFO] Skipped mistral-embed: {e}")

    # ── Score each method against QUERIES ──
    def score(sentences_by_query) -> Tuple[int, Dict[str, str | None]]:
        hits = 0
        detail = {}
        for query, keywords in QUERIES:
            candidates = sentences_by_query if isinstance(sentences_by_query, list) else sentences_by_query[query]
            hit_sentence = find_hit(candidates, keywords)
            if hit_sentence:
                hits += 1
            detail[query] = hit_sentence
        return hits, detail

    ontology_hits, ontology_detail = score(ontology_pool)
    tfidf_hits, tfidf_detail = score(tfidf_results)
    method_results["ontology (production, query-agnostic)"] = {
        "hits": ontology_hits, "n": len(QUERIES), "detail": ontology_detail,
        "pool_size": len(ontology_pool), "time_sec": ontology_time,
    }
    method_results["tfidf (this module, query-aware, top-5)"] = {
        "hits": tfidf_hits, "n": len(QUERIES), "detail": tfidf_detail,
        "pool_size": f"{TOP_K}/query", "time_sec": tfidf_time,
    }

    if mistral_results is not None:
        mistral_hits, mistral_detail = score(mistral_results)
        method_results["mistral-embed (this module, query-aware, top-5)"] = {
            "hits": mistral_hits, "n": len(QUERIES), "detail": mistral_detail,
            "pool_size": f"{TOP_K}/query", "time_sec": None,
        }

    # ── Write results table ──
    lines.append("## Results\n")
    lines.append("| Method | Hit rate | Pool size | Wall time |")
    lines.append("|---|---|---|---|")
    for name, r in method_results.items():
        wt = f"{r['time_sec']:.2f}s" if r["time_sec"] is not None else "–"
        lines.append(f"| {name} | {r['hits']}/{r['n']} | {r['pool_size']} | {wt} |")
    if mistral_results is None:
        lines.append(f"| mistral-embed (this module, query-aware, top-5) | **not run** | – | – |")
        lines.append(f"\n> mistral-embed was not run: {mistral_error}\n")

    lines.append("\n## Per-query detail\n")
    for query, keywords in QUERIES:
        lines.append(f"### \"{query}\"")
        lines.append(f"Expected keywords: {keywords}\n")
        for name, r in method_results.items():
            hit_sentence = r["detail"].get(query)
            status = "✅ hit" if hit_sentence else "❌ miss"
            lines.append(f"- **{name}** — {status}")
            if hit_sentence:
                lines.append(f"  > {hit_sentence}")
        lines.append("")

    lines.append("## Discussion\n")
    lines.append(
        "- **ontology vs. tfidf isn't a fair apples-to-apples comparison by "
        "construction**: the ontology method builds one fixed, query-agnostic "
        "pool per paper (it's a domain gate + summarizer, not a retriever), "
        "while tfidf/mistral-embed re-rank the *entire* corpus fresh for every "
        "query. A query-agnostic method structurally cannot adapt to a specific "
        "question the way a retriever can — that's the actual argument for "
        "adding retrieval alongside the existing gate, not a knock against the "
        "gate (which is doing a different job: deciding whether a paper is "
        "in-scope at all, not finding the best sentence for a specific question)."
    )
    lines.append(
        "- **n=5 keyword-matched queries over 3 papers is a smoke test, not a "
        "benchmark.** It's enough to catch a broken retriever (e.g. wrong "
        "vocabulary space, off-by-one in scoring) and to produce real, "
        "reproducible numbers instead of describing untested code — it is "
        "not enough to claim one method is generally better than another."
    )
    lines.append(
        "- **mistral-embed is implemented but its actual retrieval quality is "
        "unverified in this repo** (no MISTRAL_API_KEY in the environment "
        "this was generated in) — the honest status is \"code exists and is "
        "unit-testable, results not yet measured,\" not \"working.\""
    )

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[INFO] Wrote results to {out_path}")


if __name__ == "__main__":
    main()
