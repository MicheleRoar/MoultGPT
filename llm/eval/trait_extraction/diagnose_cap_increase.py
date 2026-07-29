#!/usr/bin/env python3
# llm/eval/trait_extraction/diagnose_cap_increase.py
"""
Follow-up diagnostic (no API calls): for the "lost_in_selection" rows found
by diagnose_recall_loss.py (gold value present in full text, absent from
the selected evidence), check whether simply raising the num_sentences cap
would recover them, versus cases where the gold-supporting sentence never
even passes the ontology-score filter in the first place (a cap increase
would not help those; only a lower min_total_score or better ontology
surface matching would).

Optimization note: the ontology-score filtering pass (the expensive part,
owlready2 concept matching per sentence) is cap-independent, so it is run
ONCE per paper and cached; only the cheap TF-IDF/K-Means clustering step is
repeated per candidate cap.

Usage
-----
    cd llm
    python eval/trait_extraction/diagnose_cap_increase.py \
        --scored_csv /path/to/results_scored.csv --model mistral-large-latest
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

EVAL_ROOT = Path(__file__).resolve().parent
LLM_ROOT = EVAL_ROOT.parent.parent
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

from pipeline.parser import tei_to_text  # type: ignore  # noqa: E402
from pipeline.summarization import _simple_sentence_split, _ontology_sentence_signal  # type: ignore  # noqa: E402
from domain.moulting_ontology_gate import MoultingOntologyGate  # type: ignore  # noqa: E402

DEFAULT_PAPERS_DIR = LLM_ROOT / "finetuning" / "papers"
DEFAULT_OWL_PATH = LLM_ROOT / "data" / "moultdb_moulting_ontology_v3_8.owl"
MIN_TOTAL_SCORE = 2.5
CAPS_TO_TEST = [20, 30, 40, 60, 999]

_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize(s: str) -> str:
    return _NORM_RE.sub(" ", s.lower()).strip()


def gold_values_present(gold_values, haystack_norm: str) -> bool:
    for gv in gold_values:
        gv = gv.strip()
        if not gv or gv == "0":
            continue
        gv_norm = normalize(gv)
        if len(gv_norm) < 3:
            continue
        if gv_norm in haystack_norm:
            return True
    return False


def cluster_select(filtered: list[str], cap: int) -> str:
    """Same clustering logic as extract_relevant_sentences, but operating on
    an already-ontology-filtered sentence list (cap-independent part done
    once by the caller)."""
    if not filtered:
        return ""
    if len(filtered) <= cap:
        return "\n".join(filtered)
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(filtered)
    k = min(cap, len(filtered))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    chosen = []
    for i in range(k):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if not cluster_indices.size:
            continue
        center = kmeans.cluster_centers_[i]
        scores = X[cluster_indices] @ center.T
        closest_idx = cluster_indices[np.argmax(scores)]
        chosen.append(filtered[closest_idx])
    return "\n".join(chosen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_csv", type=str, required=True)
    ap.add_argument("--model", type=str, default="mistral-large-latest")
    ap.add_argument("--papers_dir", type=str, default=str(DEFAULT_PAPERS_DIR))
    ap.add_argument("--ontology_owl", type=str, default=str(DEFAULT_OWL_PATH))
    args = ap.parse_args()

    print("[INFO] Loading ontology gate...")
    gate = MoultingOntologyGate(owl_path=args.ontology_owl)

    rows = []
    with open(args.scored_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("condition") != "main" or r.get("model_id") != args.model:
                continue
            if r.get("item_type") not in ("single", "combo"):
                continue
            if r.get("label") not in ("abstained", "disagreement"):
                continue
            rows.append(r)

    needed_papers = sorted({r["paper_id"] for r in rows}, key=lambda x: int(x))
    print(f"[INFO] {len(rows)} candidate rows across {len(needed_papers)} papers. Running ontology filter once per paper...")

    filtered_cache: dict[str, list[str]] = {}
    full_text_cache: dict[str, str] = {}
    for pid in needed_papers:
        tei_path = Path(args.papers_dir) / f"{pid}.tei.xml"
        full_text = tei_to_text(str(tei_path)) if tei_path.exists() else ""
        full_text_cache[pid] = full_text
        if not full_text:
            filtered_cache[pid] = []
            continue
        sentences = [s.strip() for s in _simple_sentence_split(full_text) if s.strip()]
        filtered = []
        for sent in sentences:
            signal = _ontology_sentence_signal(sentence=sent, ontology_gate=gate, min_total_score=MIN_TOTAL_SCORE)
            if signal["allow"]:
                filtered.append(sent)
        filtered_cache[pid] = filtered
        print(f"  paper {pid}: {len(sentences)} sentences -> {len(filtered)} pass ontology filter")

    # evidence per (paper, cap), cheap now (reuses filtered_cache)
    evidence_cache: dict[tuple, str] = {}

    def get_evidence(paper_id: str, cap: int) -> str:
        key = (paper_id, cap)
        if key not in evidence_cache:
            evidence_cache[key] = cluster_select(filtered_cache.get(paper_id, []), cap)
        return evidence_cache[key]

    lost_rows = []
    for r in rows:
        full_text = full_text_cache.get(r["paper_id"], "")
        if not full_text:
            continue
        gold_values = [v.strip() for v in r["gold_values"].split(";") if v.strip()]
        if not gold_values:
            continue
        if not gold_values_present(gold_values, normalize(full_text)):
            continue
        ev20 = get_evidence(r["paper_id"], 20)
        if gold_values_present(gold_values, normalize(ev20)):
            continue
        lost_rows.append((r, gold_values))

    print(f"\n[INFO] {len(lost_rows)} 'lost_in_selection' rows to re-test across caps {CAPS_TO_TEST}.")

    recovered_at_cap = Counter()
    never_recovered = []
    never_passes_ontology_filter = []

    for r, gold_values in lost_rows:
        paper_id = r["paper_id"]
        recovered = False
        for cap in CAPS_TO_TEST:
            ev = get_evidence(paper_id, cap)
            if gold_values_present(gold_values, normalize(ev)):
                recovered_at_cap[cap] += 1
                recovered = True
                break
        if not recovered:
            never_recovered.append(r)
            any_pass = any(
                gold_values_present(gold_values, normalize(sent))
                for sent in filtered_cache.get(paper_id, [])
            )
            # filtered_cache already only contains ontology-passing sentences,
            # so "any_pass" here really means "a passing sentence contains the gold value"
            if not any_pass:
                never_passes_ontology_filter.append(r)

    print(f"\n[RESULTS] Of {len(lost_rows)} rows lost at cap=20:")
    cum = 0
    for cap in CAPS_TO_TEST:
        cum += recovered_at_cap[cap]
        print(f"  recovered by raising cap to {cap}: +{recovered_at_cap[cap]} (cumulative {cum}/{len(lost_rows)})")
    print(f"  never recovered even with cap=999 (~no cap): {len(never_recovered)}/{len(lost_rows)}")
    print(f"    of those, no ontology-filtered sentence contains the gold value at all "
          f"(min_total_score={MIN_TOTAL_SCORE} is the real blocker, not the cap): {len(never_passes_ontology_filter)}/{max(len(never_recovered),1)}")

    if never_passes_ontology_filter:
        print("\n  Examples where the ontology filter itself is the blocker (cap increase can't help):")
        for r in never_passes_ontology_filter[:6]:
            print(f"    - paper {r['paper_id']}, field \"{r['field']}\": gold={r['gold_values']}")


if __name__ == "__main__":
    main()
