#!/usr/bin/env python3
# llm/eval/trait_extraction/rejudge_unclassified.py
"""
DEPRECATED / INCOMPATIBLE with the current run_model_comparison.py.

This script predates the grouped-per-paper prompting rewrite and the
single/combo/negative item typing (see README.md's "Grouped (per-paper)
prompting" section) -- it assumes the older, non-typed results_scored.csv
shape. run_model_comparison.py's judge pass now uses `_call_with_retry`
directly and defaults to `--judge_model mistral-medium-latest` (avoiding
the self-rate-limiting problem this script worked around). Left here for
reference only -- do not run against current output.

Second selective follow-up to run_model_comparison.py, companion to
rerun_on_hard_items.py.

The real run (see results_scored.csv from 2026-07-27) hit two separate
rate-limit problems, not one:
  1. mistral-large-latest as an EXTRACTION model errored on 26/37 items
     (rerun_on_hard_items.py addresses this).
  2. mistral-large-latest as the JUDGE model *also* got rate-limited --
     of ~48 real disagreements across all three extraction models, only
     4 were actually judged (ambiguous/incorrect); the other ~31 have
     judge_verdict == "unclassified" (the literal string recorded when
     the judge call itself raised a 429 -- see run_model_comparison.py's
     judge-call except branch). Since the judge model and one of the
     extraction models were the same model hitting the same free-tier
     limit, most of the judge pass silently failed.

This script re-judges exactly those unclassified disagreements, using a
DIFFERENT judge model by default (mistral-medium-latest, which had zero
errors across its own 37 extraction calls in the same run -- i.e. it
wasn't anywhere near its own rate limit), with the same retry+backoff
pattern as rerun_on_hard_items.py for safety.

It does NOT re-run any extraction -- it only re-derives the evidence
text (by re-running the deterministic, temperature=0 domain pipeline
per paper, same as the original run) needed to build the judge prompt,
then re-judges.

Usage
-----
    cd llm
    python eval/trait_extraction/rejudge_unclassified.py \\
        --input_csv eval/trait_extraction/results/results_scored.csv \\
        --judge_model mistral-medium-latest \\
        --out_dir eval/trait_extraction/results
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

EVAL_ROOT = Path(__file__).resolve().parent
LLM_ROOT = EVAL_ROOT.parents[1]
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(LLM_ROOT / ".env")
except ImportError:
    pass

from pipeline.parser import tei_to_text  # type: ignore  # noqa: E402
from pipeline.domain_pipeline import run_domain_pipeline  # type: ignore  # noqa: E402
from domain.optimized_taxonomy_lookup import TaxonomyLookup  # type: ignore  # noqa: E402
from domain.moulting_ontology_gate import MoultingOntologyGate  # type: ignore  # noqa: E402
from backend.providers import generate as llm_generate, ProviderError  # type: ignore  # noqa: E402
from run_model_comparison import build_judge_prompt, parse_judge_verdict  # type: ignore  # noqa: E402

DEFAULT_PAPERS_DIR = LLM_ROOT / "finetuning" / "papers"
UNCLASSIFIED_VALUES = {None, "", "unclassified"}


def load_results_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def call_with_retry(system_prompt: str, user_content: str, model_id: str,
                     max_new_tokens: int, max_retries: int, base_delay: float) -> Dict[str, Any]:
    attempt = 0
    while True:
        try:
            return llm_generate(system_prompt=system_prompt, user_content=user_content,
                                 model_id=model_id, max_new_tokens=max_new_tokens)
        except ProviderError as e:
            is_rate_limit = "429" in str(e) or "rate_limited" in str(e).lower()
            if not is_rate_limit or attempt >= max_retries:
                raise
            wait = base_delay * (2 ** attempt)
            print(f"      [RATE LIMIT] retry {attempt + 1}/{max_retries} for {model_id} in {wait:.1f}s...")
            time.sleep(wait)
            attempt += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="mistral-medium-latest",
                         help="Deliberately NOT mistral-large-latest by default -- that model was "
                              "both an extraction model and the original judge, and its rate limit "
                              "is exactly what produced the unclassified rows this script fixes.")
    parser.add_argument("--taxonomy_csv", type=str, default=str(LLM_ROOT / "data" / "arthropod_taxonomy.csv"))
    parser.add_argument("--taxonomy_pickle", type=str, default=str(LLM_ROOT / "data" / "taxonomy_lookup.pkl"))
    parser.add_argument("--ontology_owl", type=str, default=str(LLM_ROOT / "data" / "moultdb_moulting_ontology_v3_8.owl"))
    parser.add_argument("--out_dir", type=str, default=str(EVAL_ROOT / "results"))
    parser.add_argument("--request_delay_sec", type=float, default=1.5)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--base_backoff_sec", type=float, default=5.0)
    parser.add_argument("--sample_review", type=int, default=10,
                         help="Print this many random verdicts to stdout for a human sanity-check.")
    args = parser.parse_args()

    all_rows = load_results_csv(Path(args.input_csv))
    to_rejudge = [r for r in all_rows if r["label"] == "disagreement"
                  and (r.get("judge_verdict") or "") in UNCLASSIFIED_VALUES]
    print(f"[INFO] {len(to_rejudge)} unclassified disagreements to re-judge with {args.judge_model}.")
    if not to_rejudge:
        print("[INFO] Nothing to do.")
        return

    by_paper: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in to_rejudge:
        by_paper[r["paper_id"]].append(r)

    print("[INFO] Loading taxonomy lookup + ontology gate...")
    taxonomy_lookup = TaxonomyLookup(csv_path=args.taxonomy_csv, pickle_path=args.taxonomy_pickle, rebuild=False)
    ontology_gate = MoultingOntologyGate(owl_path=args.ontology_owl)

    updates: Dict[str, tuple] = {}  # keyed by (item_id, model_id) -> (verdict, reason)
    for paper_id, rows in by_paper.items():
        print(f"\n=== Paper {paper_id} ({len(rows)} disagreements to re-judge) ===")
        tei_path = DEFAULT_PAPERS_DIR / f"{paper_id}.tei.xml"
        if not tei_path.exists():
            print(f"[WARN] Missing {tei_path}, skipping {len(rows)} rows.")
            continue
        full_text = tei_to_text(str(tei_path))
        pipeline_result = run_domain_pipeline(
            full_text=full_text, user_query="Extract moulting-related traits from this paper.",
            taxonomy_lookup=taxonomy_lookup, ontology_gate=ontology_gate,
        )
        evidence = pipeline_result["summary"]
        if not evidence.strip():
            print(f"[WARN] Empty evidence for paper {paper_id}, skipping.")
            continue

        for r in rows:
            gold_values = [v.strip() for v in r["gold_values"].split(";") if v.strip()]
            sys_p, user_p = build_judge_prompt(r["field"], evidence, gold_values, r["cleaned_prediction"])
            time.sleep(args.request_delay_sec)
            try:
                result = call_with_retry(sys_p, user_p, args.judge_model, 128,
                                          args.max_retries, args.base_backoff_sec)
                verdict, reason = parse_judge_verdict(result["text"])
                key = (r["item_id"], r["model_id"])
                updates[key] = (verdict, reason)
                print(f"    item {r['item_id']} / {r['model_id']} ({r['field']}): {verdict}")
            except ProviderError as e:
                print(f"    item {r['item_id']} / {r['model_id']}: judge ERROR {e}")

    # Merge updates back into all_rows and write a new CSV.
    for r in all_rows:
        key = (r["item_id"], r["model_id"])
        if key in updates:
            verdict, reason = updates[key]
            r["judge_verdict"] = verdict
            r["judge_reason"] = reason

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results_scored_rejudged.csv"
    fieldnames = list(all_rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    n_resolved = len(updates)
    n_ambiguous = sum(1 for v, _ in updates.values() if v == "ambiguous")
    n_incorrect = sum(1 for v, _ in updates.values() if v == "incorrect")
    print(f"\n[DONE] Re-judged {n_resolved}/{len(to_rejudge)} disagreements "
          f"({n_ambiguous} ambiguous, {n_incorrect} incorrect).")
    print(f"Wrote {out_csv}")

    if args.sample_review > 0 and updates:
        import random
        sample_keys = random.sample(list(updates.keys()), min(args.sample_review, len(updates)))
        print(f"\n[REVIEW] Random sample of {len(sample_keys)} new verdicts:")
        for r in all_rows:
            key = (r["item_id"], r["model_id"])
            if key in sample_keys:
                print(f"  paper {r['paper_id']} / {r['field']} / {r['model_id']}: "
                      f"gold={r['gold_values']!r} predicted={r['cleaned_prediction']!r} "
                      f"-> {r['judge_verdict']} ({r['judge_reason']})")


if __name__ == "__main__":
    main()
