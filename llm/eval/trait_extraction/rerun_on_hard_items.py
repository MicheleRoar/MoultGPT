#!/usr/bin/env python3
# llm/eval/trait_extraction/rerun_on_hard_items.py
"""
DEPRECATED / INCOMPATIBLE with the current run_model_comparison.py.

This script predates the grouped-per-paper prompting rewrite and the
single/combo/negative item typing (see README.md's "Grouped (per-paper)
prompting" section). It assumes a results_scored.csv without
`item_type`/`combo_split_ok` columns and calls pipeline.prompting's
single_trait mode directly, which the current harness no longer uses.
run_model_comparison.py now has its own retry+backoff
(`_call_with_retry`) built in, which is what this script's role has been
folded into. Left here for reference only -- do not run against current
output.

Selective follow-up to run_model_comparison.py: instead of running a
larger/more expensive model on all ~50 gold items (which is what burned
through mistral-large-latest's free-tier rate limit in practice --
consecutive 429s partway through a full run), this script reads a
PREVIOUS run's results_scored.csv, finds the items where the base models
(e.g. mistral-small-latest, mistral-medium-latest) did NOT produce a
"correct" answer, and runs a single stronger model on ONLY those items.

This answers a real, useful question for the paper: does a bigger model
actually resolve the cases the smaller ones get wrong, or do they fail
for reasons (missing evidence, genuinely ambiguous text) that model size
doesn't fix? It also happens to make far fewer requests than a full
re-run, which is the practical reason it's more likely to complete
without tripping the free-tier rate limit -- helped further here by
retry-with-backoff on 429s and a fixed delay between requests (neither of
which run_model_comparison.py's main condition loop has, since that
script's main condition intentionally stays simple/fast for the bulk of
the comparison).

Usage
-----
    cd llm
    python eval/trait_extraction/rerun_on_hard_items.py \\
        --input_csv eval/trait_extraction/results/results_scored.csv \\
        --base_models mistral-small-latest mistral-medium-latest \\
        --rerun_model mistral-large-latest \\
        --out_dir eval/trait_extraction/results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

EVAL_ROOT = Path(__file__).resolve().parent
LLM_ROOT = EVAL_ROOT.parents[1]
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

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
from pipeline.prompting import MODE_SINGLE_TRAIT, build_system_prompt, build_user_content  # type: ignore  # noqa: E402
from scoring import classify_prediction  # type: ignore  # noqa: E402

DEFAULT_GOLD_QUESTIONS = EVAL_ROOT / "gold_questions.json"
DEFAULT_PAPERS_DIR = LLM_ROOT / "finetuning" / "papers"
SINGLE_TRAIT_MAX_TOKENS = 256


def load_results_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_hard_item_ids(rows: List[Dict[str, Any]], base_models: List[str]) -> List[int]:
    """
    An item is "hard" if, among its `condition == "main"` rows for the
    given base_models, NONE has label == "correct" -- i.e. every base
    model either disagreed with gold or abstained. If a base model's row
    is simply missing (e.g. it errored/gated-out in the original run),
    that counts as "not correct" too -- we don't want to silently treat a
    missing row as if it were a success.
    """
    by_item: Dict[str, Dict[str, str]] = defaultdict(dict)
    for row in rows:
        if row.get("condition") != "main":
            continue
        by_item[row["item_id"]][row["model_id"]] = row["label"]

    hard_ids = []
    for item_id, labels_by_model in by_item.items():
        labels = [labels_by_model.get(m) for m in base_models]
        if not any(label == "correct" for label in labels):
            hard_ids.append(int(item_id))
    return sorted(hard_ids)


def call_with_retry(system_prompt: str, user_content: str, model_id: str,
                     max_new_tokens: int, max_retries: int, base_delay: float) -> Dict[str, Any]:
    """
    Wraps backend.providers.generate with retry-with-exponential-backoff
    specifically for 429 rate-limit errors (Mistral's error message
    format: "Mistral error 429: ..."). Other ProviderErrors (bad model
    id, auth failure, malformed response) are NOT retried -- retrying
    those would just burn time reproducing the same failure.
    """
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
    parser.add_argument("--input_csv", type=str, required=True,
                         help="results_scored.csv from a previous run_model_comparison.py run.")
    parser.add_argument("--base_models", type=str, nargs="+",
                         default=["mistral-small-latest", "mistral-medium-latest"])
    parser.add_argument("--rerun_model", type=str, default="mistral-large-latest")
    parser.add_argument("--gold_questions", type=str, default=str(DEFAULT_GOLD_QUESTIONS))
    parser.add_argument("--taxonomy_csv", type=str, default=str(LLM_ROOT / "data" / "arthropod_taxonomy.csv"))
    parser.add_argument("--taxonomy_pickle", type=str, default=str(LLM_ROOT / "data" / "taxonomy_lookup.pkl"))
    parser.add_argument("--ontology_owl", type=str, default=str(LLM_ROOT / "data" / "moultdb_moulting_ontology_v3_8.owl"))
    parser.add_argument("--out_dir", type=str, default=str(EVAL_ROOT / "results"))
    parser.add_argument("--request_delay_sec", type=float, default=2.0,
                         help="Fixed pause between requests, proactively -- cheaper than hitting the rate limit and retrying.")
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--base_backoff_sec", type=float, default=5.0)
    args = parser.parse_args()

    previous_rows = load_results_csv(Path(args.input_csv))
    hard_item_ids = set(find_hard_item_ids(previous_rows, args.base_models))
    print(f"[INFO] {len(hard_item_ids)} hard items found (none of {args.base_models} scored 'correct'): "
          f"{sorted(hard_item_ids)}")
    if not hard_item_ids:
        print("[INFO] No hard items -- nothing to rerun.")
        return

    gold = json.loads(Path(args.gold_questions).read_text(encoding="utf-8"))
    items = [it for it in gold["items"] if it["item_id"] in hard_item_ids]
    items_by_paper: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        items_by_paper[it["paper_id"]].append(it)

    print("[INFO] Loading taxonomy lookup + ontology gate...")
    taxonomy_lookup = TaxonomyLookup(csv_path=args.taxonomy_csv, pickle_path=args.taxonomy_pickle, rebuild=False)
    ontology_gate = MoultingOntologyGate(owl_path=args.ontology_owl)
    single_trait_system_prompt = build_system_prompt(MODE_SINGLE_TRAIT)

    rows: List[Dict[str, Any]] = []
    for paper_id, paper_items in items_by_paper.items():
        print(f"\n=== Paper {paper_id} ({len(paper_items)} hard items) ===")
        tei_path = DEFAULT_PAPERS_DIR / f"{paper_id}.tei.xml"
        if not tei_path.exists():
            print(f"[WARN] Missing {tei_path}, skipping.")
            continue
        full_text = tei_to_text(str(tei_path))
        pipeline_result = run_domain_pipeline(
            full_text=full_text, user_query="Extract moulting-related traits from this paper.",
            taxonomy_lookup=taxonomy_lookup, ontology_gate=ontology_gate,
        )
        summary = pipeline_result["summary"]
        if not pipeline_result["decision"]["allow"] or not summary.strip():
            print(f"[WARN] Paper {paper_id} gated out -- recording as gated_out.")
            for it in paper_items:
                rows.append({"item_id": it["item_id"], "paper_id": paper_id, "field": it["field"],
                             "gold_values": "; ".join(it["gold_values"]), "model_id": args.rerun_model,
                             "condition": "hard_item_rerun", "raw_answer": None, "label": "gated_out",
                             "cleaned_prediction": None})
            continue

        for it in paper_items:
            user_content = build_user_content(summary, it["question"], MODE_SINGLE_TRAIT)
            time.sleep(args.request_delay_sec)
            try:
                result = call_with_retry(single_trait_system_prompt, user_content, args.rerun_model,
                                          SINGLE_TRAIT_MAX_TOKENS, args.max_retries, args.base_backoff_sec)
                label, cleaned = classify_prediction(result["text"], it["gold_values"])
                rows.append({"item_id": it["item_id"], "paper_id": paper_id, "field": it["field"],
                             "gold_values": "; ".join(it["gold_values"]), "model_id": args.rerun_model,
                             "condition": "hard_item_rerun", "raw_answer": result["text"],
                             "label": label, "cleaned_prediction": cleaned})
                print(f"    item {it['item_id']} ({it['field']}): {label} ({cleaned!r})")
            except ProviderError as e:
                rows.append({"item_id": it["item_id"], "paper_id": paper_id, "field": it["field"],
                             "gold_values": "; ".join(it["gold_values"]), "model_id": args.rerun_model,
                             "condition": "hard_item_rerun", "raw_answer": None, "label": "error",
                             "cleaned_prediction": None})
                print(f"    item {it['item_id']} ({it['field']}): ERROR {e}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results_hard_items_rerun.csv"
    fieldnames = ["item_id", "paper_id", "field", "gold_values", "model_id", "condition",
                  "raw_answer", "label", "cleaned_prediction"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_resolved = sum(1 for r in rows if r["label"] == "correct")
    n_total = len(rows)
    report_path = out_dir / "hard_items_rerun_report.md"
    report_lines = [
        "# Hard-item rerun — does a bigger model resolve what smaller models missed?\n",
        f"{n_total} items where none of {args.base_models} scored 'correct', rerun with "
        f"`{args.rerun_model}` alone (single_trait mode, same evidence context).\n",
        f"**{args.rerun_model} resolved {n_resolved}/{n_total} ({100*n_resolved/n_total:.1f}%) "
        f"of the hard items** that {' and '.join(args.base_models)} both missed.\n",
        "| item | paper | field | gold | rerun label | rerun prediction |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        report_lines.append(f"| {r['item_id']} | {r['paper_id']} | {r['field']} | {r['gold_values']} | "
                            f"{r['label']} | {r['cleaned_prediction']} |")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"\n[DONE] {n_resolved}/{n_total} hard items resolved by {args.rerun_model}.")
    print(f"Wrote {csv_path}, {report_path}")


if __name__ == "__main__":
    main()
