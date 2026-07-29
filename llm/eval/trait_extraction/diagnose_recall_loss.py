#!/usr/bin/env python3
# llm/eval/trait_extraction/diagnose_recall_loss.py
"""
Diagnostic (no API calls needed): for every non-correct main-condition item
in an existing results_scored.csv, check whether the gold value appears
verbatim (case-insensitive substring) in the paper's FULL text but NOT in
the sentence-selection evidence the model was actually shown.

This directly tests one of the three hypotheses raised in the paper's
Discussion for the low trait-extraction accuracy: that ontology-based
sentence selection may be discarding evidence the model would otherwise
use (suggested by the ungated-full-text baseline slightly outperforming
the gated pipeline on raw correct count).

Re-derives each paper's selected evidence deterministically via
pipeline/summarization.py (same production code, no LLM call, no network
needed) rather than re-running the whole harness.

Usage
-----
    cd llm
    python eval/trait_extraction/diagnose_recall_loss.py \
        --scored_csv /path/to/results_scored.csv \
        --model mistral-large-latest \
        --out eval/trait_extraction/results/recall_loss_diagnostic.md
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from collections import Counter

EVAL_ROOT = Path(__file__).resolve().parent
LLM_ROOT = EVAL_ROOT.parent.parent
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

from pipeline.parser import tei_to_text  # type: ignore  # noqa: E402
from pipeline.summarization import extract_relevant_sentences  # type: ignore  # noqa: E402
from domain.moulting_ontology_gate import MoultingOntologyGate  # type: ignore  # noqa: E402

DEFAULT_PAPERS_DIR = LLM_ROOT / "finetuning" / "papers"
DEFAULT_OWL_PATH = LLM_ROOT / "data" / "moultdb_moulting_ontology_v3_8.owl"
NUM_SUMMARY_SENTENCES = 20
MIN_TOTAL_SCORE = 2.5

_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize(s: str) -> str:
    return _NORM_RE.sub(" ", s.lower()).strip()


def gold_values_present(gold_values: list[str], haystack_norm: str) -> bool:
    """True if ANY of the (possibly multi-value, ';'-separated) gold values
    appears as a normalized substring in haystack. Deliberately loose (this
    is a diagnostic, not the paper's scorer) -- a substring hit is a lower
    bound on "the fact was available", not a claim the model would have
    gotten the exact right phrasing."""
    for gv in gold_values:
        gv = gv.strip()
        if not gv or gv in ("0",):  # "0"/empty are structurally ambiguous, skip
            continue
        gv_norm = normalize(gv)
        if len(gv_norm) < 3:
            continue
        if gv_norm in haystack_norm:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored_csv", type=str, required=True)
    ap.add_argument("--model", type=str, default="mistral-large-latest")
    ap.add_argument("--papers_dir", type=str, default=str(DEFAULT_PAPERS_DIR))
    ap.add_argument("--ontology_owl", type=str, default=str(DEFAULT_OWL_PATH))
    ap.add_argument("--out", type=str, default=str(EVAL_ROOT / "results" / "recall_loss_diagnostic.md"))
    args = ap.parse_args()

    print(f"[INFO] Loading ontology gate...")
    gate = MoultingOntologyGate(owl_path=args.ontology_owl)

    print(f"[INFO] Reading {args.scored_csv} for model={args.model}, condition=main...")
    rows = []
    with open(args.scored_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("condition") != "main":
                continue
            if r.get("model_id") != args.model:
                continue
            if r.get("item_type") not in ("single", "combo"):
                continue
            if r.get("label") not in ("abstained", "disagreement"):
                continue
            rows.append(r)
    print(f"[INFO] {len(rows)} non-correct main-condition rows for {args.model} (single+combo, abstained/disagreement).")

    papers_dir = Path(args.papers_dir)
    full_text_cache: dict[str, str] = {}
    evidence_cache: dict[str, str] = {}

    def get_texts(paper_id: str):
        if paper_id not in full_text_cache:
            tei_path = papers_dir / f"{paper_id}.tei.xml"
            if not tei_path.exists():
                full_text_cache[paper_id] = ""
                evidence_cache[paper_id] = ""
                return "", ""
            full_text = tei_to_text(str(tei_path))
            full_text_cache[paper_id] = full_text
            evidence_cache[paper_id] = extract_relevant_sentences(
                full_text=full_text, ontology_gate=gate,
                num_sentences=NUM_SUMMARY_SENTENCES, min_total_score=MIN_TOTAL_SCORE,
            )
        return full_text_cache[paper_id], evidence_cache[paper_id]

    categories = Counter()
    examples = {"lost_in_selection": [], "not_in_fulltext": [], "in_evidence_still_wrong": []}

    for r in rows:
        paper_id = r["paper_id"]
        full_text, evidence = get_texts(paper_id)
        if not full_text:
            categories["no_paper_text"] += 1
            continue
        gold_values = [v.strip() for v in r["gold_values"].split(";") if v.strip()]
        if not gold_values:
            categories["no_gold_value_to_check"] += 1
            continue

        full_norm = normalize(full_text)
        evidence_norm = normalize(evidence)

        in_full = gold_values_present(gold_values, full_norm)
        in_evidence = gold_values_present(gold_values, evidence_norm)

        if not in_full:
            categories["not_in_fulltext"] += 1
            if len(examples["not_in_fulltext"]) < 8:
                examples["not_in_fulltext"].append(r)
        elif in_full and not in_evidence:
            categories["lost_in_selection"] += 1
            if len(examples["lost_in_selection"]) < 15:
                examples["lost_in_selection"].append(r)
        else:  # in_full and in_evidence
            categories["in_evidence_still_wrong"] += 1
            if len(examples["in_evidence_still_wrong"]) < 8:
                examples["in_evidence_still_wrong"].append(r)

    total = sum(categories[k] for k in ("not_in_fulltext", "lost_in_selection", "in_evidence_still_wrong"))
    print(f"\n[RESULTS] model={args.model}, {total} checkable rows")
    for k in ("not_in_fulltext", "lost_in_selection", "in_evidence_still_wrong", "no_paper_text", "no_gold_value_to_check"):
        print(f"  {k}: {categories[k]}")

    lines = []
    lines.append(f"# Recall-loss diagnostic: is sentence-selection discarding evidence the model would use?\n")
    lines.append(
        f"For every main-condition item where `{args.model}` did NOT get it right (label `abstained` or "
        f"`disagreement`, item_type single/combo), this checks whether each gold value appears (normalized "
        f"substring match, a looser check than the paper's real scorer) in the paper's full text versus in "
        f"the ~13-20 sentences actually selected and shown to the model. Source: `{Path(args.scored_csv).name}`.\n"
    )
    lines.append(f"Rows checked: {total} (of {len(rows)} non-correct rows; "
                 f"{categories['no_paper_text']} skipped for missing paper text, "
                 f"{categories['no_gold_value_to_check']} skipped for empty/degenerate gold value).\n")
    lines.append("| Category | n | % of checked | Interpretation |")
    lines.append("|---|---|---|---|")
    pct = lambda n: f"{100.0*n/total:.1f}%" if total else "n/a"
    lines.append(f"| Gold value NOT in full text (verbatim) | {categories['not_in_fulltext']} | {pct(categories['not_in_fulltext'])} | Expected for inferred/paraphrased MoultDB annotations (e.g. \"direct development\"); not a selection bug. |")
    lines.append(f"| Gold value in full text, NOT in selected evidence | {categories['lost_in_selection']} | {pct(categories['lost_in_selection'])} | **Sentence-selection dropped it before the model ever saw it** -- a real, fixable recall bottleneck. |")
    lines.append(f"| Gold value in full text AND in selected evidence | {categories['in_evidence_still_wrong']} | {pct(categories['in_evidence_still_wrong'])} | The model had the evidence and still got it wrong/abstained -- a genuine model-capability or prompt-format issue, not retrieval. |")

    for cat_key, title in [("lost_in_selection", "Examples: lost in sentence selection"),
                             ("in_evidence_still_wrong", "Examples: evidence present, model still wrong"),
                             ("not_in_fulltext", "Examples: gold not verbatim in full text")]:
        lines.append(f"\n## {title}\n")
        for r in examples[cat_key]:
            lines.append(f"- paper {r['paper_id']}, field \"{r['field']}\": gold=`{r['gold_values']}`, "
                         f"label={r['label']}, predicted=`{r['cleaned_prediction']}`")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
