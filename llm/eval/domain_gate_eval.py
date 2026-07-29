#!/usr/bin/env python3
# llm/eval/domain_gate_eval.py
"""
Real, executed evaluation of the domain gate (domain/domain_gate.py) for
the paper's "Domain Gating Performance" placeholder: accepted/rejected
counts against a small hand-labeled set of document-query pairs, with a
confusion table and rejected-query examples.

No network access needed: taxonomy + ontology gate load from local files;
paper text comes from the already-parsed llm/finetuning/papers/*.tei.xml.

Evaluation set (documented, not hidden, given its small size)
---------------------------------------------------------------
- POSITIVE pairs (expected allow=True): each of the 21 real papers in the
  corpus, paired with a generic in-domain query. Every one of these 21
  papers has real, expert-annotated moulting trait values in the MoultDB
  ground truth (llm/finetuning/MoultDB character annotations.xlsx), so
  they are all genuine positives by construction, not guessed.
- NEGATIVE QUERY controls (expected allow=False): the same 21 real papers,
  but paired with an explicitly non-arthropod moulting query (bird feather
  moult). Tests whether the query gate alone can reject an off-domain
  question regardless of an otherwise in-domain paper.
- NEGATIVE PAPER controls (expected allow=False): three short SYNTHETIC
  (hand-written for this test, not real literature excerpts) passages
  about non-arthropod moulting/shedding phenomena (bird feather moult,
  snake ecdysis, human hair shedding), paired with a generic moulting
  query. Tests whether the taxonomy gate alone can reject a paper with no
  arthropod signal. These are clearly labeled as synthetic in the output
  -- not passed off as real papers.

n=21 positives + 21 negative-query controls + 3 negative-paper controls =
45 pairs is a real but modest evaluation set, sized to what's actually
constructible from this repo's own corpus without inventing a larger
"ground truth" of off-domain papers that doesn't exist. See the paper's
Discussion for what this evaluation does and doesn't establish.

Usage
-----
    cd llm
    python eval/domain_gate_eval.py --out eval/domain_gate_eval.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

EVAL_ROOT = Path(__file__).resolve().parent
LLM_ROOT = EVAL_ROOT.parent
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

from pipeline.parser import tei_to_text  # type: ignore  # noqa: E402
from pipeline.domain_pipeline import run_domain_pipeline  # type: ignore  # noqa: E402
from domain.optimized_taxonomy_lookup import TaxonomyLookup  # type: ignore  # noqa: E402
from domain.moulting_ontology_gate import MoultingOntologyGate  # type: ignore  # noqa: E402

DEFAULT_PAPERS_DIR = LLM_ROOT / "finetuning" / "papers"
DEFAULT_TAXONOMY_CSV = LLM_ROOT / "data" / "arthropod_taxonomy.csv"
DEFAULT_TAXONOMY_PICKLE = LLM_ROOT / "data" / "taxonomy_lookup.pkl"
DEFAULT_OWL_PATH = LLM_ROOT / "data" / "moultdb_moulting_ontology_v3_8.owl"

IN_DOMAIN_QUERY = "What are the moulting-related traits described in this paper?"
NON_ARTHROPOD_QUERY = "Describe feather moulting behaviour in birds."

# Hand-written, deliberately off-domain synthetic passages -- see module
# docstring: NOT real literature, just enough arthropod-free "shedding"
# text to test whether the taxonomy gate correctly finds no arthropod
# signal and rejects them regardless of query.
SYNTHETIC_NEGATIVE_PAPERS = {
    "synthetic_bird_feather": (
        "Feather moult in birds is a seasonally regulated process in which old, worn feathers are "
        "shed and replaced by new plumage. Most passerine species undergo a complete post-breeding "
        "moult once per year, while some undergo a partial pre-breeding moult as well. Moult "
        "sequence typically proceeds symmetrically from the innermost primary feathers outward, and "
        "moult duration varies with body size, migratory strategy, and breeding investment."
    ),
    "synthetic_snake_ecdysis": (
        "Ecdysis in snakes involves the periodic shedding of the entire outer layer of skin. Prior "
        "to shedding, the eyes become cloudy as a lubricating fluid separates the old skin from the "
        "new one beneath. The snake then rubs its snout against a rough surface to initiate a tear, "
        "and crawls out of the old skin, which is typically shed in one piece starting from the head."
    ),
    "synthetic_human_hair": (
        "Human hair follicles cycle through growth (anagen), regression (catagen), and rest "
        "(telogen) phases, with roughly 50 to 100 scalp hairs shed daily during the telogen phase. "
        "Hair shedding rate can be influenced by hormonal changes, nutritional status, and seasonal "
        "variation, and is distinct from pathological hair loss conditions such as alopecia."
    ),
}


def get_paper_ids(papers_dir: Path) -> List[str]:
    return sorted((p.stem.split(".")[0] for p in papers_dir.glob("*.tei.xml")),
                  key=lambda s: int(s))


def evaluate_pair(full_text: str, query: str, taxonomy_lookup, ontology_gate) -> Dict[str, Any]:
    result = run_domain_pipeline(
        full_text=full_text, user_query=query,
        taxonomy_lookup=taxonomy_lookup, ontology_gate=ontology_gate,
    )
    return result["decision"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--papers_dir", type=str, default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--taxonomy_csv", type=str, default=str(DEFAULT_TAXONOMY_CSV))
    parser.add_argument("--taxonomy_pickle", type=str, default=str(DEFAULT_TAXONOMY_PICKLE))
    parser.add_argument("--ontology_owl", type=str, default=str(DEFAULT_OWL_PATH))
    parser.add_argument("--out", type=str, default=str(EVAL_ROOT / "domain_gate_eval.md"))
    args = parser.parse_args()

    papers_dir = Path(args.papers_dir)
    paper_ids = get_paper_ids(papers_dir)
    print(f"[INFO] {len(paper_ids)} real papers found.")

    print("[INFO] Loading taxonomy lookup + ontology gate...")
    taxonomy_lookup = TaxonomyLookup(csv_path=args.taxonomy_csv, pickle_path=args.taxonomy_pickle, rebuild=False)
    ontology_gate = MoultingOntologyGate(owl_path=args.ontology_owl)

    rows: List[Dict[str, Any]] = []

    for pid in paper_ids:
        full_text = tei_to_text(str(papers_dir / f"{pid}.tei.xml"))
        decision = evaluate_pair(full_text, IN_DOMAIN_QUERY, taxonomy_lookup, ontology_gate)
        rows.append({"pair": f"paper {pid} + in-domain query", "condition": "positive",
                     "expected": True, "actual": decision["allow"], "label": decision["final_label"]})

        decision_neg_q = evaluate_pair(full_text, NON_ARTHROPOD_QUERY, taxonomy_lookup, ontology_gate)
        rows.append({"pair": f"paper {pid} + non-arthropod query", "condition": "negative_query",
                     "expected": False, "actual": decision_neg_q["allow"], "label": decision_neg_q["final_label"]})
        print(f"  paper {pid}: positive={decision['allow']} ({decision['final_label']}), "
              f"neg_query={decision_neg_q['allow']} ({decision_neg_q['final_label']})")

    for name, text in SYNTHETIC_NEGATIVE_PAPERS.items():
        decision = evaluate_pair(text, IN_DOMAIN_QUERY, taxonomy_lookup, ontology_gate)
        rows.append({"pair": f"{name} + in-domain query", "condition": "negative_paper",
                     "expected": False, "actual": decision["allow"], "label": decision["final_label"]})
        print(f"  {name}: allow={decision['allow']} ({decision['final_label']})")

    _write_report(Path(args.out), rows)
    print(f"\n[DONE] Wrote {args.out}")


def _write_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    tp = sum(1 for r in rows if r["expected"] and r["actual"])
    fn = sum(1 for r in rows if r["expected"] and not r["actual"])
    tn = sum(1 for r in rows if not r["expected"] and not r["actual"])
    fp = sum(1 for r in rows if not r["expected"] and r["actual"])
    n = len(rows)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    lines = []
    lines.append("# Domain gate — real evaluation\n")
    lines.append(
        f"Generated by `domain_gate_eval.py`. {n} document-query pairs: "
        f"{sum(1 for r in rows if r['condition']=='positive')} positive (real paper + in-domain query), "
        f"{sum(1 for r in rows if r['condition']=='negative_query')} negative-query controls (real paper + "
        f"off-domain query), {sum(1 for r in rows if r['condition']=='negative_paper')} negative-paper "
        f"controls (synthetic non-arthropod text + in-domain query). See this script's module docstring "
        f"for exactly how each set was constructed and its limitations.\n"
    )
    lines.append("## Confusion table\n")
    lines.append("| | Predicted allow=True | Predicted allow=False |")
    lines.append("|---|---|---|")
    lines.append(f"| **Expected allow=True** | TP={tp} | FN={fn} |")
    lines.append(f"| **Expected allow=False** | FP={fp} | TN={tn} |")
    lines.append(f"\nAccuracy: {accuracy:.3f} ({tp+tn}/{n}). Precision: {precision:.3f}. Recall: {recall:.3f}.\n")

    lines.append("## False negatives (real moulting papers/queries incorrectly rejected)\n")
    fn_rows = [r for r in rows if r["expected"] and not r["actual"]]
    if fn_rows:
        for r in fn_rows:
            lines.append(f"- {r['pair']}: rejected as `{r['label']}`")
    else:
        lines.append("(none)")

    lines.append("\n## False positives (negative controls incorrectly allowed)\n")
    fp_rows = [r for r in rows if not r["expected"] and r["actual"]]
    if fp_rows:
        for r in fp_rows:
            lines.append(f"- {r['pair']}: accepted as `{r['label']}`")
    else:
        lines.append("(none)")

    lines.append("\n## Correctly rejected negative controls (examples)\n")
    correct_neg = [r for r in rows if not r["expected"] and not r["actual"]][:5]
    for r in correct_neg:
        lines.append(f"- {r['pair']}: rejected as `{r['label']}` (correct)")

    lines.append(
        "\n## Limitations\n"
        "- Negative-paper controls are 3 hand-written synthetic passages, not real literature -- a "
        "real non-arthropod paper (with citations, methods sections, figure captions, etc.) may "
        "stress the taxonomy gate differently than a short clean paragraph does.\n"
        "- All 21 positive-set papers come from the same curated corpus (llm/finetuning/papers/), "
        "which was itself assembled by the project as clearly moulting-relevant -- this evaluation "
        "measures recall on \"papers a domain expert already judged relevant\", not on an unfiltered "
        "sample of the literature.\n"
        "- n=45 pairs total is enough to catch a badly miscalibrated gate, not enough to report a "
        "tight confidence interval on precision/recall.\n"
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
