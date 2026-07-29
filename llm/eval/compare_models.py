# llm/eval/compare_models.py
"""
Run the SAME MoultGPT pipeline (paper -> ontology summary -> domain gate ->
LLM call) across several remote models, for side-by-side comparison.

This is the benchmarking counterpart to llm/backend/app.py: both call
pipeline.domain_pipeline.run_domain_pipeline() for the summary/gating step,
so the only thing that varies between rows in the output is which model
answered the query. That is what makes the comparison meaningful for a
publication — every model sees exactly the same context and question.

Input: a small JSON dataset describing which papers/queries to test, e.g.

    [
      {
        "paper_id": "smith2020",
        "doi": "10.1038/s41598-022-18146-3",
        "queries": [
          "What moulting stage is described for the adult specimens?",
          "How many instars are reported for this species?"
        ]
      },
      {
        "paper_id": "local_pdf_example",
        "pdf": "llm/test/paper.pdf",
        "queries": ["Describe the moulting process in this species."]
      }
    ]

Output: one CSV row per (paper, query, model) with the model's raw response,
the routing decision, and timing — ready to drop into a results table.

Usage:
    cd llm
    python eval/compare_models.py --dataset eval/dataset_example.json \\
        --out eval/results.csv

    # restrict to a subset of the registry:
    python eval/compare_models.py --dataset eval/dataset_example.json \\
        --models mistral-small-latest meta-llama/llama-3.3-70b-instruct:free
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

LLM_ROOT = Path(__file__).resolve().parents[1]
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(LLM_ROOT / ".env")
except ImportError:
    pass

from pipeline.processor import input_to_text  # type: ignore
from pipeline.domain_pipeline import run_domain_pipeline  # type: ignore

from domain.optimized_taxonomy_lookup import TaxonomyLookup  # type: ignore
from domain.moulting_ontology_gate import MoultingOntologyGate  # type: ignore

from backend.providers import generate as llm_generate, ProviderError  # type: ignore
from config.models import list_models  # type: ignore
from pipeline.prompting import (  # type: ignore
    MODE_FULL_TRAITS,
    VALID_MODES,
    build_system_prompt,
    build_user_content,
)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_full_text(entry: Dict[str, Any]) -> Optional[str]:
    if entry.get("doi"):
        return input_to_text(doi=entry["doi"])
    if entry.get("pdf"):
        return input_to_text(pdf_path=entry["pdf"])
    if entry.get("text"):
        return entry["text"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset JSON file (see module docstring for the format).")
    parser.add_argument("--out", type=str, default=str(LLM_ROOT / "eval" / "results.csv"), help="Output CSV path.")
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Subset of model ids to run (default: everything in llm/config/models.py).")
    parser.add_argument("--taxonomy_csv", type=str, default=str(LLM_ROOT / "data" / "arthropod_taxonomy.csv"))
    parser.add_argument("--taxonomy_pickle", type=str, default=str(LLM_ROOT / "data" / "taxonomy_lookup.pkl"))
    parser.add_argument("--ontology_owl", type=str, default=str(LLM_ROOT / "data" / "moultdb_moulting_ontology_v3_8.owl"))
    parser.add_argument("--mode", type=str, default=MODE_FULL_TRAITS, choices=list(VALID_MODES),
                         help="full_traits (default, ~55 MoultDB fields) or single_trait (answer only the query).")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                         help="Defaults to 1024 for full_traits, 256 for single_trait.")
    args = parser.parse_args()

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = 1024 if args.mode == MODE_FULL_TRAITS else 256

    dataset = load_dataset(Path(args.dataset))

    all_models = list_models()
    if args.models:
        wanted = set(args.models)
        models_to_run = [m for m in all_models if m["id"] in wanted]
        missing = wanted - {m["id"] for m in models_to_run}
        if missing:
            print(f"[WARN] Unknown model ids ignored: {sorted(missing)}")
    else:
        models_to_run = all_models

    print(f"[INFO] Comparing {len(models_to_run)} models: {[m['id'] for m in models_to_run]}")

    print("[INFO] Loading taxonomy lookup + ontology gate...")
    taxonomy_lookup = TaxonomyLookup(
        csv_path=args.taxonomy_csv,
        pickle_path=args.taxonomy_pickle,
        rebuild=False,
    )
    ontology_gate = MoultingOntologyGate(owl_path=args.ontology_owl)

    rows: List[Dict[str, Any]] = []

    for paper_entry in dataset:
        paper_id = paper_entry.get("paper_id", "unknown")
        print(f"\n=== Paper: {paper_id} ===")

        full_text = get_full_text(paper_entry)
        if not full_text or len(full_text.strip()) < 100:
            print(f"[WARN] Could not extract text for paper '{paper_id}', skipping.")
            continue

        for query in paper_entry.get("queries", []):
            print(f"  Query: {query}")

            pipeline_result = run_domain_pipeline(
                full_text=full_text,
                user_query=query,
                taxonomy_lookup=taxonomy_lookup,
                ontology_gate=ontology_gate,
            )
            summary = pipeline_result["summary"]
            decision = pipeline_result["decision"]

            base_row = {
                "paper_id": paper_id,
                "query": query,
                "mode": args.mode,
                "allow": decision["allow"],
                "routing_label": decision["final_label"],
                "routing_message": decision["message"],
                "n_summary_sentences": decision["paper_summary_gate"]["n_summary_sentences"],
            }

            if not decision["allow"] or not summary.strip():
                # Still record one row so the "why was this skipped" reason
                # shows up in the results table, but don't call any model.
                rows.append({**base_row, "provider": None, "model_id": None,
                             "model_label": None, "response": None,
                             "latency_sec": None, "error": "gated_out_or_empty_summary"})
                continue

            system_prompt = build_system_prompt(args.mode)
            user_content = build_user_content(summary, query, args.mode)

            for model in models_to_run:
                model_id = model["id"]
                print(f"    -> {model_id} ...", end=" ", flush=True)
                t0 = time.time()
                try:
                    result = llm_generate(
                        system_prompt=system_prompt,
                        user_content=user_content,
                        model_id=model_id,
                        max_new_tokens=max_new_tokens,
                    )
                    rows.append({
                        **base_row,
                        "provider": result["provider"],
                        "model_id": result["model_id"],
                        "resolved_model_id": result["resolved_model_id"],
                        "model_label": result["model_label"],
                        "response": result["text"],
                        "latency_sec": round(result["latency_sec"], 2),
                        "error": None,
                    })
                    resolved_note = (
                        f", resolved={result['resolved_model_id']}"
                        if result["resolved_model_id"] != model_id else ""
                    )
                    print(f"ok ({time.time() - t0:.1f}s{resolved_note})")
                except ProviderError as e:
                    rows.append({
                        **base_row,
                        "provider": model["provider"],
                        "model_id": model_id,
                        "resolved_model_id": None,
                        "model_label": model["label"],
                        "response": None,
                        "latency_sec": None,
                        "error": str(e),
                    })
                    print(f"FAILED: {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper_id", "query", "mode", "allow", "routing_label", "routing_message",
        "n_summary_sentences", "provider", "model_id", "resolved_model_id", "model_label",
        "response", "latency_sec", "error",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[DONE] Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
