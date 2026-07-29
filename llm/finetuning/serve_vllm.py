"""
llm/finetuning/serve_vllm.py
=============================

Fast local inference for the LoRA-fine-tuned MoultGPT model, using vLLM's
paged-attention engine and dynamic LoRA adapter loading instead of plain
HF `generate()`. This is the "make the research artifact actually usable"
step after `train_lora.py`: once an adapter exists in `--output_dir`, this
script serves it for batch generation over the *same* dataset format and
*same* domain-gating + prompt-building pipeline used by
`eval/compare_models.py` (see that script's docstring for the JSON shape:
a list of `{paper_id, doi|pdf|text, queries}` objects).

That means the local fine-tuned model can be benchmarked on exactly the
same papers/queries/prompts as the remote models, and this script's output
CSV uses the same columns as `compare_models.py` (with `provider` set to
`"local-vllm-lora"`) so the two result files can simply be concatenated
into one comparison table.

Usage
-----
    python serve_vllm.py \\
        --base_model mistralai/Mistral-7B-Instruct-v0.3 \\
        --lora_path ../output/lora_mistral \\
        --dataset ../eval/dataset_example.json \\
        --out ../eval/results_local_lora.csv

Or run vLLM's own OpenAI-compatible server if you'd rather query the
fine-tuned model like any other HTTP provider (e.g. from `providers.py`):

    python -m vllm.entrypoints.openai.api_server \\
        --model mistralai/Mistral-7B-Instruct-v0.3 \\
        --enable-lora \\
        --lora-modules moultgpt-lora=../output/lora_mistral

Note: requires `requirements-research.txt` (vLLM + CUDA), and is Linux/GPU
only — vLLM does not support macOS/CPU-only inference.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

LLM_ROOT = Path(__file__).resolve().parents[1]
if str(LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_ROOT))

from pipeline.processor import input_to_text  # type: ignore
from pipeline.domain_pipeline import run_domain_pipeline  # type: ignore
from pipeline.prompting import (  # type: ignore
    MODE_FULL_TRAITS,
    VALID_MODES,
    build_system_prompt,
    build_user_content,
)
from domain.optimized_taxonomy_lookup import TaxonomyLookup  # type: ignore
from domain.moulting_ontology_gate import MoultingOntologyGate  # type: ignore
from eval.compare_models import load_dataset, get_full_text  # type: ignore

PROVIDER_LABEL = "local-vllm-lora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base_model", type=str, required=True, help="HF hub id or local path of the base model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA adapter produced by train_lora.py")
    parser.add_argument("--dataset", type=str, required=True, help="Same dataset JSON format as eval/compare_models.py")
    parser.add_argument("--out", type=str, default=str(LLM_ROOT / "eval" / "results_local_lora.csv"))
    parser.add_argument("--mode", type=str, default=MODE_FULL_TRAITS, choices=list(VALID_MODES))
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 for deterministic, reproducible eval runs")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--taxonomy_csv", type=str, default=str(LLM_ROOT / "data" / "arthropod_taxonomy.csv"))
    parser.add_argument("--taxonomy_pickle", type=str, default=str(LLM_ROOT / "data" / "taxonomy_lookup.pkl"))
    parser.add_argument("--ontology_owl", type=str, default=str(LLM_ROOT / "data" / "moultdb_moulting_ontology_v3_8.owl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Imported lazily: vLLM pulls in CUDA/torch, and the rest of this
    # script (dataset loading, gating) should stay importable without it.
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    dataset = load_dataset(Path(args.dataset))

    print("[INFO] Loading taxonomy lookup + ontology gate...")
    taxonomy_lookup = TaxonomyLookup(csv_path=args.taxonomy_csv, pickle_path=args.taxonomy_pickle, rebuild=False)
    ontology_gate = MoultingOntologyGate(owl_path=args.ontology_owl)

    print(f"[INFO] Loading base model {args.base_model} with LoRA support enabled...")
    llm = LLM(model=args.base_model, enable_lora=True, max_lora_rank=64, gpu_memory_utilization=args.gpu_memory_utilization)
    lora_request = LoRARequest("moultgpt-lora", 1, args.lora_path)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    system_prompt = build_system_prompt(args.mode)
    rows: List[Dict[str, Any]] = []
    to_generate: List[Dict[str, Any]] = []  # rows still needing a model call, plus their built prompt

    for paper_entry in dataset:
        paper_id = paper_entry.get("paper_id", "unknown")
        print(f"\n=== Paper: {paper_id} ===")

        full_text: Optional[str] = get_full_text(paper_entry)
        if not full_text or len(full_text.strip()) < 100:
            print(f"[WARN] Could not extract text for paper '{paper_id}', skipping.")
            continue

        for query in paper_entry.get("queries", []):
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
                "provider": PROVIDER_LABEL,
                "model_id": args.base_model,
                "resolved_model_id": f"{args.base_model}+{Path(args.lora_path).name}",
                "model_label": "Local LoRA fine-tune (vLLM)",
            }

            if not decision["allow"] or not summary.strip():
                rows.append({**base_row, "response": None, "latency_sec": None, "error": "gated_out_or_empty_summary"})
                continue

            user_content = build_user_content(summary, query, args.mode)
            prompt = f"<s>[INST] {system_prompt}\n\n{user_content} [/INST]"
            to_generate.append({"row": base_row, "prompt": prompt})

    if to_generate:
        print(f"\n[INFO] Generating for {len(to_generate)} (paper, query) pairs with vLLM...")
        t0 = time.time()
        outputs = llm.generate([item["prompt"] for item in to_generate], sampling_params, lora_request=lora_request)
        elapsed = time.time() - t0
        for item, output in zip(to_generate, outputs):
            rows.append({
                **item["row"],
                "response": output.outputs[0].text.strip(),
                "latency_sec": round(elapsed / len(to_generate), 2),  # batched call: report the average
                "error": None,
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper_id", "query", "mode", "allow", "routing_label", "routing_message",
        "n_summary_sentences", "provider", "model_id", "resolved_model_id",
        "model_label", "response", "latency_sec", "error",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
