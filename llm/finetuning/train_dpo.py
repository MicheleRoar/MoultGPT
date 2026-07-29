"""
llm/finetuning/train_dpo.py
=============================

Preference-optimization (DPO) fine-tuning on top of the LoRA SFT model
from `train_lora.py`, using preference pairs built by
`feedback_to_preferences.py` from real 👍/👎 feedback collected through the
unified demo (`frontend/`). This is the second half of a loop that starts
at the feedback buttons in `frontend/index.html`:

    demo user rates a response (👍/👎)
        -> POST /feedback -> llm/backend/feedback/feedback.jsonl
        -> feedback_to_preferences.py -> {prompt, chosen, rejected} pairs
        -> train_dpo.py (this script) -> a DPO-tuned LoRA adapter

Uses Hugging Face `trl`'s `DPOTrainer` with a fresh LoRA adapter applied on
top of the (optionally SFT-merged) base model — `trl` handles the DPO
reference model implicitly when a `peft_config` is passed (the frozen base
weights under the new adapter serve as the reference policy).

IMPORTANT — like `llm/config/models.py` warns for the OpenRouter model
catalog, `trl`'s public API for DPO has moved across versions (older
releases took `ref_model` + a separate `TrainingArguments`; this script is
written against the current `DPOConfig`/`DPOTrainer` split in trl>=0.9).
Pin an exact `trl` version in `requirements-research.txt` before relying on
this for anything beyond local experimentation, and re-check the trainer's
constructor signature against your installed version if it errors.

Usage
-----
    # 1) Build preference pairs from collected feedback
    python feedback_to_preferences.py --out output/dpo_pairs.jsonl

    # 2) Train (optionally starting from an existing SFT LoRA adapter)
    python train_dpo.py \\
        --preferences output/dpo_pairs.jsonl \\
        --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3 \\
        --sft_adapter_path ../output/lora_mistral \\
        --output_dir ../output/dpo_mistral \\
        --tracking wandb --run_name mistral7b-dpo-v1

Requirements: `requirements-research.txt` in this directory, same as
`train_lora.py`, plus `trl`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, PeftModel, TaskType

DEFAULT_MODEL_PATH = os.environ.get(
    "MOULTGPT_BASE_MODEL",
    "mistralai/Mistral-7B-Instruct-v0.3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # --- data ---
    parser.add_argument("--preferences", type=str, required=True,
                         help="JSONL of {prompt, chosen, rejected} from feedback_to_preferences.py")
    parser.add_argument("--output_dir", type=str, default="dpo_output")
    parser.add_argument("--eval_split", type=float, default=0.1)
    parser.add_argument("--min_pairs", type=int, default=8,
                         help="Refuse to start training below this many pairs (use --force to override) — "
                              "DPO on a handful of examples mostly just overfits to them.")
    parser.add_argument("--force", action="store_true", help="Train anyway even below --min_pairs.")

    # --- model ---
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sft_adapter_path", type=str, default=None,
                         help="Optional existing LoRA adapter (e.g. train_lora.py's --output_dir) to merge in "
                              "as the SFT starting point before applying a fresh DPO LoRA on top.")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")

    # --- LoRA hyperparameters (the DPO adapter, separate from any SFT one) ---
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    # --- DPO / training hyperparameters ---
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature — how strongly to penalize the rejected completion.")
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                         help="DPO typically needs a much smaller LR than SFT — 5e-6 to 5e-5, not 2e-4.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # --- tracking ---
    parser.add_argument("--tracking", choices=["none", "wandb", "mlflow"], default="none")
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


def load_base_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype="auto",
        load_in_4bit=args.load_in_4bit,
    )

    if args.sft_adapter_path:
        print(f"[INFO] Merging SFT adapter from {args.sft_adapter_path} as the DPO starting point...")
        model = PeftModel.from_pretrained(model, args.sft_adapter_path)
        model = model.merge_and_unload()

    return model, tokenizer


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    prefs_path = Path(args.preferences)
    if not prefs_path.exists():
        print(f"[ERROR] {prefs_path} does not exist. Run feedback_to_preferences.py first.")
        sys.exit(1)

    dataset = load_dataset("json", data_files=str(prefs_path), split="train")
    n_pairs = len(dataset)
    print(f"[INFO] Loaded {n_pairs} preference pairs from {prefs_path}")

    if n_pairs < args.min_pairs and not args.force:
        print(
            f"[ERROR] Only {n_pairs} pairs (< --min_pairs={args.min_pairs}). "
            f"Collect more feedback via the unified demo (frontend/) first, or pass --force "
            f"if you specifically want to experiment on this little data."
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    if args.eval_split > 0 and n_pairs >= 10:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_dataset, eval_dataset = split["train"], split["test"]
    else:
        train_dataset, eval_dataset = dataset, None
        if args.eval_split > 0:
            print(f"[INFO] Too few pairs ({n_pairs}) for a held-out eval split — training on all of them.")

    print(f"[INFO] Loading model from {args.model_name_or_path}...")
    model, tokenizer = load_base_model(args)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Imported here (not at module top) so `--help` and feedback_to_preferences.py
    # unit tests don't require trl to be installed just to inspect this file.
    from trl import DPOConfig, DPOTrainer

    report_to = [args.tracking] if args.tracking != "none" else "none"
    if args.tracking == "mlflow":
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "moultgpt-dpo")
    if args.tracking == "wandb":
        os.environ.setdefault("WANDB_PROJECT", "moultgpt-dpo")

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        run_name=args.run_name,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to=report_to,
        seed=args.seed,
    )

    print("[INFO] Starting DPO training...")
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        print(f"[INFO] Final eval metrics: {metrics}")
        with open(output_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[INFO] DPO training completed. Adapter + tokenizer saved to {output_dir}")


if __name__ == "__main__":
    main()
