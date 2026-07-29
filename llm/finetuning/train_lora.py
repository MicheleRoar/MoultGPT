"""
llm/finetuning/train_lora.py
=============================

PEFT/LoRA (optionally 4-bit / QLoRA) supervised fine-tuning of a local
causal LM on the MoultGPT trait-extraction dataset produced by
`main_generate_dataset.py`.

This script is intentionally separate from the remote-API backend
(`llm/backend/`): the backend calls hosted models for the live product,
while this is the research-side training loop used to fine-tune and
evaluate a local base model (originally Mistral-7B-Instruct) on the
moulting trait-extraction task, and to compare that specialised model
against the remote general-purpose ones in `llm/eval/compare_models.py`.

What this adds over a minimal LoRA script, on purpose, for reproducible
research rather than a one-off run:
  - a held-out eval split with per-epoch eval loss / perplexity, instead of
    reporting only training loss;
  - all training/LoRA hyperparameters are CLI flags, not hardcoded, so a
    sweep is just several invocations with different args;
  - a fixed seed plus a `run_config.json` dump of every argument, so a run
    can be reproduced or diffed against another;
  - optional experiment tracking (Weights & Biases or MLflow) via
    `--tracking`, off by default so the script still runs with zero
    tracking infra configured.

Usage
-----
    python train_lora.py \\
        --dataset ../finetuning/output/finetune_full.jsonl \\
        --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3 \\
        --output_dir ../output/lora_mistral \\
        --tracking wandb --run_name mistral7b-lora-r8

Requirements: see `requirements-research.txt` in this directory (torch,
transformers, peft, accelerate, bitsandbytes, datasets, plus wandb/mlflow
if using `--tracking`) — deliberately kept out of the lean `llm/requirements.txt`
used by the remote-only backend (see that file's header comment).

Multi-GPU / distributed: launch with `accelerate launch` instead of `python`
(e.g. `accelerate launch --config_file accelerate_config.yaml train_lora.py ...`);
see `accelerate_config.yaml` in this directory for an FSDP starting point.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model

DEFAULT_MODEL_PATH = os.environ.get(
    "MOULTGPT_BASE_MODEL",
    "mistralai/Mistral-7B-Instruct-v0.3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # --- data ---
    parser.add_argument("--dataset", type=str, required=True, help="Path to the .jsonl dataset file")
    parser.add_argument("--output_dir", type=str, default="lora_output", help="Output directory for the fine-tuned adapter")
    parser.add_argument("--eval_split", type=float, default=0.1, help="Fraction of the dataset held out for eval (0 disables eval)")
    parser.add_argument("--max_length", type=int, default=1024, help="Tokenizer truncation/padding length")

    # --- model / quantization ---
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL_PATH,
                         help="HF hub id or local path of the base model. Defaults to $MOULTGPT_BASE_MODEL.")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="QLoRA-style 4-bit base model loading (default: on)")
    parser.add_argument("--no_4bit", dest="load_in_4bit", action="store_false", help="Disable 4-bit loading (full/half precision LoRA)")

    # --- LoRA hyperparameters ---
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj",
                         help="Comma-separated module names to wrap with LoRA adapters")

    # --- training hyperparameters ---
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="Seed for dataset split, init, and Trainer (reproducibility)")

    # --- experiment tracking ---
    parser.add_argument("--tracking", choices=["none", "wandb", "mlflow"], default="none",
                         help="Experiment tracker to report metrics to. Off by default.")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for the tracker / output logs")

    return parser.parse_args()


def build_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(model_name_or_path: str, load_in_4bit: bool):
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype="auto",
        load_in_4bit=load_in_4bit,
    )


def make_tokenize_fn(tokenizer: AutoTokenizer, max_length: int):
    def tokenize(example):
        prompt = example["instruction"]
        if example.get("input"):
            prompt += "\n" + example["input"]
        prompt += "\n\n### Answer:\n" + example["output"]
        return tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length)

    return tokenize


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dump the exact config for this run so it can be reproduced or diffed
    # against another run later — the point of a fixed seed is undermined
    # if the hyperparameters used aren't recorded anywhere.
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("[INFO] Loading dataset...")
    full_dataset = load_dataset("json", data_files=args.dataset, split="train")

    if args.eval_split > 0:
        split = full_dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_dataset, eval_dataset = split["train"], split["test"]
        print(f"[INFO] Split dataset: {len(train_dataset)} train / {len(eval_dataset)} eval")
    else:
        train_dataset, eval_dataset = full_dataset, None
        print(f"[INFO] Using full dataset for training ({len(train_dataset)} examples), no eval split")

    print(f"[INFO] Loading model from {args.model_name_or_path}...")
    tokenizer = build_tokenizer(args.model_name_or_path)
    model = build_model(args.model_name_or_path, args.load_in_4bit)

    print("[INFO] Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenize = make_tokenize_fn(tokenizer, args.max_length)
    train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(tokenize, remove_columns=eval_dataset.column_names)

    report_to = [args.tracking] if args.tracking != "none" else "none"
    if args.tracking == "mlflow":
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "moultgpt-lora")
    if args.tracking == "wandb":
        os.environ.setdefault("WANDB_PROJECT", "moultgpt-lora")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        report_to=report_to,
        seed=args.seed,
    )

    print("[INFO] Starting fine-tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    if eval_dataset is not None:
        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            metrics["eval_perplexity"] = math.exp(eval_loss)
        print(f"[INFO] Final eval metrics: {metrics}")
        with open(output_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[INFO] Training completed. Adapter + tokenizer saved to {output_dir}")


if __name__ == "__main__":
    main()
