import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling
import os

# === Argparse for flexibility ===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Path to the .jsonl dataset file")
parser.add_argument("--output_dir", type=str, default="lora_output", help="Output directory for fine-tuned model")
args = parser.parse_args()

# === Load dataset ===
print("[INFO] Loading dataset...")
dataset = load_dataset("json", data_files=args.dataset, split="train")

# === Model and tokenizer path ===
MODEL_PATH = "/reference/LLMs/Mistral_AI/mistral-7B-Instruct-v0.3-hf"
print(f"[INFO] Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=True
)

# === Apply LoRA ===
print("[INFO] Applying LoRA configuration...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# === Tokenization ===
def tokenize(example):
    prompt = example["instruction"]
    if example.get("input"):
        prompt += "\n" + example["input"]
    prompt += "\n\n### Answer:\n" + example["output"]
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=1024)

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# === Training configuration ===
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    report_to="none"
)

# === Launch training ===
print("[INFO] Starting fine-tuning...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print("[INFO] Training completed. Model saved.")
