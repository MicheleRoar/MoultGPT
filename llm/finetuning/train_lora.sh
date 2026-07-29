#!/bin/bash
#SBATCH --job-name=lora_finetune
#SBATCH --output=output/lora_finetune_%j.log
#SBATCH --error=output/lora_finetune_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

# === Activate environment ===
cd /work/FAC/FBM/DEE/mrobinso/moult/michele/LLM
source mistral_env/bin/activate

# === Launch fine-tuning ===
cd finetuning
python train_lora.py \
    --dataset output/finetune_full.jsonl \
    --output_dir ../output/lora_mistral \
    --model_name_or_path "$MOULTGPT_BASE_MODEL" \
    --tracking wandb \
    --run_name "lora-mistral-${SLURM_JOB_ID}" \
    --seed 42

# For multi-GPU nodes, launch with accelerate instead, e.g.:
#   accelerate launch --config_file accelerate_config.yaml train_lora.py --dataset ... --output_dir ...

