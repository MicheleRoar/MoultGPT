#!/bin/bash
#SBATCH --job-name=llm_back
#SBATCH --output=output/backend_%j.log
#SBATCH --error=output/backend_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# === Load required modules ===
module load openjdk/17.0.8.1_1

# === Navigate to project folder ===
cd /work/FAC/FBM/DEE/mrobinso/moult/michele/LLM

# === Activate Python environment ===
source mistral_env/bin/activate

# === Start GROBID service ===
echo "[INFO] Starting GROBID on port 8070..."
cd tools/grobid
nohup ./gradlew run > ../../output/grobid_%j.log 2>&1 &

# Optional: wait for GROBID to be available
echo "[INFO] Waiting for GROBID to start..."
sleep 20  # or use a curl check loop if needed

# === Return to root dir and launch Flask backend ===
cd ../../
echo "[INFO] Starting Flask backend on port 5001..."
python backend/app.py

