#!/bin/bash
#SBATCH --job-name=finetune_moultgpt
#SBATCH --output=finetune_output.log
#SBATCH --error=finetune_error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu

# Load Java
module load openjdk/17.0.8.1_1

# Activate the Python environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mistral_env

# Start GROBID in the background
cd "$(dirname "$0")/../tools/grobid/grobid-0.7.1/"
nohup ./gradlew run > grobid.log 2>&1 &

# Wait for GROBID to be available on port 8070 (max 60s)
echo "Waiting for GROBID to be ready..."
for i in {{1..60}}; do
  nc -z localhost 8070 && break
  sleep 1
done

# Launch the main dataset generation
cd "$(dirname "$0")/modules"
python main_generate_dataset.py
