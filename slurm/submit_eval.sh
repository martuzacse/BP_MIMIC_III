#!/bin/bash
#SBATCH --job-name=BP_EVAL
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_out_%j.log

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/mmfs1/home/0890ahamadm/BP_Project}"
cd "$PROJECT_ROOT"

# Load Environment
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate rajgpu

# Data Transfer
SCRATCH_FILE="/tmp/0890ahamadm_data.h5"
cp /mmfs1/home/0890ahamadm/BP_Project/data/MIMIC-III_ppg_dataset.h5 $SCRATCH_FILE

# Run Evaluation
python evaluation/generate_csv.py --data_path $SCRATCH_FILE

# Cleanup
rm $SCRATCH_FILE
