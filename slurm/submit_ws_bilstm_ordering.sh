#!/bin/bash
#SBATCH --job-name=WS_BLSTM_ORD
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ws_bilstm_ordering_%j.log

# Ordering constraint loss: MSE + lambda * ReLU(10 - (SBP_pred - DBP_pred))^2
# Enforces SBP - DBP >= 10 mmHg (physiological ordering).
# lambda=1.0 — strong enough to actually penalise violations.

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/mmfs1/home/0890ahamadm/BP_Project}"
cd "$PROJECT_ROOT"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate rajgpu

SCRATCH_FILE="/tmp/0890ahamadm_data_ws_bilstm_ordering_${SLURM_JOB_ID}.h5"
cp /mmfs1/home/0890ahamadm/BP_Project/data/MIMIC-III_ppg_dataset.h5 "$SCRATCH_FILE"

python -m training.run_experiment \
  --data_path "$SCRATCH_FILE" \
  --model     bilstm \
  --loss_mode ordering \
  --split     within_subject \
  --seed 42 --split_seed 42 --epochs 60 --patience 15 \
  --pp_weight 1.0

rm -f "$SCRATCH_FILE"
