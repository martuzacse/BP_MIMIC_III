#!/bin/bash
#SBATCH --job-name=WS_ATTN_PPM
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ws_attn_bilstm_pp_map_%j.log

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/mmfs1/home/0890ahamadm/BP_Project}"
cd "$PROJECT_ROOT"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate rajgpu

SCRATCH_FILE="/tmp/0890ahamadm_data_ws_attn_bilstm_pp_map_${SLURM_JOB_ID}.h5"
cp /mmfs1/home/0890ahamadm/BP_Project/data/MIMIC-III_ppg_dataset.h5 "$SCRATCH_FILE"

python -m training.run_experiment \
  --data_path "$SCRATCH_FILE" \
  --model     attn_bilstm \
  --loss_mode pp_map \
  --split     within_subject \
  --seed 42 --split_seed 42 --epochs 60 --patience 15 \
  --pp_weight 0.1 --map_weight 0.05

rm -f "$SCRATCH_FILE"
