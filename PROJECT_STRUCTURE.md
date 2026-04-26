# Project Structure

This repository is organized by responsibility:

- `models/`: neural network definitions
  - `model.py`: original `BP_PINN`
  - `model_variants.py`: `BP_TCN`, `BP_HybridTransformer`
- `training/`: all training entry points and shared training utilities
  - `train.py`: baseline training
  - `train_map.py`: MAP-augmented training
  - `train_ablation_common.py`: shared trainer used by ablation/architecture scripts
  - `train_pp_mse.py`, `train_pp_smoothl1.py`, `train_pp_mse_scaled.py`
  - `train_tcn.py`, `train_hybrid_transformer.py`
- `evaluation/`: evaluation/inference tools
  - `generate_csv.py`
- `analysis/`: plotting and run-comparison helpers
  - `plot_results.py`, `visualize_sample.py`
  - `compare_ablation_runs.py`, `compare_architecture_runs.py`
  - `compare_loss_combo_runs.py`
- `slurm/`: all Slurm submission scripts
- `data/`: dataset files
- `checkpoints/`: saved model weights
- `logs/`: training/evaluation outputs and metrics history

## Common Commands

- Baseline train: `sbatch slurm/submit.sh`
- Evaluation CSV: `sbatch slurm/submit_eval.sh`
- MAP train: `sbatch slurm/submit_map.sh`
- Loss ablations (varying loss function, fixed seed):
  - `sbatch slurm/submit_pp_mse.sh`
  - `sbatch slurm/submit_pp_smoothl1.sh`
  - `sbatch slurm/submit_pp_mse_scaled.sh`
- Architecture runs:
  - `sbatch slurm/submit_tcn.sh`
  - `sbatch slurm/submit_hybrid_transformer.sh`
- **Loss combination comparison** (all three modes in one array job):
  - `sbatch slurm/submit_loss_combo.sh`
  - Array tasks: 0=pp, 1=map, 2=pp_map
  - Logs: `logs/loss_combo_<task>_<jobid>.log`

## Comparison Helpers

- Loss ablations summary:
  - `python analysis/compare_ablation_runs.py --log_dir ./logs`
- Architecture summary:
  - `python analysis/compare_architecture_runs.py --log_dir ./logs`
- **Loss combination comparison**:
  - `python analysis/compare_loss_combo_runs.py --log_dir ./logs`
