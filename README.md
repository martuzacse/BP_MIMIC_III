# Blood Pressure Estimation from PPG (MIMIC-III)

Estimating systolic (SBP) and diastolic (DBP) blood pressure from raw photoplethysmography (PPG) signals using deep learning on the MIMIC-III waveform dataset.

---

## Model

**BP_PINN** (`models/model.py`) — a Conv1D + Bidirectional LSTM architecture:

```
Conv1D encoder (1→64→128 channels)
      ↓
BiLSTM (hidden=128, bidirectional → 256-dim output)
      ↓
Regressor (256×32 → 128 → 2)   [SBP, DBP in mmHg]
```

Model variants (`models/model_variants.py`):
- **BP_TCN** — dilated residual temporal convolutional network
- **BP_HybridTransformer** — Conv1D stem + 3-layer Transformer encoder

---

## Dataset

**MIMIC-III PPG dataset** (`data/MIMIC-III_ppg_dataset.h5`)

| Property | Value |
|----------|-------|
| Subjects | 4,527 |
| Samples per subject | 2,000 |
| Total samples | 9,054,000 |
| Input | Raw PPG waveform |
| Labels | [SBP, DBP] in mmHg |

HDF5 keys: `ppg`, `label`, `subject_idx`

---

## Experiments

### 1. Loss Combination Study (cross-subject split, seed=42)

All three variants train the same BP_PINN with a different auxiliary loss term alongside MSE (λ_pp=0.1, λ_map=0.05).

| Loss | SBP MAE | DBP MAE | AVG MAE | SBP R² | DBP R² |
|------|--------:|--------:|--------:|-------:|-------:|
| MSE only | 15.03 | 8.11 | 11.57 | 0.3544 | 0.2572 |
| MSE + λ_pp · PP_MSE | 15.02 | 8.10 | 11.56 | 0.3564 | 0.2583 |
| MSE + λ_map · MAP_MSE | 15.02 | 8.12 | 11.57 | 0.3570 | 0.2585 |
| MSE + λ_pp · PP_MSE + λ_map · MAP_MSE | 15.02 | **8.08** | **11.55** | 0.3564 | **0.2610** |

PP_MSE = pulse pressure constraint: (SBP − DBP)
MAP_MSE = mean arterial pressure constraint: (SBP + 2·DBP) / 3

**Finding:** auxiliary loss terms provide negligible gain at these weights — the bottleneck is the split strategy, not the loss.

### 2. Architecture Comparison (cross-subject split, 3 seeds: 42/43/44)

| Model | AVG MAE | SBP MAE | DBP MAE | SBP R² | DBP R² |
|-------|--------:|--------:|--------:|-------:|-------:|
| BP_PINN (Bi-LSTM) | **11.55** | **15.02** | **8.08** | **0.356** | **0.261** |
| BP_HybridTransformer | 14.71 ± 0.05 | 19.71 ± 0.05 | 9.71 ± 0.05 | 0.004 ± 0.009 | 0.002 ± 0.004 |

**Finding:** Hybrid Transformer achieves R² ≈ 0 across all seeds (predicts near the population mean). BP_PINN is the best architecture.

### 3. Within-Subject Split (BP_PINN, best loss: pp_map, seed=42)

Each subject's 2,000 samples are split temporally:
- Train: first 1,400 (70%)
- Val: next 300 (15%)
- Test: last 300 (15%)

| Split | Loss | SBP MAE | DBP MAE | AVG MAE | SBP R² | DBP R² |
|-------|------|--------:|--------:|--------:|-------:|-------:|
| Cross-subject | pp_map | 15.02 | 8.08 | 11.55 | 0.3564 | 0.2610 |
| **Within-subject** | **pp_map** | **12.82** | **7.27** | **10.05** | **0.5035** | **0.3977** |

**Finding:** Within-subject split raises SBP R² by +0.15 and DBP R² by +0.14 vs cross-subject. Inter-subject variability is the main bottleneck.

---

## Setup

```bash
conda activate rajgpu
```

Dependencies: `torch`, `h5py`, `numpy`

---

## Running Experiments

### Loss combination comparison (3 separate jobs)
```bash
sbatch slurm/submit_loss_mse.sh
sbatch slurm/submit_loss_pp.sh
sbatch slurm/submit_loss_map.sh
sbatch slurm/submit_loss_pp_map.sh
```

### Within-subject split (best loss)
```bash
sbatch slurm/submit_within_subject.sh
```

### Architecture comparison
```bash
sbatch slurm/submit_tcn.sh
sbatch slurm/submit_hybrid_transformer.sh
```

### Compare results
```bash
# Loss combinations
python analysis/compare_loss_combo_runs.py --log_dir ./logs

# Architecture runs
python analysis/compare_architecture_runs.py --log_dir ./logs

# Loss ablations (pp_mse / smoothl1 / scaled)
python analysis/compare_ablation_runs.py --log_dir ./logs
```

---

## Project Structure

```
BP_Project/
├── models/
│   ├── model.py               # BP_PINN (Bi-LSTM baseline)
│   └── model_variants.py      # BP_TCN, BP_HybridTransformer
├── training/
│   ├── train.py                           # original baseline script
│   ├── train_loss_combo_common.py         # shared loss combo trainer
│   ├── train_loss_combo.py                # entry point (--loss_mode)
│   ├── train_within_subject_common.py     # within-subject split trainer
│   ├── train_within_subject.py            # entry point
│   ├── train_ablation_common.py           # ablation trainer
│   ├── train_pp_mse.py / train_pp_smoothl1.py / train_pp_mse_scaled.py
│   ├── train_tcn.py / train_hybrid_transformer.py
│   └── train_map.py
├── analysis/
│   ├── compare_loss_combo_runs.py
│   ├── compare_ablation_runs.py
│   ├── compare_architecture_runs.py
│   ├── plot_results.py
│   └── visualize_sample.py
├── evaluation/
│   └── generate_csv.py
├── slurm/                     # all SBATCH submission scripts
├── data/                      # MIMIC-III_ppg_dataset.h5
├── checkpoints/               # saved model weights (.pth)
└── logs/                      # training logs and history JSONs
```

---

## Metrics

- **MAE** — mean absolute error (mmHg), primary metric
- **RMSE** — root mean squared error (mmHg)
- **R²** — coefficient of determination (higher = better; 1.0 is perfect)
