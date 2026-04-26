# Blood Pressure Estimation from PPG (MIMIC-III)

Estimating systolic (SBP) and diastolic (DBP) blood pressure from raw photoplethysmography (PPG) signals using deep learning on the MIMIC-III waveform dataset.

---

## Model Architectures

The project explores several deep learning architectures for BP estimation:

*   **BP_PINN** (`models/model.py`): A hybrid architecture combining a Conv1D encoder for spatial features and a Bidirectional LSTM to capture temporal dynamics.
*   **BP_AttentionBiLSTM** (`models/model_variants.py`): An evolution of BP_PINN that integrates multi-head self-attention to prioritize critical phases of the PPG pulse.
*   **BP_TCN** (`models/model_variants.py`): A dilated residual Temporal Convolutional Network.
*   **BP_HybridTransformer** (`models/model_variants.py`): A combination of a convolutional stem and a Transformer encoder.

---

## Dataset

The experiments are conducted using the **MIMIC-III PPG dataset** (`data/MIMIC-III_ppg_dataset.h5`).

| Property | Value |
|----------|-------|
| Subjects | 4,527 |
| Samples per subject | 2,000 |
| Total samples | 9,054,000 |
| Input | Raw PPG waveform |
| Labels | [SBP, DBP] in mmHg |

**HDF5 Structure:** `ppg` (signal), `label` (SBP/DBP), `subject_idx` (subject identification).

---

## Experiments & Results

### 1. Loss Function Optimization
Investigated the impact of auxiliary loss terms (Pulse Pressure and Mean Arterial Pressure) alongside standard MSE.

| Loss Configuration | SBP MAE | DBP MAE | AVG MAE | SBP R² | DBP R² |
|--------------------|--------:|--------:|--------:|-------:|-------:|
| Baseline MSE       | 15.03   | 8.11    | 11.57   | 0.3544 | 0.2572 |
| MSE + λ_pp + λ_map | 15.02   | **8.08**| **11.55**| 0.3564 | **0.2610**|

### 2. Architecture Benchmarking
Compared the performance of different model architectures under consistent training conditions.

| Model | AVG MAE | SBP MAE | DBP MAE | SBP R² | DBP R² |
|-------|--------:|--------:|--------:|-------:|-------:|
| **BP_PINN (Bi-LSTM)** | **11.55** | **15.02** | **8.08** | **0.356** | **0.261** |
| BP_HybridTransformer | 14.71 | 19.71 | 9.71 | 0.004 | 0.002 |

### 3. Split Strategy Analysis
Analyzed the difference between cross-subject and within-subject data splitting.

| Strategy | SBP MAE | DBP MAE | AVG MAE | SBP R² | DBP R² |
|----------|--------:|--------:|--------:|-------:|-------:|
| Cross-subject (Unseen subjects) | 15.02 | 8.08 | 11.55 | 0.3564 | 0.2610 |
| **Within-subject**              | **12.82** | **7.27** | **10.05** | **0.5035** | **0.3977** |

---

## Usage Guide

### 1. Environment Setup
The project relies on standard deep learning libraries: `torch`, `h5py`, `numpy`, `pandas`, and `matplotlib`.

### 2. Training Models
All experiments are managed through a unified entry point:

```bash
# Example: Training a TCN model with Within-Subject split and auxiliary loss
python -m training.run_experiment \
    --data_path ./data/MIMIC-III_ppg_dataset.h5 \
    --model tcn \
    --loss_mode pp_map \
    --split within_subject
```

### 3. Slurm Integration
Batch submission scripts for cluster environments are available in the `slurm/` directory for each specific experimental configuration.

### 4. Analysis & Visualization
*   **Compare results:** `python analysis/compare_all_experiments.py` scans logs and generates a summary table.
*   **Generate plots:** `python analysis/make_plots.py` creates publication-quality performance visualizations.
*   **Visualize samples:** `python analysis/visualize_sample.py --index 0` renders individual PPG windows and their labels.

---

## Project Structure

*   `models/`: PyTorch implementations of all architectures.
*   `training/`: Unified training logic and split strategies.
*   `analysis/`: Scripts for performance comparison and visualization.
*   `evaluation/`: Inference scripts for generating detailed prediction CSVs.
*   `slurm/`: SBATCH scripts for cluster execution.
*   `logs/`: Training history, metrics, and generated figures.
*   `data/`: Directory for the HDF5 dataset.
