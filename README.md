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

## Experimental Results

### 1. Cross-Subject Performance
In the cross-subject split, the model is tested on subjects it has never seen during training, evaluating its generalization capability.

| Model | Loss | AVG MAE | SBP MAE | DBP MAE | SBP RMSE | DBP RMSE | SBP R² | DBP R² |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Bi-LSTM | MSE | 11.57 | 15.05 | 8.09 | 19.43 | 10.62 | 0.355 | 0.254 |
| Bi-LSTM | pp_map | 11.55 | 14.96 | 8.13 | 19.33 | 10.61 | 0.361 | 0.256 |
| Attn-BiLSTM | pp_map | 11.40 | 14.76 | 8.04 | 19.22 | 10.53 | 0.369 | 0.267 |
| **TCN** | **pp_map** | **11.29** | **14.64** | **7.94** | **19.26** | **10.53** | **0.366** | **0.267** |
| Hybrid Transformer | MSE | 14.58 | 19.47 | 9.70 | 23.95 | 12.27 | 0.020 | 0.005 |
| Hybrid Transformer | pp_map | 14.74 | 19.76 | 9.72 | 24.19 | 12.30 | -0.000 | -0.000 |

### 2. Within-Subject Performance
In the within-subject split, samples from every subject are present in both training and testing sets, evaluating the model's ability to learn personalized features.

| Model | Loss | AVG MAE | SBP MAE | DBP MAE | SBP RMSE | DBP RMSE | SBP R² | DBP R² |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Bi-LSTM | MSE | 10.12 | 12.94 | 7.29 | 17.15 | 9.73 | 0.496 | 0.395 |
| Bi-LSTM | pp_map | 10.06 | 12.85 | 7.28 | 17.01 | 9.70 | 0.505 | 0.399 |
| Attn-BiLSTM | pp_map | 9.60 | 12.18 | 7.01 | 16.33 | 9.39 | 0.544 | 0.437 |
| **TCN** | **pp_map** | **9.07** | **11.58** | **6.55** | **15.70** | **8.95** | **0.578** | **0.488** |
| Hybrid Transformer | MSE | 10.62 | 13.77 | 7.48 | 18.47 | 10.11 | 0.416 | 0.347 |
| Hybrid Transformer | pp_map | 10.64 | 13.71 | 7.58 | 18.41 | 10.22 | 0.420 | 0.333 |

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
*   **Compare results:** `python analysis/compare_all_experiments.py` scans logs and generates tables.
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
