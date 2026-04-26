import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os

DEFAULT_H5   = "./data/MIMIC-III_ppg_dataset.h5"
DEFAULT_OUT  = "./logs/sample_visualization"
PPG_FS       = 125

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.constrained_layout.use": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
})

C_PPG   = "#1a6faf"
C_SBP   = "#c0392b"
C_DBP   = "#27ae60"
C_GRID  = "#cccccc"

def load_sample(h5_path: str, index: int):
    with h5py.File(h5_path, "r") as f:
        ppg = f["ppg"][index]
        label = f["label"][index]
        sub_id = int(f["subject_idx"][index])
    ppg = np.array(ppg).squeeze()
    return ppg, float(label[0]), float(label[1]), sub_id

def plot_sample(ppg: np.ndarray, sbp: float, dbp: float, sub_id: int, index: int, out_base: str):
    seq_len = len(ppg)
    time = np.arange(seq_len) / PPG_FS
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(time, ppg, color=C_PPG, linewidth=2.0, label="PPG Signal")
    ax.fill_between(time, ppg, ppg.min(), color=C_PPG, alpha=0.1)

    ax.axhline(ppg.max(), color=C_SBP, linewidth=1.2, linestyle="--", alpha=0.6)
    ax.axhline(ppg.min(), color=C_DBP, linewidth=1.2, linestyle="--", alpha=0.6)

    ppg_patch = mpatches.Patch(color=C_PPG, label="PPG waveform")
    sbp_patch = mpatches.Patch(color=C_SBP, label=f"SBP: {sbp:.1f} mmHg")
    dbp_patch = mpatches.Patch(color=C_DBP, label=f"DBP: {dbp:.1f} mmHg")
    
    ax.legend(handles=[ppg_patch, sbp_patch, dbp_patch], 
              loc="upper right", frameon=True, framealpha=0.9, fontsize=10)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title(f"Sample Index: {index} | Subject: {sub_id}")
    
    stats_text = (f"SBP: {sbp:.1f}\nDBP: {dbp:.1f}\nPP: {sbp-dbp:.1f}")
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=C_GRID))

    os.makedirs(os.path.dirname(os.path.abspath(out_base)), exist_ok=True)
    for ext in [".pdf", ".png"]:
        path = out_base + ext
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=DEFAULT_H5)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    ppg, sbp, dbp, sub_id = load_sample(args.data_path, args.index)
    plot_sample(ppg, sbp, dbp, sub_id, args.index, args.out)

if __name__ == "__main__":
    main()
