"""
LaTeX-ready learning curve figures.

Usage:
    python3 analysis/plot_results.py
    python3 analysis/plot_results.py --history ./logs/history.json --out ./logs/
"""

import json
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          12,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "figure.titlesize":   14,

    "lines.linewidth":    2.0,
    "lines.markersize":   5,

    "axes.linewidth":     1.0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.6,
    "grid.alpha":         0.5,
    "grid.color":         "#aaaaaa",

    "figure.facecolor":   "white",
    "axes.facecolor":     "white",

    "figure.constrained_layout.use": True,

    "pdf.fonttype":  42,
    "ps.fonttype":   42,
    "svg.fonttype": "none",
})

C_TRAIN = "#1a6faf"
C_SBP   = "#c0392b"
C_DBP   = "#27ae60"
C_LR    = "#8e44ad"

SINGLE_COL = (3.5, 2.8)
DOUBLE_COL = (7.16, 2.8)


def _save(fig, base_path: str):
    for ext in (".pdf", ".png"):
        path = base_path + ext
        dpi = 300 if ext == ".png" else None
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved → {path}")


def smooth(values, window=5):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def fig_training_loss(epochs, history, out_dir):
    raw  = np.array(history["train_loss"])
    smth = smooth(raw, window=5)

    fig, ax = plt.subplots(figsize=SINGLE_COL)

    ax.plot(epochs, raw,  color=C_TRAIN, linewidth=1.0, alpha=0.35)
    ax.plot(epochs, smth, color=C_TRAIN, linewidth=2.2,
            label="Train loss (MSE + PINN)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Training Convergence")
    ax.set_xlim(epochs[0], epochs[-1])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.legend(loc="upper right", framealpha=0.9)

    _save(fig, os.path.join(out_dir, "fig_training_loss"))
    plt.close(fig)


def fig_val_mae(epochs, history, out_dir):
    sbp = np.array(history["val_sbp_mae"])
    dbp = np.array(history["val_dbp_mae"])

    best_epoch = epochs[int(np.argmin(sbp))]
    best_sbp   = sbp.min()
    best_dbp   = dbp[int(np.argmin(sbp))]

    fig, ax = plt.subplots(figsize=SINGLE_COL)

    ax.plot(epochs, sbp, color=C_SBP, linewidth=2.2,
            marker="o", markevery=max(1, len(epochs)//10),
            label="SBP MAE (mmHg)")
    ax.plot(epochs, dbp, color=C_DBP, linewidth=2.2,
            marker="s", markevery=max(1, len(epochs)//10),
            label="DBP MAE (mmHg)")

    ax.axvline(best_epoch, color="#555555", linewidth=1.0,
               linestyle=":", zorder=0)
    ax.annotate(f"Best\nep. {best_epoch}",
                xy=(best_epoch, best_sbp),
                xytext=(best_epoch + max(1, len(epochs)*0.04), best_sbp * 1.05),
                fontsize=9, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (mmHg)")
    ax.set_title("(b) Validation MAE")
    ax.set_xlim(epochs[0], epochs[-1])
    ax.legend(loc="upper right", framealpha=0.9)

    fig.text(0.99, 0.01,
             f"Final: SBP {sbp[-1]:.2f} | DBP {dbp[-1]:.2f} mmHg",
             ha="right", va="bottom", fontsize=8, color="#666666",
             transform=fig.transFigure)

    _save(fig, os.path.join(out_dir, "fig_val_mae"))
    plt.close(fig)


def fig_combined(epochs, history, out_dir):
    raw  = np.array(history["train_loss"])
    smth = smooth(raw, window=5)
    sbp  = np.array(history["val_sbp_mae"])
    dbp  = np.array(history["val_dbp_mae"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE_COL)

    ax1.plot(epochs, raw,  color=C_TRAIN, linewidth=1.0, alpha=0.35)
    ax1.plot(epochs, smth, color=C_TRAIN, linewidth=2.2,
             label="Train loss (MSE + PINN)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("(a) Training Convergence")
    ax1.set_xlim(epochs[0], epochs[-1])
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax1.legend(loc="upper right", framealpha=0.9)

    ax2.plot(epochs, sbp, color=C_SBP, linewidth=2.2,
             marker="o", markevery=max(1, len(epochs)//10),
             label="SBP MAE")
    ax2.plot(epochs, dbp, color=C_DBP, linewidth=2.2,
             marker="s", markevery=max(1, len(epochs)//10),
             label="DBP MAE")
    best_epoch = epochs[int(np.argmin(sbp))]
    ax2.axvline(best_epoch, color="#555555", linewidth=1.0, linestyle=":", zorder=0)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (mmHg)")
    ax2.set_title("(b) Validation MAE")
    ax2.set_xlim(epochs[0], epochs[-1])
    ax2.legend(loc="upper right", framealpha=0.9)

    _save(fig, os.path.join(out_dir, "fig_learning_curves"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="./logs/history.json")
    parser.add_argument("--out",     default="./logs/")
    args = parser.parse_args()

    if not os.path.exists(args.history):
        raise FileNotFoundError(f"history.json not found at: {args.history}")

    with open(args.history) as f:
        history = json.load(f)

    os.makedirs(args.out, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    print("Generating figures ...")
    fig_training_loss(epochs, history, args.out)
    fig_val_mae(epochs, history, args.out)
    fig_combined(epochs, history, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
