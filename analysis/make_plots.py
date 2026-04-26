"""
Generate publication-quality figures for BP estimation experiments.

Outputs saved to logs/figures/ as both PDF and PNG.

Usage:
    python analysis/make_plots.py --log_dir ./logs
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
    "legend.framealpha":  0.9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.6,
})

EXPERIMENTS = [
    ("cs", "bilstm",      "mse"),
    ("cs", "bilstm",      "pp_map"),
    ("cs", "attn_bilstm", "pp_map"),
    ("cs", "tcn",         "pp_map"),
    ("cs", "transformer", "mse"),
    ("cs", "transformer", "pp_map"),
    ("ws", "bilstm",      "mse"),
    ("ws", "bilstm",      "pp_map"),
    ("ws", "attn_bilstm", "pp_map"),
    ("ws", "tcn",         "pp_map"),
    ("ws", "transformer", "mse"),
    ("ws", "transformer", "pp_map"),
]

MODEL_LABEL = {
    "bilstm":      "Bi-LSTM",
    "attn_bilstm": "Attn-BiLSTM",
    "tcn":         "TCN",
    "transformer": "Hybrid Transformer",
}
MODEL_COLOR = {
    "bilstm":      "#1565C0",
    "attn_bilstm": "#E65100",
    "tcn":         "#2E7D32",
    "transformer": "#C62828",
}
MODEL_MARKER = {
    "bilstm":      "o",
    "attn_bilstm": "s",
    "tcn":         "^",
    "transformer": "D",
}
LOSS_STYLE = {"mse": (0, (4, 2)), "pp_map": "solid"}
LOSS_LABEL = {"mse": "MSE only", "pp_map": "PINN (MSE+PP+MAP)"}
SPLIT_LABEL = {"cs": "Cross-Subject", "ws": "Within-Subject"}

FINAL_RE = re.compile(
    r"FINAL TEST (SBP|DBP) -> MAE: ([0-9.]+), RMSE: ([0-9.]+), R2: ([0-9.\-]+)"
)


def load_history(log_dir):
    out = {}
    for split, model, loss in EXPERIMENTS:
        exp_id = f"{split}_{model}_{loss}"
        p = os.path.join(log_dir, f"history_{exp_id}.json")
        if os.path.exists(p):
            out[exp_id] = json.load(open(p))
    return out


def load_test(log_dir):
    out = {}
    for split, model, loss in EXPERIMENTS:
        exp_id = f"{split}_{model}_{loss}"
        for p in sorted(glob.glob(os.path.join(log_dir, f"{exp_id}_*.log"))):
            text = open(p).read()
            m = FINAL_RE.findall(text)
            if len(m) >= 2:
                r = {}
                for metric, mae, rmse, r2 in m[-2:]:
                    r[f"{metric}_MAE"]  = float(mae)
                    r[f"{metric}_RMSE"] = float(rmse)
                    r[f"{metric}_R2"]   = float(r2)
                r["AVG_MAE"] = (r["SBP_MAE"] + r["DBP_MAE"]) / 2.0
                out[exp_id] = r
                break
    return out


def save(fig, path_stem):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}")
    plt.close(fig)
    print(f"  saved {path_stem}.pdf/png")


def fig_training_curves(histories, out_dir):
    """Val avg MAE vs epoch for all models, CS and WS panels."""

    show = {
        "cs": [
            ("cs", "bilstm",      "pp_map"),
            ("cs", "attn_bilstm", "pp_map"),
            ("cs", "tcn",         "pp_map"),
            ("cs", "transformer", "mse"),
        ],
        "ws": [
            ("ws", "bilstm",      "pp_map"),
            ("ws", "attn_bilstm", "pp_map"),
            ("ws", "tcn",         "pp_map"),
            ("ws", "transformer", "mse"),
        ],
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, split in zip(axes, ["cs", "ws"]):
        for (sp, model, loss) in show[split]:
            exp_id = f"{sp}_{model}_{loss}"
            if exp_id not in histories:
                continue
            h = histories[exp_id]
            avg_val = [(a + b) / 2 for a, b in
                       zip(h["val_sbp_mae"], h["val_dbp_mae"])]
            epochs = list(range(1, len(avg_val) + 1))
            best_ep = int(np.argmin(avg_val)) + 1
            best_val = min(avg_val)

            ax.plot(epochs, avg_val,
                    color=MODEL_COLOR[model],
                    linestyle=LOSS_STYLE[loss],
                    linewidth=1.8,
                    marker=MODEL_MARKER[model],
                    markevery=max(1, len(epochs) // 8),
                    markersize=5,
                    label=MODEL_LABEL[model])
            ax.axvline(best_ep, color=MODEL_COLOR[model],
                       linestyle=":", linewidth=0.8, alpha=0.5)
            ax.scatter(best_ep, best_val,
                       color=MODEL_COLOR[model], zorder=5,
                       s=60, marker="*")

        ax.set_title(SPLIT_LABEL[split])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Avg MAE (mmHg)")
        ax.legend(loc="upper right")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle("Validation Learning Curves", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "fig1_training_curves"))


def fig_test_mae(test, out_dir):
    """Grouped bar chart of SBP/DBP MAE for each model."""

    models = ["bilstm", "attn_bilstm", "tcn", "transformer"]
    best_loss = {
        "bilstm":      "pp_map",
        "attn_bilstm": "pp_map",
        "tcn":         "pp_map",
        "transformer": "mse",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    bar_w = 0.35
    x = np.arange(len(models))

    for ax, split in zip(axes, ["cs", "ws"]):
        sbp_vals, dbp_vals = [], []
        for m in models:
            exp_id = f"{split}_{m}_{best_loss[m]}"
            r = test.get(exp_id, {})
            sbp_vals.append(r.get("SBP_MAE", 0))
            dbp_vals.append(r.get("DBP_MAE", 0))

        b1 = ax.bar(x - bar_w / 2, sbp_vals, bar_w,
                    color=[MODEL_COLOR[m] for m in models],
                    alpha=0.85, label="SBP MAE", edgecolor="white", linewidth=0.6)
        b2 = ax.bar(x + bar_w / 2, dbp_vals, bar_w,
                    color=[MODEL_COLOR[m] for m in models],
                    alpha=0.45, label="DBP MAE", edgecolor="white", linewidth=0.6,
                    hatch="///")

        for bar in b1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{bar.get_height():.1f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{bar.get_height():.1f}", ha="center", va="bottom",
                    fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in models], rotation=12, ha="right")
        ax.set_ylabel("MAE (mmHg)")
        ax.set_title(SPLIT_LABEL[split])
        ax.set_ylim(0, max(sbp_vals) * 1.22)

        from matplotlib.patches import Patch
        leg = [Patch(facecolor="grey", alpha=0.85, label="SBP MAE"),
               Patch(facecolor="grey", alpha=0.45, hatch="///", label="DBP MAE")]
        ax.legend(handles=leg, loc="upper right")

    fig.suptitle("Test MAE by Model and Split Strategy", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "fig2_test_mae"))


def fig_test_r2(test, out_dir):
    """Grouped bar chart of SBP/DBP R² for each model."""

    models = ["bilstm", "attn_bilstm", "tcn", "transformer"]
    best_loss = {
        "bilstm": "pp_map", "attn_bilstm": "pp_map",
        "tcn": "pp_map", "transformer": "mse",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    bar_w = 0.35
    x = np.arange(len(models))

    for ax, split in zip(axes, ["cs", "ws"]):
        sbp_r2, dbp_r2 = [], []
        for m in models:
            exp_id = f"{split}_{m}_{best_loss[m]}"
            r = test.get(exp_id, {})
            sbp_r2.append(r.get("SBP_R2", 0))
            dbp_r2.append(r.get("DBP_R2", 0))

        b1 = ax.bar(x - bar_w / 2, sbp_r2, bar_w,
                    color=[MODEL_COLOR[m] for m in models],
                    alpha=0.85, label="SBP R²", edgecolor="white", linewidth=0.6)
        b2 = ax.bar(x + bar_w / 2, dbp_r2, bar_w,
                    color=[MODEL_COLOR[m] for m in models],
                    alpha=0.45, label="DBP R²", edgecolor="white", linewidth=0.6,
                    hatch="///")

        for bar in b1:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(v, 0) + 0.01,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")
        for bar in b2:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(v, 0) + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL[m] for m in models], rotation=12, ha="right")
        ax.set_ylabel("R²")
        ax.set_title(SPLIT_LABEL[split])
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")

        from matplotlib.patches import Patch
        leg = [Patch(facecolor="grey", alpha=0.85, label="SBP R²"),
               Patch(facecolor="grey", alpha=0.45, hatch="///", label="DBP R²")]
        ax.legend(handles=leg, loc="upper left")

    fig.suptitle("Test R² by Model and Split Strategy", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "fig3_test_r2"))


def fig_pinn_loss_effect(test, histories, out_dir):
    """MSE vs PINN loss comparison for Bi-LSTM."""

    metrics = ["SBP_MAE", "DBP_MAE", "SBP_R2", "DBP_R2"]
    metric_label = {
        "SBP_MAE": "SBP MAE\n(mmHg)",
        "DBP_MAE": "DBP MAE\n(mmHg)",
        "SBP_R2":  "SBP R²",
        "DBP_R2":  "DBP R²",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    bar_w = 0.32
    x = np.arange(len(metrics))
    colors = {"mse": "#78909C", "pp_map": "#1565C0"}
    labels = {"mse": "MSE only", "pp_map": "PINN (MSE+PP+MAP)"}

    for ax, split in zip(axes, ["cs", "ws"]):
        for i, loss in enumerate(["mse", "pp_map"]):
            exp_id = f"{split}_bilstm_{loss}"
            r = test.get(exp_id, {})
            vals = [r.get(m, 0) for m in metrics]
            bars = ax.bar(x + (i - 0.5) * bar_w, vals, bar_w,
                          color=colors[loss], alpha=0.88,
                          label=labels[loss], edgecolor="white")
            for bar, m in zip(bars, metrics):
                v = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        max(v, 0) + 0.005 * (20 if "MAE" in m else 1),
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold" if loss == "pp_map" else "normal")

        ax.set_xticks(x)
        ax.set_xticklabels([metric_label[m] for m in metrics])
        ax.set_title(SPLIT_LABEL[split])
        ax.legend()
        ax.axhline(0, color="black", linewidth=0.6)

    fig.suptitle("PINN Loss Effect on Bi-LSTM Performance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "fig4_pinn_loss_effect"))


def fig_cs_vs_ws(test, out_dir):
    """Slope chart: AVG MAE from cross-subject to within-subject."""

    models_best = [
        ("bilstm",      "pp_map"),
        ("attn_bilstm", "pp_map"),
        ("tcn",         "pp_map"),
        ("transformer", "mse"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, metric_key, metric_name, invert in [
        (axes[0], "AVG_MAE",  "Avg MAE (mmHg)",  False),
        (axes[1], "SBP_R2",   "SBP R²",           True),
    ]:
        cs_vals, ws_vals, labels, colors_list = [], [], [], []
        for model, loss in models_best:
            cs_id = f"cs_{model}_{loss}"
            ws_id = f"ws_{model}_{loss}"
            cv = test.get(cs_id, {}).get(metric_key)
            wv = test.get(ws_id, {}).get(metric_key)
            if cv is None or wv is None:
                continue
            cs_vals.append(cv)
            ws_vals.append(wv)
            labels.append(MODEL_LABEL[model])
            colors_list.append(MODEL_COLOR[model])

        x_pos = [0, 1]
        for cv, wv, lbl, col in zip(cs_vals, ws_vals, labels, colors_list):
            ax.plot(x_pos, [cv, wv], "o-",
                    color=col, linewidth=2.2, markersize=8,
                    markerfacecolor=col, markeredgecolor="white",
                    markeredgewidth=1.2, label=lbl, zorder=3)
            ax.text(-0.07, cv, f"{cv:.2f}", ha="right", va="center",
                    fontsize=9, color=col, fontweight="bold")
            ax.text(1.07, wv, f"{wv:.2f}", ha="left", va="center",
                    fontsize=9, color=col, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Cross-Subject", "Within-Subject"], fontsize=11)
        ax.set_ylabel(metric_name)
        ax.set_xlim(-0.4, 1.4)
        ax.grid(axis="y")
        ax.legend(loc="center right" if not invert else "center left",
                  bbox_to_anchor=(1.0, 0.5) if not invert else (0.0, 0.5))
        ax.set_title(f"Impact of Split Strategy on {metric_name}")

    fig.suptitle("Cross-Subject vs Within-Subject Performance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "fig5_cs_vs_ws"))


def fig_train_loss(histories, out_dir):
    """Train MSE loss vs epoch for all models."""

    show = {
        "cs": [
            ("cs", "bilstm",      "pp_map"),
            ("cs", "attn_bilstm", "pp_map"),
            ("cs", "tcn",         "pp_map"),
            ("cs", "transformer", "mse"),
        ],
        "ws": [
            ("ws", "bilstm",      "pp_map"),
            ("ws", "attn_bilstm", "pp_map"),
            ("ws", "tcn",         "pp_map"),
            ("ws", "transformer", "mse"),
        ],
    }

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)

    for ax, split in zip(axes, ["cs", "ws"]):
        for (sp, model, loss) in show[split]:
            exp_id = f"{sp}_{model}_{loss}"
            if exp_id not in histories:
                continue
            h = histories[exp_id]
            train_mse = h["train_mse_loss"]
            epochs = list(range(1, len(train_mse) + 1))
            ax.plot(epochs, train_mse,
                    color=MODEL_COLOR[model],
                    linestyle=LOSS_STYLE[loss],
                    linewidth=1.8,
                    marker=MODEL_MARKER[model],
                    markevery=max(1, len(epochs) // 8),
                    markersize=5,
                    label=MODEL_LABEL[model])

        ax.set_ylabel("Train MSE Loss")
        ax.set_title(SPLIT_LABEL[split])
        ax.legend(loc="upper right")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlabel("Epoch")

    fig.suptitle("Training Loss Curves (MSE component)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "fig6_train_loss"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="./logs")
    args = parser.parse_args()

    out_dir = os.path.join(args.log_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading data ...")
    histories = load_history(args.log_dir)
    test      = load_test(args.log_dir)
    print(f"  {len(histories)} history files,  {len(test)} test results")

    print("\nGenerating figures ...")
    fig_training_curves(histories, out_dir)
    fig_test_mae(test, out_dir)
    fig_test_r2(test, out_dir)
    fig_pinn_loss_effect(test, histories, out_dir)
    fig_cs_vs_ws(test, out_dir)
    fig_train_loss(histories, out_dir)

    print(f"\nAll figures saved to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
