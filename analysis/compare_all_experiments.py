"""
Compare all experiments in a table format.

Usage:
    python analysis/compare_all_experiments.py --log_dir ./logs
"""

import argparse
import glob
import os
import re

FINAL_RE = re.compile(
    r"FINAL TEST (SBP|DBP) -> MAE: ([0-9.]+), RMSE: ([0-9.]+), R2: ([0-9.\-]+)"
)

LOG_PATTERNS = {
    "cs_bilstm_mse":        ["cs_bilstm_mse_*.log",        "loss_mse_out_*.log"],
    "cs_bilstm_pp_map":     ["cs_bilstm_pp_map_*.log",     "loss_pp_map_out_*.log"],
    "cs_transformer_mse":   ["cs_transformer_mse_*.log"],
    "cs_transformer_pp_map":["cs_transformer_pp_map_*.log"],
    "ws_bilstm_mse":        ["ws_bilstm_mse_*.log"],
    "ws_bilstm_pp_map":     ["ws_bilstm_pp_map_*.log",     "ws_pp_map_out_*.log"],
    "ws_transformer_mse":       ["ws_transformer_mse_*.log"],
    "ws_transformer_pp_map":    ["ws_transformer_pp_map_*.log"],
    "cs_attn_bilstm_pp_map":    ["cs_attn_bilstm_pp_map_*.log"],
    "ws_attn_bilstm_pp_map":    ["ws_attn_bilstm_pp_map_*.log"],
    "cs_tcn_pp_map":            ["cs_tcn_pp_map_*.log"],
    "ws_tcn_pp_map":            ["ws_tcn_pp_map_*.log"],
}

MODELS  = ["bilstm", "transformer"]
SPLITS  = ["cs", "ws"]
LOSSES  = ["mse", "pp_map"]

MODEL_LABEL  = {"bilstm": "Bi-LSTM", "transformer": "Hybrid Transformer"}
SPLIT_LABEL  = {"cs": "Cross-subject", "ws": "Within-subject"}
LOSS_LABEL   = {"mse": "MSE only", "pp_map": "MSE+λ_pp·PP+λ_map·MAP"}


def parse_metrics(path):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    matches = FINAL_RE.findall(text)
    if len(matches) < 2:
        return None
    out = {}
    for metric, mae, rmse, r2 in matches[-2:]:
        out[f"{metric}_MAE"]  = float(mae)
        out[f"{metric}_RMSE"] = float(rmse)
        out[f"{metric}_R2"]   = float(r2)
    out["AVG_MAE"] = (out["SBP_MAE"] + out["DBP_MAE"]) / 2.0
    return out


def find_result(exp_id, log_dir):
    """Return (path, metrics) for the most recent completed run, or None."""
    patterns = LOG_PATTERNS.get(exp_id, [])
    candidates = []
    for pat in patterns:
        candidates.extend(sorted(glob.glob(os.path.join(log_dir, pat))))
    for path in reversed(candidates):
        m = parse_metrics(path)
        if m is not None:
            return path, m
    return None, None


def fmt(val, decimals=2):
    return f"{val:.{decimals}f}" if val is not None else "  —  "


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="./logs")
    args = parser.parse_args()

    print(f"\nScanning logs in: {os.path.abspath(args.log_dir)}\n")

    results = {}
    for model in MODELS:
        for split in SPLITS:
            for loss in LOSSES:
                exp_id = f"{split}_{model}_{loss}"
                path, metrics = find_result(exp_id, args.log_dir)
                results[exp_id] = (path, metrics)

    col_heads = [
        f"Cross-subject\n  {LOSS_LABEL['mse']}",
        f"Cross-subject\n  {LOSS_LABEL['pp_map']}",
        f"Within-subject\n  {LOSS_LABEL['mse']}",
        f"Within-subject\n  {LOSS_LABEL['pp_map']}",
    ]
    exp_order = [
        ("cs", "mse"), ("cs", "pp_map"), ("ws", "mse"), ("ws", "pp_map")
    ]

    metrics_to_show = [
        ("AVG MAE",  "AVG_MAE",  2),
        ("SBP MAE",  "SBP_MAE",  2),
        ("DBP MAE",  "DBP_MAE",  2),
        ("SBP RMSE", "SBP_RMSE", 2),
        ("DBP RMSE", "DBP_RMSE", 2),
        ("SBP R²",   "SBP_R2",   4),
        ("DBP R²",   "DBP_R2",   4),
    ]

    ROW_W  = 20
    COL_W  = 14

    header_line1 = f"{'':>{ROW_W}}  {'Cross-subject':^{COL_W*2+2}}  {'Within-subject':^{COL_W*2+2}}"
    header_line2 = (
        f"{'':>{ROW_W}}  "
        f"{'MSE only':>{COL_W}}  {'MSE+PP+MAP':>{COL_W}}  "
        f"{'MSE only':>{COL_W}}  {'MSE+PP+MAP':>{COL_W}}"
    )
    divider = "-" * len(header_line2)

    for model in MODELS:
        print(f"\n{'═'*len(header_line2)}")
        print(f"  {MODEL_LABEL[model]}")
        print(f"{'═'*len(header_line2)}")
        print(header_line1)
        print(header_line2)
        print(divider)

        for metric_label, metric_key, decimals in metrics_to_show:
            row = f"{metric_label:>{ROW_W}}  "
            for split, loss in exp_order:
                exp_id = f"{split}_{model}_{loss}"
                _, m = results[exp_id]
                val = m[metric_key] if m else None
                row += f"{fmt(val, decimals):>{COL_W}}  "
            print(row)

        print(divider)
        src_row = f"{'log':>{ROW_W}}  "
        for split, loss in exp_order:
            exp_id = f"{split}_{model}_{loss}"
            path, _ = results[exp_id]
            label = os.path.basename(path) if path else "(pending)"
            src_row += f"{label[:COL_W]:>{COL_W}}  "
        print(src_row)

    print()


if __name__ == "__main__":
    main()
