"""
Unified entry point for running experiments.

Usage:
    python -m training.run_experiment \\
        --data_path /tmp/data.h5 \\
        --model tcn \\
        --loss_mode pp_map \\
        --split within_subject
"""

import argparse
import os

from models.model import BP_PINN
from models.model_variants import BP_AttentionBiLSTM, BP_HybridTransformer, BP_TCN
from training.train_loss_combo_common import train_loss_combo
from training.train_within_subject_common import train_within_subject

MODEL_CLS = {
    "bilstm":       BP_PINN,
    "attn_bilstm":  BP_AttentionBiLSTM,
    "transformer":  BP_HybridTransformer,
    "tcn":          BP_TCN,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one of the 8 experiment combinations."
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--model", type=str, required=True, choices=list(MODEL_CLS),
        help="bilstm | attn_bilstm | transformer | tcn",
    )
    parser.add_argument(
        "--loss_mode", type=str, required=True, choices=["mse", "pp_map", "ordering"],
        help="mse | pp_map | ordering",
    )
    parser.add_argument(
        "--split", type=str, required=True,
        choices=["cross_subject", "within_subject"],
        help="cross_subject | within_subject",
    )
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--split_seed", type=int,   default=42)
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--patience",   type=int,   default=15)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--pp_weight",  type=float, default=0.1)
    parser.add_argument("--map_weight", type=float, default=0.05)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--history_path",    type=str, default=None)

    args = parser.parse_args()

    split_short  = "cs" if args.split == "cross_subject" else "ws"
    args.exp_id  = f"{split_short}_{args.model}_{args.loss_mode}"

    if args.checkpoint_path is None:
        args.checkpoint_path = f"./checkpoints/model_best_{args.exp_id}.pth"
    if args.history_path is None:
        args.history_path = f"./logs/history_{args.exp_id}.json"

    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.history_path),    exist_ok=True)
    return args


if __name__ == "__main__":
    args      = parse_args()
    model_cls = MODEL_CLS[args.model]

    if args.split == "cross_subject":
        train_loss_combo(
            args=args,
            loss_mode=args.loss_mode,
            experiment_name=args.exp_id,
            model_cls=model_cls,
        )
    else:
        train_within_subject(
            args=args,
            loss_mode=args.loss_mode,
            experiment_name=args.exp_id,
            model_cls=model_cls,
        )
