import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from training.train_loss_combo_common import (
    PPGDataset,
    _build_combo_loss,
    get_metrics,
    set_global_seed,
)
from models.model import BP_PINN
from models.model_variants import BP_HybridTransformer  # noqa: F401


def _within_subject_split(subjects, split_seed):
    """Split each subject's samples temporally into train/val/test."""
    unique_subjects = np.unique(subjects)

    rng = np.random.default_rng(split_seed)
    rng.shuffle(unique_subjects)

    train_idx, val_idx, test_idx = [], [], []
    for subj in unique_subjects:
        idx = np.where(subjects == subj)[0]
        n = len(idx)
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train : n_train + n_val])
        test_idx.append(idx[n_train + n_val :])

    return (
        np.concatenate(train_idx),
        np.concatenate(val_idx),
        np.concatenate(test_idx),
    )


def train_within_subject(args, loss_mode, experiment_name, model_cls=None):
    if model_cls is None:
        model_cls = BP_PINN
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- Starting {experiment_name} on {device} ---")
    print(
        f"Config: experiment={experiment_name}, model={model_cls.__name__}, "
        f"split=within_subject, seed={args.seed}, split_seed={args.split_seed}, "
        f"loss_mode={loss_mode}, pp_weight={args.pp_weight}, map_weight={args.map_weight}"
    )

    full_dataset = PPGDataset(args.data_path)
    subjects = np.array(full_dataset.subject_idx).flatten()

    train_idx, val_idx, test_idx = _within_subject_split(subjects, args.split_seed)

    print(
        f"Split sizes -> train: {len(train_idx):,}  "
        f"val: {len(val_idx):,}  test: {len(test_idx):,}"
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=generator,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    model = model_cls().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )

    history = {
        "experiment": experiment_name,
        "model_name": model_cls.__name__,
        "split_mode": "within_subject",
        "seed": args.seed,
        "split_seed": args.split_seed,
        "loss_mode": loss_mode,
        "pp_weight": args.pp_weight,
        "map_weight": args.map_weight,
        "train_loss": [],
        "train_mse_loss": [],
        "train_pp_loss": [],
        "train_map_loss": [],
        "val_sbp_mae": [],
        "val_dbp_mae": [],
        "val_sbp_r2": [],
        "val_dbp_r2": [],
        "learning_rate": [],
    }

    best_val_mae = float("inf")
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_total = epoch_mse = epoch_pp = epoch_map = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            preds = model(x)
            total_loss, mse_loss, pp_loss, map_loss = _build_combo_loss(
                preds=preds,
                y=y,
                criterion=criterion,
                loss_mode=loss_mode,
                pp_weight=args.pp_weight,
                map_weight=args.map_weight,
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_mse   += mse_loss.item()
            epoch_pp    += pp_loss.item()
            epoch_map   += map_loss.item()

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_preds.append(model(x).cpu())
                val_trues.append(y.cpu())

        val_preds = torch.cat(val_preds)
        val_trues = torch.cat(val_trues)
        v_mae, v_rmse, v_r2 = get_metrics(val_preds, val_trues)

        avg_total    = epoch_total / len(train_loader)
        avg_mse      = epoch_mse   / len(train_loader)
        avg_pp       = epoch_pp    / len(train_loader)
        avg_map      = epoch_map   / len(train_loader)
        avg_val_mae  = (v_mae[0] + v_mae[1]) / 2.0

        print(f"\nEpoch [{epoch + 1}/{args.epochs}] | LR: {current_lr:.6f}")
        print(
            f"  Train -> Total: {avg_total:.4f}, MSE: {avg_mse:.4f}, "
            f"PP: {avg_pp:.4f}, MAP: {avg_map:.4f}"
        )
        print(f"  VAL SBP -> MAE: {v_mae[0]:.2f}, RMSE: {v_rmse[0]:.2f}, R2: {v_r2[0]:.4f}")
        print(f"  VAL DBP -> MAE: {v_mae[1]:.2f}, RMSE: {v_rmse[1]:.2f}, R2: {v_r2[1]:.4f}")

        scheduler.step(avg_val_mae)

        history["train_loss"].append(avg_total)
        history["train_mse_loss"].append(avg_mse)
        history["train_pp_loss"].append(avg_pp)
        history["train_map_loss"].append(avg_map)
        history["learning_rate"].append(current_lr)
        history["val_sbp_mae"].append(v_mae[0].item())
        history["val_sbp_r2"].append(v_r2[0].item())
        history["val_dbp_mae"].append(v_mae[1].item())
        history["val_dbp_r2"].append(v_r2[1].item())

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"  [*] New best model saved! (Avg Val MAE: {best_val_mae:.2f})")
        else:
            epochs_no_improve += 1
            print(f"  [no improve {epochs_no_improve}/{args.patience}]")

        with open(args.history_path, "w") as f:
            json.dump(history, f)

        if epochs_no_improve >= args.patience:
            print(f"\n  Early stopping triggered at epoch {epoch + 1}.")
            break

    print("\n========================================================")
    print(f"Evaluating Best Model: {experiment_name}")
    print("========================================================")

    model.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))
    model.eval()

    test_preds, test_trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            test_preds.append(model(x).cpu())
            test_trues.append(y.cpu())

    test_preds = torch.cat(test_preds)
    test_trues = torch.cat(test_trues)
    t_mae, t_rmse, t_r2 = get_metrics(test_preds, test_trues)

    print(f"FINAL TEST SBP -> MAE: {t_mae[0]:.2f}, RMSE: {t_rmse[0]:.2f}, R2: {t_r2[0]:.4f}")
    print(f"FINAL TEST DBP -> MAE: {t_mae[1]:.2f}, RMSE: {t_rmse[1]:.2f}, R2: {t_r2[1]:.4f}")
    print(f"Artifacts: checkpoint={args.checkpoint_path}, history={args.history_path}")
