import json
import os
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from models.model import BP_PINN
from models.model_variants import BP_HybridTransformer  # noqa: F401


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_metrics(y_pred, y_true):
    mae = torch.abs(y_pred - y_true).mean(dim=0)
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    ss_tot = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2, dim=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return mae, rmse, r2


class PPGDataset(Dataset):
    def __init__(self, path):
        self.data = h5py.File(path, "r")
        self.ppg_data = self.data["ppg"]
        self.label_data = self.data["label"]
        self.subject_idx = self.data["subject_idx"]

    def __len__(self):
        return self.ppg_data.shape[0]

    def __getitem__(self, i):
        x = torch.from_numpy(self.ppg_data[i]).float().unsqueeze(0)
        y = torch.from_numpy(self.label_data[i]).float()
        return x, y


def compute_map(bp_tensor):
    sbp = bp_tensor[:, 0]
    dbp = bp_tensor[:, 1]
    return (sbp + 2.0 * dbp) / 3.0


def _build_combo_loss(preds, y, criterion, loss_mode, pp_weight, map_weight):
    mse_loss = criterion(preds, y)

    pp_pred = preds[:, 0] - preds[:, 1]
    pp_true = y[:, 0] - y[:, 1]
    pp_loss = torch.mean((pp_pred - pp_true) ** 2)

    map_pred = compute_map(preds)
    map_true = compute_map(y)
    map_loss = torch.mean((map_pred - map_true) ** 2)

    if loss_mode == "mse":
        total_loss = mse_loss
    elif loss_mode == "pp":
        total_loss = mse_loss + pp_weight * pp_loss
    elif loss_mode == "map":
        total_loss = mse_loss + map_weight * map_loss
    elif loss_mode == "pp_map":
        total_loss = mse_loss + pp_weight * pp_loss + map_weight * map_loss
    elif loss_mode == "ordering":
        min_pp = 10.0
        ordering_loss = torch.mean(torch.relu(min_pp - pp_pred) ** 2)
        total_loss = mse_loss + pp_weight * ordering_loss
    else:
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

    return total_loss, mse_loss, pp_loss, map_loss


def train_loss_combo(args, loss_mode, experiment_name, model_cls=None):
    if model_cls is None:
        model_cls = BP_PINN
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting {experiment_name} on {device} ---")
    print(
        f"Config: experiment={experiment_name}, model={model_cls.__name__}, "
        f"split=cross_subject, seed={args.seed}, split_seed={args.split_seed}, "
        f"loss_mode={loss_mode}, pp_weight={args.pp_weight}, map_weight={args.map_weight}"
    )

    full_dataset = PPGDataset(args.data_path)
    subjects = np.array(full_dataset.subject_idx).flatten()
    unique_subjects = np.unique(subjects)

    np.random.seed(args.split_seed)
    np.random.shuffle(unique_subjects)

    num_sub = len(unique_subjects)
    train_subs = unique_subjects[: int(0.7 * num_sub)]
    val_subs = unique_subjects[int(0.7 * num_sub) : int(0.85 * num_sub)]
    test_subs = unique_subjects[int(0.85 * num_sub) :]

    train_idx = np.where(np.isin(subjects, train_subs))[0]
    val_idx = np.where(np.isin(subjects, val_subs))[0]
    test_idx = np.where(np.isin(subjects, test_subs))[0]

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
        epoch_total = 0.0
        epoch_mse = 0.0
        epoch_pp = 0.0
        epoch_map = 0.0

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
            epoch_mse += mse_loss.item()
            epoch_pp += pp_loss.item()
            epoch_map += map_loss.item()

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                val_preds.append(preds.cpu())
                val_trues.append(y.cpu())

        val_preds = torch.cat(val_preds)
        val_trues = torch.cat(val_trues)
        v_mae, v_rmse, v_r2 = get_metrics(val_preds, val_trues)

        avg_total = epoch_total / len(train_loader)
        avg_mse = epoch_mse / len(train_loader)
        avg_pp = epoch_pp / len(train_loader)
        avg_map = epoch_map / len(train_loader)
        avg_val_mae = (v_mae[0] + v_mae[1]) / 2.0

        print(f"\nEpoch [{epoch + 1}/{args.epochs}] | LR: {current_lr:.6f}")
        print(
            "  Train Losses -> "
            f"Total: {avg_total:.4f}, "
            f"MSE: {avg_mse:.4f}, "
            f"PP: {avg_pp:.4f}, "
            f"MAP: {avg_map:.4f}"
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
            preds = model(x)
            test_preds.append(preds.cpu())
            test_trues.append(y.cpu())

    test_preds = torch.cat(test_preds)
    test_trues = torch.cat(test_trues)
    t_mae, t_rmse, t_r2 = get_metrics(test_preds, test_trues)

    print(f"FINAL TEST SBP -> MAE: {t_mae[0]:.2f}, RMSE: {t_rmse[0]:.2f}, R2: {t_r2[0]:.4f}")
    print(f"FINAL TEST DBP -> MAE: {t_mae[1]:.2f}, RMSE: {t_rmse[1]:.2f}, R2: {t_r2[1]:.4f}")
    print(f"Artifacts: checkpoint={args.checkpoint_path}, history={args.history_path}")
