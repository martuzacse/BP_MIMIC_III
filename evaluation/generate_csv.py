import torch
import h5py
import numpy as np
import pandas as pd
import argparse
import os
from torch.utils.data import Dataset, DataLoader, Subset
from models.model import BP_PINN


class PPGTestDataset(Dataset):
    def __init__(self, path):
        self.data = h5py.File(path, 'r')
        self.ppg_data = self.data['ppg']
        self.label_data = self.data['label']
        self.subject_idx = self.data['subject_idx']
        
    def __len__(self): 
        return self.ppg_data.shape[0]
    
    def __getitem__(self, i):
        x = torch.from_numpy(self.ppg_data[i]).float().unsqueeze(0)
        y = torch.from_numpy(self.label_data[i]).float()
        sub = int(self.subject_idx[i])
        return x, y, sub


def generate_test_csv(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading best model on {device}...")
    
    full_dataset = PPGTestDataset(args.data_path)
    subjects = np.array(full_dataset.subject_idx).flatten()
    unique_subjects = np.unique(subjects)
    
    np.random.seed(42)
    np.random.shuffle(unique_subjects)
    
    num_sub = len(unique_subjects)
    test_subs = unique_subjects[int(0.85 * num_sub):]
    test_idx = np.where(np.isin(subjects, test_subs))[0]
    
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=512, shuffle=False, num_workers=4)
    print(f"Evaluating {len(test_idx)} samples from {len(test_subs)} unseen subjects...")

    model = BP_PINN().to(device)
    model.load_state_dict(torch.load("./checkpoints/model_best.pth", weights_only=True))
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for x, y, sub in test_loader:
            x = x.to(device)
            preds = model(x).cpu()
            
            for i in range(len(preds)):
                gt_s = y[i, 0].item()
                gt_d = y[i, 1].item()
                pr_s = preds[i, 0].item()
                pr_d = preds[i, 1].item()
                
                results.append({
                    'sub_id': sub[i].item(),
                    'gt_sbp': round(gt_s, 2),
                    'pred_sbp': round(pr_s, 2),
                    'error_sbp': round(abs(pr_s - gt_s), 2),
                    'gt_dbp': round(gt_d, 2),
                    'pred_dbp': round(pr_d, 2),
                    'error_dbp': round(abs(pr_d - gt_d), 2)
                })

    df = pd.DataFrame(results)
    csv_path = "./logs/test_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()
    
    generate_test_csv(args)
