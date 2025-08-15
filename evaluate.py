# evaluate.py

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from src.models.astgcn import ASTGCN
from src.utils.dataset import TrafficDataset
from src.utils.checkpoint_manager import load_checkpoint

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    # Avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def main(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Test Dataset
    mode = config['dataset']['mode']
    if mode not in ['semi-real', 'metr-la']:
        print("Evaluation is only supported for 'semi-real' and 'metr-la' modes.")
        return
        
    test_dataset = TrafficDataset(config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    adj = test_dataset.adj.to(device)
    
    # Un-normalize function for metrics calculation
    mean = test_dataset.mean
    std = test_dataset.std

    # 2. Initialize Model
    model = ASTGCN(config).to(device)

    # 3. Load the BEST saved checkpoint
    # We pass optimizer=None because we are not training
    _, model, _, _ = load_checkpoint(
        model, optimizer=None, save_path=config['checkpoint']['path'], filename=config['checkpoint']['filename']
    )

    # 4. Evaluation Loop
    model.eval()
    all_preds = []
    all_labels = []

    print("\n--- Starting Evaluation on Test Set ---")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch_xh, batch_xd, batch_xw, batch_y = [b.to(device) for b in batch]
            
            # Get model predictions
            outputs = model(batch_xh, batch_xd, batch_xw, adj)
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
    
    # Concatenate all batches
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Un-normalize the data to get real-world metrics
    preds_unscaled = preds * std + mean
    labels_unscaled = labels * std + mean
    
    # 5. Calculate and Print Metrics for different prediction horizons
    print("\n--- Evaluation Results ---")
    for horizon_step in [2, 5, 11]: # Corresponds to 15min, 30min, 60min ahead if step is 5min
        horizon_index = horizon_step
        
        pred_horizon = preds_unscaled[:, :, horizon_index]
        label_horizon = labels_unscaled[:, :, horizon_index]
        
        mae = mean_absolute_error(label_horizon, pred_horizon)
        rmse = root_mean_squared_error(label_horizon, pred_horizon)
        mape = mean_absolute_percentage_error(label_horizon, pred_horizon)
        
        print(f"\nHorizon: { (horizon_step + 1) * 5 } minutes")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.4f}%")
        
if __name__ == '__main__':
    main()