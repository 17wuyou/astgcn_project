# run.py

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

from src.models.astgcn import ASTGCN
from src.utils.data_loader import generate_dummy_data
from src.utils.dataset import TrafficDataset
from src.utils.checkpoint_manager import save_checkpoint, load_checkpoint

def main(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- UPDATED DATA LOADING LOGIC ---
    mode = config['dataset']['mode']
    if mode == 'dummy':
        print("Using DUMMY data mode.")
        x_h, x_d, x_w, adj, labels = generate_dummy_data(config)
        adj = adj.to(device)
        train_loader = [(x_h.to(device), x_d.to(device), x_w.to(device), labels.to(device))]
        val_loader = None # No validation in dummy mode
    elif mode in ['semi-real', 'metr-la']:
        print(f"Using {mode.upper()} data mode.")
        filepath = config['dataset']['filepath']
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}. Please run generate_dataset.py or prepare_metr_la.py first.")
        
        train_dataset = TrafficDataset(config, split='train')
        val_dataset = TrafficDataset(config, split='val')
        
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        
        adj = train_dataset.adj.to(device)
    else:
        raise ValueError("Invalid dataset mode specified. Use 'dummy', 'semi-real', or 'metr-la'.")
    # ------------------------------------
    
    model = ASTGCN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()

    # Load checkpoint logic might need adjustment if you want to resume from best model
    # For simplicity, we start fresh or resume from the last epoch
    start_epoch = 0 
    best_val_loss = float('inf')

    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, config['training']['epochs']):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch in train_bar:
            batch_xh, batch_xd, batch_xw, batch_y = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(batch_xh, batch_xd, batch_xw, adj)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Phase ---
        if val_loader is None:
            print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Train Loss: {avg_train_loss:.4f}")
            continue # Skip validation for dummy mode

        model.eval()
        total_val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        with torch.no_grad():
            for batch in val_bar:
                batch_xh, batch_xd, batch_xw, batch_y = [b.to(device) for b in batch]
                outputs = model(batch_xh, batch_xd, batch_xw, adj)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss
            }
            save_checkpoint(state, config['checkpoint']['path'], config['checkpoint']['filename'])

    print("--- Training Finished ---")

if __name__ == '__main__':
    main()