# run.py

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.models.astgcn import ASTGCN
# --- MODIFIED IMPORTS ---
from src.utils.data_loader import generate_dummy_data # For dummy mode
from src.utils.dataset import TrafficDataset         # For real mode
from src.utils.checkpoint_manager import save_checkpoint, load_checkpoint

def main(config_path='config.yaml'):
    # 1. Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- MODIFIED DATA LOADING ---
    if config['dataset']['mode'] == 'dummy':
        print("Using DUMMY data mode.")
        x_h, x_d, x_w, adj, labels = generate_dummy_data(config)
        adj = adj.to(device)
        # Wrap in a list to simulate a dataloader with one batch
        train_loader = [(
            x_h.to(device), x_d.to(device), x_w.to(device), labels.to(device)
        )]
    elif config['dataset']['mode'] == 'real':
        print("Using REAL data mode.")
        if not os.path.exists(config['dataset']['filepath']):
            raise FileNotFoundError("Dataset not found. Please run generate_dataset.py first.")
        
        dataset = TrafficDataset(config)
        train_loader = DataLoader(
            dataset, 
            batch_size=config['training']['batch_size'], 
            shuffle=True
        )
        adj = dataset.adj.to(device)
    else:
        raise ValueError("Invalid dataset mode specified in config.yaml. Use 'dummy' or 'real'.")
    # -----------------------------

    # 4. Initialize model, optimizer, loss
    model = ASTGCN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()

    # 5. Load checkpoint
    start_epoch, model, optimizer = load_checkpoint(
        model, optimizer, config['checkpoint']['path'], config['checkpoint']['filename']
    )

    # 6. Training loop
    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        total_loss = 0
        
        # Use tqdm for a nice progress bar with the real dataloader
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['training']['epochs']}]")
        for batch_xh, batch_xd, batch_xw, batch_y in progress_bar:
            # Move data to device inside the loop
            batch_xh, batch_xd, batch_xw, batch_y = \
                batch_xh.to(device), batch_xd.to(device), batch_xw.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_xh, batch_xd, batch_xw, adj)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Average Loss: {avg_loss:.4f}")

        # 7. Save checkpoint
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        save_checkpoint(state, config['checkpoint']['path'], config['checkpoint']['filename'])

    print("--- Training Finished ---")

if __name__ == '__main__':
    main()