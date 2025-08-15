# src/utils/checkpoint_manager.py
# --- CORRECTED VERSION ---

import torch
import os

def save_checkpoint(state, save_path, filename):
    """
    Saves the model state, including the best validation loss.
    The state dictionary should contain:
    - 'epoch': current epoch
    - 'model_state_dict': model's state dictionary
    - 'optimizer_state_dict': optimizer's state dictionary
    - 'best_val_loss': the best validation loss achieved so far
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved with val_loss: {state['best_val_loss']:.4f} to {filepath}")

def load_checkpoint(model, optimizer, save_path, filename):
    """
    Loads a checkpoint for resuming training or evaluation.
    Returns:
        - start_epoch (int): The epoch to start training from.
        - model (nn.Module): The model with loaded weights.
        - optimizer (Optimizer): The optimizer with loaded state.
        - best_val_loss (float): The best validation loss from the previous session.
    """
    filepath = os.path.join(save_path, filename)
    if not os.path.exists(filepath):
        print("=> No checkpoint found, starting from scratch.")
        # Return initial values if no checkpoint exists
        return 0, model, optimizer, float('inf')

    # When loading for training, map to the correct device
    device = next(model.parameters()).device
    checkpoint = torch.load(filepath, map_location=device)
    
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Optimizer state is only needed for resuming training
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load the best validation loss to continue tracking correctly
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"=> Checkpoint loaded. Resuming from epoch {start_epoch} with best_val_loss: {best_val_loss:.4f}.")
    return start_epoch, model, optimizer, best_val_loss