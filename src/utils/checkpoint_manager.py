# src/utils/checkpoint_manager.py

import torch
import os

def save_checkpoint(state, save_path, filename):
    """保存检查点"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, save_path, filename):
    """加载检查点"""
    filepath = os.path.join(save_path, filename)
    if not os.path.exists(filepath):
        print("=> No checkpoint found, starting from scratch.")
        return 0, model, optimizer # 返回起始 epoch 0

    checkpoint = torch.load(filepath)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"=> Checkpoint loaded. Resuming training from epoch {start_epoch}.")
    return start_epoch, model, optimizer