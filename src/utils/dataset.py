# src/utils/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np

class TrafficDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        filepath = config['dataset']['filepath']
        
        archive = np.load(filepath)
        self.data = archive['data']
        self.adj = torch.from_numpy(archive['adj']).float()
        
        # --- Data Splitting ---
        total_timesteps = self.data.shape[0]
        train_size = int(total_timesteps * config['dataset']['metr-la']['train_split'])
        val_size = int(total_timesteps * config['dataset']['metr-la']['val_split'])
        
        if split == 'train':
            self.data = self.data[:train_size]
        elif split == 'val':
            self.data = self.data[train_size : train_size + val_size]
        elif split == 'test':
            self.data = self.data[train_size + val_size :]
        else:
            raise ValueError("Invalid split specified. Use 'train', 'val', or 'test'.")

        # --- Normalization ---
        # Calculate mean and std ONLY on the training set
        # For val/test sets, we use the scaler from the training set
        train_data = archive['data'][:train_size]
        self.mean = train_data.mean()
        self.std = train_data.std()
        self.data = (self.data - self.mean) / self.std
        
        self.Th = config['data']['Th']
        self.Td = config['data']['Td']
        self.Tw = config['data']['Tw']
        self.Tp = config['data']['Tp']
        
        # For METR-LA, q (samples per day) is 288 (24 * 60 / 5)
        self.q = 288 
        
        max_weekly_history = 7 * self.q * (self.Tw // self.Tp)
        self.start_offset = int(max(self.Th, max_weekly_history))
        
        self.num_samples = len(self.data) - self.start_offset - self.Tp
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, index):
        t = index + self.start_offset
        
        y = self.data[t : t + self.Tp, :, 0]
        x_h = self.data[t - self.Th : t]

        daily_start_indices = [t - self.q * d for d in range(1, (self.Td // self.Tp) + 1)]
        x_d_list = [self.data[i : i + self.Tp] for i in reversed(daily_start_indices)]
        x_d = np.concatenate(x_d_list, axis=0)

        weekly_start_indices = [t - 7 * self.q * w for w in range(1, (self.Tw // self.Tp) + 1)]
        x_w_list = [self.data[i : i + self.Tp] for i in reversed(weekly_start_indices)]
        x_w = np.concatenate(x_w_list, axis=0)
        
        x_h_tensor = torch.from_numpy(x_h).float()
        x_d_tensor = torch.from_numpy(x_d).float()
        x_w_tensor = torch.from_numpy(x_w).float()
        y_tensor = torch.from_numpy(y).float().permute(1, 0)
        
        return x_h_tensor, x_d_tensor, x_w_tensor, y_tensor