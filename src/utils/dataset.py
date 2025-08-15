# src/utils/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np

class TrafficDataset(Dataset):
    def __init__(self, config):
        self.config = config
        filepath = config['dataset']['filepath']
        
        # Load data from file
        archive = np.load(filepath)
        self.data = archive['data'] # Shape (T, N, F)
        self.adj = torch.from_numpy(archive['adj']).float()
        
        # Normalize data (Min-Max Scaler)
        self.mean = self.data.mean()
        self.std = self.data.std()
        self.data = (self.data - self.mean) / self.std
        
        # Get config values
        self.Th = config['data']['Th']
        self.Td = config['data']['Td']
        self.Tw = config['data']['Tw']
        self.Tp = config['data']['Tp']
        self.q = config['dataset']['generation']['q']
        
        total_timesteps = self.data.shape[0]
        
        # Calculate the start index for valid samples
        # We need enough history for the weekly component
        self.start_offset = (self.Tw // self.Tp) * 7 * self.q
        
        # Calculate the number of valid samples
        self.num_samples = total_timesteps - self.start_offset - self.Tp
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, index):
        t = index + self.start_offset # Current timestamp for prediction start
        
        # 1. Get Target Y
        y = self.data[t : t + self.Tp] # Shape: (Tp, N, F)
        # We only predict the first feature (traffic flow)
        y = y[:, :, 0] # Shape: (Tp, N)
        
        # 2. Get Recent Input X_h
        x_h = self.data[t - self.Th : t] # Shape: (Th, N, F)

        # 3. Get Daily Input X_d
        daily_indices = [t - self.q * d for d in range(1, (self.Td // self.Tp) + 1)]
        x_d_list = [self.data[i - self.Tp : i] for i in reversed(daily_indices)]
        x_d = np.concatenate(x_d_list, axis=0) # Shape: (Td, N, F)

        # 4. Get Weekly Input X_w
        weekly_indices = [t - 7 * self.q * w for w in range(1, (self.Tw // self.Tp) + 1)]
        x_w_list = [self.data[i - self.Tp : i] for i in reversed(weekly_indices)]
        x_w = np.concatenate(x_w_list, axis=0) # Shape: (Tw, N, F)
        
        # Convert to tensors
        x_h_tensor = torch.from_numpy(x_h).float()
        x_d_tensor = torch.from_numpy(x_d).float()
        x_w_tensor = torch.from_numpy(x_w).float()
        y_tensor = torch.from_numpy(y).float().permute(1, 0) # Shape: (N, Tp)
        
        return x_h_tensor, x_d_tensor, x_w_tensor, y_tensor