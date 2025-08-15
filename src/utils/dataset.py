# src/utils/dataset.py
# --- CORRECTED VERSION ---

import torch
from torch.utils.data import Dataset
import numpy as np

class TrafficDataset(Dataset):
    def __init__(self, config):
        self.config = config
        filepath = config['dataset']['filepath']
        
        archive = np.load(filepath)
        self.data = archive['data']
        self.adj = torch.from_numpy(archive['adj']).float()
        
        self.mean = self.data.mean()
        self.std = self.data.std()
        self.data = (self.data - self.mean) / self.std
        
        self.Th = config['data']['Th']
        self.Td = config['data']['Td']
        self.Tw = config['data']['Tw']
        self.Tp = config['data']['Tp']
        self.q = config['dataset']['generation']['q']
        
        total_timesteps = self.data.shape[0]
        
        # Calculate the start index for valid samples.
        # This offset ensures we have enough history for all three components.
        # It must be the maximum of the required history lengths.
        max_daily_history = self.q * (self.Td // self.Tp)
        max_weekly_history = 7 * self.q * (self.Tw // self.Tp)
        
        self.start_offset = int(max(self.Th, max_daily_history, max_weekly_history))
        
        self.num_samples = total_timesteps - self.start_offset - self.Tp
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, index):
        # The current timestamp 't' marks the beginning of the prediction window
        t = index + self.start_offset
        
        # 1. Get Target Y (from t to t+Tp)
        y = self.data[t : t + self.Tp]
        y = y[:, :, 0] # Predict only the first feature (flow)
        
        # 2. Get Recent Input X_h (from t-Th to t)
        x_h = self.data[t - self.Th : t]

        # --- LOGIC FIX: Changed slicing from [end-Tp:end] to [start:start+Tp] ---
        
        # 3. Get Daily Input X_d
        # Find the start of the same time window on previous days
        daily_start_indices = [t - self.q * d for d in range(1, (self.Td // self.Tp) + 1)]
        x_d_list = [self.data[i : i + self.Tp] for i in reversed(daily_start_indices)]
        x_d = np.concatenate(x_d_list, axis=0)

        # 4. Get Weekly Input X_w
        # Find the start of the same time window on previous weeks
        weekly_start_indices = [t - 7 * self.q * w for w in range(1, (self.Tw // self.Tp) + 1)]
        x_w_list = [self.data[i : i + self.Tp] for i in reversed(weekly_start_indices)]
        x_w = np.concatenate(x_w_list, axis=0)
        
        # Convert to tensors
        x_h_tensor = torch.from_numpy(x_h).float()
        x_d_tensor = torch.from_numpy(x_d).float()
        x_w_tensor = torch.from_numpy(x_w).float()
        y_tensor = torch.from_numpy(y).float().permute(1, 0) # Shape: (N, Tp)
        
        return x_h_tensor, x_d_tensor, x_w_tensor, y_tensor