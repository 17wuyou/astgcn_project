# prepare_metr_la.py

import numpy as np
import pandas as pd
import pickle
import yaml
import os

def load_metr_la_data(config):
    """
    Loads and preprocesses the METR-LA dataset.
    
    Returns:
        - A tuple of (data, adj_matrix)
    """
    # 1. Load Adjacency Matrix
    adj_path = config['dataset']['metr-la']['adj_path']
    try:
        with open(adj_path, 'rb') as f:
            # The pkl file contains sensor_ids, sensor_id_to_ind, adj_mx
            pickle_data = pickle.load(f, encoding='latin1')
            adj_mx = pickle_data[2]
            print("Adjacency matrix loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Adjacency matrix file not found at {adj_path}")
        return None, None

    # 2. Load Traffic Data
    data_path = config['dataset']['metr-la']['data_path']
    try:
        df = pd.read_hdf(data_path)
        data = df.values
        # The raw data has shape (T, N). We need to add a feature dimension.
        # Often, multiple features are engineered. For simplicity, we use the speed 
        # as the primary feature and add two dummy features to match the config.
        # In a real application, you would use all engineered features.
        num_timesteps, num_nodes = data.shape
        num_features = config['data']['num_features']
        
        expanded_data = np.zeros((num_timesteps, num_nodes, num_features))
        expanded_data[:, :, 0] = data # Speed is the first feature
        # Fill other features with zeros or other engineered values if available
        # This ensures compatibility with the model's expected input shape
        if num_features > 1:
            print(f"Warning: Raw data has 1 feature but model expects {num_features}. Filling others with copies/zeros.")
            for i in range(1, num_features):
                expanded_data[:, :, i] = data # Simply copying for demonstration
                
        print("Traffic data loaded and expanded successfully.")
        return expanded_data, adj_mx

    except FileNotFoundError:
        print(f"Error: Traffic data file not found at {data_path}")
        return None, None

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    print("--- Starting METR-LA Data Preparation ---")
    data, adj = load_metr_la_data(config)
    
    if data is not None and adj is not None:
        filepath = config['dataset']['filepath']
        dir_path = os.path.dirname(filepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        print(f"Saving processed data to {filepath}...")
        # Note: We are now using the same output file as the semi-real data generator.
        # This is intentional. The active dataset is the one most recently generated/prepared.
        np.savez_compressed(filepath, data=data, adj=adj)
        print("Save complete. You can now run the training with 'metr-la' mode.")
    else:
        print("Data preparation failed.")

if __name__ == '__main__':
    main()