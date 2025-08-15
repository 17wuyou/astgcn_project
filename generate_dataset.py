# generate_dataset.py
# --- CORRECTED VERSION ---

import numpy as np
import yaml
import os

def generate_semi_real_data(config):
    """
    Generates a semi-realistic traffic dataset with daily and weekly patterns.
    """
    # --- FIX: Read from the correct config key 'semi-real' instead of 'generation' ---
    gen_config = config['dataset']['semi-real'] 
    data_config = config['data']
    
    num_days = gen_config['num_days']
    q = gen_config['q']
    num_nodes = data_config['num_nodes']
    num_features = data_config['num_features']
    
    total_timesteps = num_days * q
    
    print(f"Generating dataset with {total_timesteps} timesteps for {num_nodes} nodes...")

    # Time axis
    time = np.arange(total_timesteps) / q # In units of days

    # --- Create Patterns ---
    # 1. Daily pattern (two rush hours)
    daily_pattern = (np.sin(2 * np.pi * time) + 0.8 * np.sin(4 * np.pi * time)) * 0.5 + 0.5
    
    # 2. Weekly pattern (weekends are quieter)
    day_of_week = (time * q) // q % 7
    weekly_pattern = np.ones_like(time)
    weekly_pattern[day_of_week >= 5] = 0.6 # Make weekends 60% of weekday traffic

    # 3. Base noise
    noise = np.random.randn(total_timesteps, num_nodes, num_features) * 0.1

    # --- Combine Patterns for each feature ---
    # We create a base time series and then apply it to all nodes with slight variations.
    # Shape of patterns: (total_timesteps, 1, 1) to allow broadcasting
    daily_pattern = daily_pattern[:, np.newaxis, np.newaxis]
    weekly_pattern = weekly_pattern[:, np.newaxis, np.newaxis]

    # Feature 1: Traffic Flow (strongly cyclical)
    base_flow = 500 + 400 * daily_pattern * weekly_pattern
    
    # Feature 2: Speed (inversely related to flow)
    base_speed = 80 - 40 * daily_pattern * weekly_pattern
    
    # Feature 3: Occupancy (related to flow)
    base_occupancy = 0.3 + 0.5 * daily_pattern * weekly_pattern
    
    # Combine bases into a single tensor
    base_features = np.concatenate([base_flow, base_speed, base_occupancy], axis=-1)

    # Final data is the combination of base patterns and noise
    final_data = base_features + noise
    
    # Ensure no negative values
    final_data = np.maximum(final_data, 0)
    
    # --- Generate Adjacency Matrix ---
    adj = np.random.rand(num_nodes, num_nodes)
    adj = (adj + adj.T) / 2
    adj[adj > 0.1] = 1
    adj[adj <= 0.1] = 0
    np.fill_diagonal(adj, 1)

    print("Dataset generation complete.")
    print(f"Final data shape: {final_data.shape}") # (T, N, F)
    print(f"Adjacency matrix shape: {adj.shape}")

    return final_data, adj

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    data, adj = generate_semi_real_data(config)
    
    # We will save to the main filepath defined in the config
    filepath = config['dataset']['filepath']
    dir_path = os.path.dirname(filepath)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    print(f"Saving dataset to {filepath}...")
    np.savez_compressed(filepath, data=data, adj=adj)
    print("Save complete.")

if __name__ == '__main__':
    main()