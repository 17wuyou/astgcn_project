# src/utils/data_loader.py

import torch
import numpy as np

def generate_dummy_data(config):
    """生成符合配置的虚拟数据，用于演示和测试"""
    batch_size = config['training']['batch_size']
    num_nodes = config['data']['num_nodes']
    num_features = config['data']['num_features']
    Th, Td, Tw = config['data']['Th'], config['data']['Td'], config['data']['Tw']
    Tp = config['data']['Tp']

    # 生成随机输入数据
    x_h = torch.randn(batch_size, Th, num_nodes, num_features)
    x_d = torch.randn(batch_size, Td, num_nodes, num_features)
    x_w = torch.randn(batch_size, Tw, num_nodes, num_features)
    
    # 生成随机邻接矩阵 (实际应用中应从文件加载)
    # 这里创建一个简单的随机对称矩阵
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj + adj.T) / 2 # 确保对称
    adj[adj > 0.1] = 1 # 二值化
    adj[adj <= 0.1] = 0
    adj.fill_diagonal_(1) # 确保自环

    # 生成随机标签（预测目标）
    labels = torch.randn(batch_size, num_nodes, Tp)
    
    print("--- Dummy Data Generated ---")
    print(f"Shape of x_h: {x_h.shape}")
    print(f"Shape of Adj: {adj.shape}")
    print(f"Shape of labels: {labels.shape}")
    print("----------------------------")

    return x_h, x_d, x_w, adj, labels