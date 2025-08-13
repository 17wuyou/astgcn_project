# run.py

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import os

from src.models.astgcn import ASTGCN
from src.utils.data_loader import generate_dummy_data
from src.utils.checkpoint_manager import save_checkpoint, load_checkpoint

def main(config_path='config.yaml'):
    # 1. 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. 设置设备
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. 加载数据（使用虚拟数据）
    x_h, x_d, x_w, adj, labels = generate_dummy_data(config)
    x_h, x_d, x_w = x_h.to(device), x_d.to(device), x_w.to(device)
    adj, labels = adj.to(device), labels.to(device)
    
    # 4. 初始化模型、优化器和损失函数
    model = ASTGCN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()

    # 5. 加载检查点（如果存在），实现断点续传
    start_epoch, model, optimizer = load_checkpoint(
        model, optimizer, config['checkpoint']['path'], config['checkpoint']['filename']
    )

    # 6. 训练循环
    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        
        # 对于这个示例，我们只在一个批次上进行训练
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(x_h, x_d, x_w, adj)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}], Loss: {loss.item():.4f}")

        # 7. 保存检查点
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }
        save_checkpoint(state, config['checkpoint']['path'], config['checkpoint']['filename'])

    print("--- Training Finished ---")


if __name__ == '__main__':
    main()