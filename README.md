\# ASTGCN 模型项目实现



本项目是论文 "Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting" 的一个 PyTorch 实现。



项目结构清晰，代码模块化，并使用配置文件进行参数管理。为了方便演示，本项目包含一个虚拟数据生成器，可以不依赖任何外部数据集直接运行。



\## 项目结构



```

ASTGCN\_PROJECT/

├── checkpoints/          # 存储模型检查点

├── src/                  # 源代码

├── config.yaml           # 全局配置文件

├── requirements.txt      # 依赖库

├── run.py                # 主运行脚本

└── README.md             # 本说明文件

```



\## 环境设置



1\.  克隆本项目。

2\.  安装所需的依赖库：

&nbsp;   ```bash

&nbsp;   pip install -r requirements.txt

&nbsp;   ```



\## 如何运行



1\.  \*\*配置参数\*\*: 打开 `config.yaml` 文件，您可以根据需要修改数据、模型和训练的各项参数。

2\.  \*\*开始训练\*\*: 直接运行 `run.py` 脚本即可开始训练。



&nbsp;   ```bash

&nbsp;   python run.py

&nbsp;   ```



&nbsp;   脚本会自动：

&nbsp;   - 加载 `config.yaml` 中的配置。

&nbsp;   - 生成虚拟数据。

&nbsp;   - 初始化 ASTGCN 模型。

&nbsp;   - 检查 `checkpoints/` 目录下是否有已保存的模型，如果有则加载并从上次中断的地方继续训练（断点续传）。

&nbsp;   - 执行训练循环，并实时打印损失。

&nbsp;   - 每个 epoch 结束后，将最新的模型状态保存到检查点文件中。



\## 主要功能



\- \*\*模块化实现\*\*: 模型的每个组件（注意力、卷积、时空块等）都作为独立的类实现，易于理解和修改。

\- \*\*配置驱动\*\*: 所有超参数都集中在 `config.yaml` 中，方便调参。

\- \*\*断点续传\*\*: 自动保存和加载检查点，如果训练中断，可以无缝恢复。

\- \*\*开箱即用\*\*: 内置虚拟数据生成器，无需下载数据集即可运行。

