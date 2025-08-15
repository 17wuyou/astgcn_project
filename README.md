# PyTorch 实现的 ASTGCN 用于交通流预测

本项目是论文 **《Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting》** 的一个完整、模块化的 PyTorch 实现。

代码库经过精心设计，旨在实现清晰的结构、简单的配置和可复现的实验流程。它支持三种不同的数据模式，从快速的管道测试到在真实世界基准数据集上的完整训练和评估。

## 核心特性

-   **模块化设计**: 模型的每个组件（注意力、图卷积、时空块等）都作为独立的类实现，易于理解、修改和扩展。
-   **配置驱动**: 所有的超参数、路径和模式设置都集中在 `config.yaml` 文件中，无需修改代码即可轻松进行实验。
-   **三种数据模式**:
    1.  **Dummy Mode**: 用于快速验证代码流程和调试的内存虚拟数据。
    2.  **Semi-Real Mode**: 通过脚本生成具有真实世界交通模式（如早晚高峰、周末效应）的数据集，用于在没有真实数据时进行更可靠的测试。
    3.  **Real-World Mode**: 支持在经典的 **METR-LA** 公开数据集上进行训练和评估。
-   **智能检查点**: 自动保存和加载检查点。模型只会在验证集损失 (`val_loss`) 创下新低时保存，确保 `best_model.pth` 中始终是全局最优的权重。支持断点续传。
-   **数据预处理**: 提供了独立的脚本来生成半真实数据集 (`generate_dataset.py`) 和处理原始的 METR-LA 数据集 (`prepare_metr_la.py`)。
-   **独立的评估脚本**: 使用 `evaluate.py` 在独立的测试集上评估最终训练好的模型的性能，并报告关键指标 (MAE, RMSE, MAPE)。

## 项目结构

```
ASTGCN_PROJECT/
├── checkpoints/              # 存储训练过程中最优的模型权重
│   └── best_model.pth
├── data/                     # 存储处理好的、可直接使用的数据集
│   └── traffic_data.npz
├── raw_data/                 # 存放原始的、未处理的数据集文件 (需手动下载)
│   ├── adj_mx.pkl
│   └── metr-la.h5
├── src/                      # 核心源代码
│   ├── models/
│   │   └── astgcn.py         # ASTGCN 所有模型类的定义
│   ├── utils/
│   │   ├── data_loader.py    # Dummy 模式的数据加载器
│   │   ├── dataset.py        # Semi-Real 和 METR-LA 模式的数据集类
│   │   └── checkpoint_manager.py # 保存和加载检查点的功能
│   └── __init__.py
├── config.yaml               # 集中管理所有超参数和配置
├── requirements.txt          # 项目依赖的 Python 包
├── generate_dataset.py       # 用于生成半真实数据集的脚本
├── prepare_metr_la.py        # 用于预处理 METR-LA 数据集的脚本
├── run.py                    # 项目的训练启动脚本
├── evaluate.py               # 项目的评估启动脚本
└── README.md                 # 本说明文件
```

## 环境设置

#### 1. 克隆项目

```bash
git clone https://github.com/17wuyou/astgcn_project.git
cd ASTGCN_PROJECT
```

#### 2. 安装依赖

建议使用虚拟环境（如 anaconda 或 venv）。

```bash
pip install -r requirements.txt
```

#### 3. 下载真实世界数据 (仅用于 METR-LA 模式)

这是使用 `metr-la` 模式的**必要前提**。

-   在项目根目录下创建一个名为 `raw_data` 的文件夹。
-   从以下链接下载 `metr-la.h5` 和 `adj_mx.pkl` 文件，并将它们放入 `raw_data` 文件夹中。
    -   **交通数据**: [metr-la.h5](https://github.com/liyaguang/DCRNN/raw/master/data/metr-la.h5)
    -   **邻接矩阵**: [adj_mx.pkl](https://github.com/liyaguang/DCRNN/raw/master/data/adj_mx.pkl)

## 工作流与使用指南

本项目的工作流核心是通过修改 `config.yaml` 文件中的 `dataset: mode:` 字段来切换不同的数据模式。

---

### 指南一：Dummy 模式 (用于快速测试)

此模式用于快速验证整个训练流程是否能无报错地运行，非常适合调试和开发。

#### **步骤 1: 配置模式**

打开 `config.yaml` 文件，将 `mode` 设置为 `"dummy"`。

```yaml
dataset:
  mode: "dummy"
  # ... 其他配置
```

#### **步骤 2: 开始训练**

在终端中运行 `run.py` 脚本。

```bash
python run.py
```

**预期输出**: 您将看到训练开始，并为每个 epoch 打印训练损失。由于此模式没有验证集，程序不会保存任何模型。

---

### 指南二：Semi-Real 模式 (使用生成的半真实数据集)

此模式让您可以在一个具有真实数据特性的数据集上训练模型，而无需下载任何真实数据。

#### **步骤 1: 生成数据集**

首先，运行数据生成脚本。

```bash
python generate_dataset.py
```

**预期输出**: 脚本会根据 `config.yaml` 中的 `semi-real` 配置生成数据，并将其保存到 `data/traffic_data.npz`。

#### **步骤 2: 配置模式**

打开 `config.yaml` 文件，将 `mode` 设置为 `"semi-real"`。

```yaml
dataset:
  mode: "semi-real"
  # ... 其他配置
```

#### **步骤 3: 开始训练**

在终端中运行 `run.py` 脚本。

```bash
python run.py
```

**预期输出**: 您将看到完整的训练和验证流程。当验证损失 (`Val Loss`) 创下新低时，模型权重将被保存到 `checkpoints/best_model.pth`。

---

### 指南三：Real-World 模式 (使用 METR-LA 数据集)

这是最核心的模式，用于在公开基准数据集上训练和评估模型。

#### **步骤 1: 准备数据**

**前提**: 确保您已完成“环境设置”中的第3步，即下载并放置了原始数据文件。

运行数据预处理脚本。

```bash
python prepare_metr_la.py
```

**预期输出**: 脚本会读取 `raw_data/` 目录下的文件，进行处理，并将最终的数据保存到 `data/traffic_data.npz` (这会覆盖掉之前由 `generate_dataset.py` 生成的文件)。

#### **步骤 2: 配置模式**

打开 `config.yaml` 文件，将 `mode` 设置为 `"metr-la"`。同时，请确保 `data: num_nodes:` 的值是 `207`，以匹配 METR-LA 数据集。

```yaml
dataset:
  mode: "metr-la"
  # ...
data:
  num_nodes: 207
  # ...
```

#### **步骤 3: 开始训练**

在终端中运行 `run.py` 脚本。

```bash
python run.py
```

**预期输出**: 与 Semi-Real 模式类似，模型将在 METR-LA 的训练集和验证集上进行训练，并保存性能最佳的模型。

## 评估最终模型

当您完成任意一种模式（`semi-real` 或 `metr-la`）的训练后，`checkpoints/best_model.pth` 文件中将保存着最优的模型。您可以运行评估脚本来测试它在**从未见过**的测试集上的性能。

#### **运行评估**

在终端中运行 `evaluate.py` 脚本。

```bash
python evaluate.py
```

**预期输出**: 脚本会自动加载 `best_model.pth`，在测试集上进行推理，并打印出模型在未来15分钟、30分钟、60分钟等不同预测时域上的 **MAE**, **RMSE**, 和 **MAPE** 指标。这些指标是衡量模型最终泛化能力的客观标准。
