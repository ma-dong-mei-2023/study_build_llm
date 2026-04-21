# study_build_llm —— 从零构建大语言模型学习笔记

本仓库记录我从零手写一个小规模 GPT 的完整过程。

## 路线图

| 章节 | 主题 | Notebook | 状态 |
|------|------|----------|------|
| 0 | 环境 + PyTorch 热身 | [notebooks/ch00_pytorch_warmup.ipynb](notebooks/ch00_pytorch_warmup.ipynb) | ✅ |
| 1 | Tokenization | [notebooks/ch01_tokenizer.ipynb](notebooks/ch01_tokenizer.ipynb) | ✅ |
| 2 | DataLoader + 嵌入 | [notebooks/ch02_dataloader.ipynb](notebooks/ch02_dataloader.ipynb) | ⬜ |
| 3 | 注意力机制 | [notebooks/ch03_attention.ipynb](notebooks/ch03_attention.ipynb) | ⬜ |
| 4 | GPT 模型架构 | [notebooks/ch04_gpt_model.ipynb](notebooks/ch04_gpt_model.ipynb) | ⬜ |
| 5 | 预训练 | [notebooks/ch05_pretrain.ipynb](notebooks/ch05_pretrain.ipynb) | ⬜ |
| 6 | (可选)加载 GPT-2 权重 | [notebooks/ch06_load_gpt2.ipynb](notebooks/ch06_load_gpt2.ipynb) | ⬜ |

## 目录结构

```
.
├── environment.yml       # conda 环境定义
├── notebooks/            # 每章的交互式学习笔记
├── llm/                  # 从 notebook 抽出的可复用模块
├── data/                 # 训练语料(不入版本库)
└── checkpoints/          # 模型权重(不入版本库)
```

## 快速开始

```bash
# 1. 创建并激活环境(第一次)
conda env create -f environment.yml
conda activate llm

# 2. 安装 GPU 版 PyTorch
#    打开 https://pytorch.org/get-started/locally/
#    选择 Stable / Windows / Pip / Python / CUDA 12.1(或与你 nvidia-smi 显示匹配的版本)
#    复制它给出的 pip install 命令粘贴运行,例如:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

# 4. 启动 Jupyter
jupyter lab
```

### 检查 CUDA 驱动版本

在安装 PyTorch 前先跑 `nvidia-smi` 确认显卡驱动支持的最高 CUDA 版本,在 PyTorch 官网选择 ≤ 该版本的 CUDA 构建即可。

## 参考资料

- Sebastian Raschka《Build a Large Language Model (From Scratch)》—— 主线教材,GitHub: rasbt/LLMs-from-scratch
- Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out"(YouTube)
- 论文:*Attention Is All You Need*(Vaswani et al., 2017)、*Language Models are Unsupervised Multitask Learners*(GPT-2)
