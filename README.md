# Diy-LLM

一个围绕 Stanford CS336 课程开展的动手学习仓库。从分词器、Transformer 基础组件和训练工具开始，逐步进入 FlashAttention、数据并行、参数分片和优化器状态分片等大模型系统主题。仓库同时保留实验记录、实现过程中的错误复盘，以及用于理解概念的独立练习。

> 这是学习过程仓库：代码、实验和笔记会持续迭代；其中 Assignment 2 仍有待完成的实现。

## 学习内容

- **Tokenizer 与数据**：Unicode 切分、字符/字节级 tokenizer、BPE 训练与增量更新、压缩率和编码吞吐实验。
- **Transformer 基础**：Linear、Embedding、RMSNorm、SiLU/SwiGLU、因果注意力、RoPE、多头注意力、Transformer Block 与语言模型。
- **训练工具**：交叉熵的数值稳定实现、批采样、梯度裁剪、AdamW、warmup + cosine 学习率、checkpoint。
- **训练与系统**：FlashAttention-2 的 online softmax 推导与 PyTorch 参考实现；DDP、FSDP、ZeRO 风格 optimizer state sharding 的作业骨架与测试。
- **工程分析**：模型显存与训练时间估算、GPU benchmark、MoE 路由练习。

## 项目结构

```text
.
├── assignment1/
│   └── assignment1-basics/       # CS336 A1：模型与训练基础
│       ├── cs336_basics/         # 自己实现的 BPE、模型层与训练工具
│       ├── experiments/          # 实验计划、冻结条件、运行脚本和结果 CSV
│       ├── blog_assets/          # 实验曲线图
│       ├── CS336-A1-实验复盘.md   # A1 实验复盘
│       └── EXPERIMENT_LOG.md     # 实验过程记录
├── assignment2_system/
│   └── assignment2-systems/      # CS336 A2：系统与分布式训练
│       ├── cs336_systems/        # FlashAttention、DDP、FSDP、Sharded Optimizer
│       ├── cs336-basics/         # A1 参考基础模块
│       └── flashattention_visualizer.html
├── exercises/                    # 独立概念与编程练习
│   ├── 分词器.py
│   ├── MoE.py
│   ├── GPU编程.py
│   └── 资源核算.py
└── notes/                        # 学习笔记
    └── CS336-A1-模型层手撕与踩坑.md
```

## 进度概览

| 模块 | 内容 | 当前状态 |
| --- | --- | --- |
| A1 Basics | BPE、Transformer、训练工具与单元测试 | 已完成主要实现，并保留实验与复盘 |
| A1 Experiments | BPE 压缩率、merge profile、RoPE 上下文等实验 | 已有计划、冻结记录与结果 |
| A2 Systems | PyTorch FlashAttention 参考实现 | 已开始实现 |
| A2 Systems | Triton FlashAttention、DDP、FSDP、Sharded Optimizer | 待完成 |
| Exercises | 分词、MoE、GPU benchmark、资源估算 | 持续补充 |

## 环境准备

每个 assignment 都是独立的 Python/`uv` 项目。要求 Python `>=3.12,<3.14`；建议安装 [uv](https://docs.astral.sh/uv/)。

```powershell
winget install --id=astral-sh.uv -e
```

首次在对应目录运行 `uv run` 时，依赖会按各自的 `pyproject.toml` 自动解析和安装。A1 锁定了 PyTorch CUDA 12.8 wheel；没有 NVIDIA GPU 时，可按本机环境调整 PyTorch 依赖后运行 CPU 可用的测试。

## 快速开始

### Assignment 1：基础实现与测试

```powershell
cd assignment1/assignment1-basics
uv run pytest
```

常用定向验证：

```powershell
uv run pytest tests/test_model.py -v
uv run pytest tests/test_tokenizer.py -v
uv run pytest tests/test_optimizer.py -v
uv run pytest -k test_train_bpe_speed -v
```

训练脚本位于 `assignment1/assignment1-basics/scripts/`，包括 BPE 训练、tokenize、训练语言模型和生成示例。数据集不纳入版本控制；原始下载说明见 [A1 README](assignment1/assignment1-basics/README.md)。

### Assignment 2：系统作业

```powershell
cd assignment2_system/assignment2-systems
uv run pytest -v ./tests
```

该模块覆盖 FlashAttention、DDP、FSDP 和 Sharded Optimizer 的测试接口。当前一些接口仍会抛出 `NotImplementedError`，测试结果可作为后续实现的检查清单。

### 独立练习

从仓库根目录直接运行：

```powershell
python exercises/分词器.py
python exercises/MoE.py
python exercises/GPU编程.py
python exercises/资源核算.py
```

这些脚本依赖 PyTorch、`regex` 等包；推荐复用 A1 的 `uv` 环境运行，例如：

```powershell
cd assignment1/assignment1-basics
uv run python ../../exercises/分词器.py
```

## 阅读建议

建议按下面的顺序浏览仓库：

1. 从 `exercises/分词器.py` 建立 tokenization 的直觉。
2. 阅读 `assignment1/assignment1-basics/cs336_basics/linear.py` 与 `train_bpe.py`，并用 A1 测试定位接口契约。
3. 对照 [模型层手撕与踩坑笔记](notes/CS336-A1-模型层手撕与踩坑.md) 复盘数值稳定性、张量形状、mask 极性与 RoPE 广播等常见问题。
4. 查看 `assignment1/assignment1-basics/experiments/` 中的 `PLAN.md`、`FREEZE*.md`、`DISCOVERY*.md` 与 `results/`，了解实验如何从假设走到可复现结果。
5. 最后进入 A2 的 `cs336_systems/`，从 block-wise online softmax 开始，逐步完成并行训练与状态分片。

## 参考资料

- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [uv 文档](https://docs.astral.sh/uv/)

## 说明

- 数据、模型权重、checkpoint 和构建产物已由 `.gitignore` 排除。
- Assignment 子目录保留课程原始 README、题面 PDF 与许可证；根目录 README 只描述本仓库的学习组织与当前进度。
