# CS336 Assignment 2：Systems and Parallelism

> [!summary] 一句话主线
> 先用可重复的实验找出时间和显存瓶颈，再用 activation checkpointing、FlashAttention、混合精度和分布式并行降低瓶颈。每种优化都必须同时回答：节省了什么资源、增加了什么代价、实验如何验证。

## 0. 作业地图

| 模块 | 核心问题 | 主要代价/收益 | 对应实验 |
|---|---|---|---|
| Profiling | 时间花在哪里、显存峰值在哪里 | 测量开销，但避免盲目优化 | Python benchmark、Nsight、memory viz |
| Activation checkpointing | 训练时如何少保存激活 | 少显存，多重算 | 分段与递归 checkpoint |
| FlashAttention-2 | 如何避免 `N×N` attention 中间矩阵 | 少 HBM IO/显存，增加重算和 kernel 复杂度 | PyTorch、compile、Triton 对比 |
| Mixed precision | 如何利用 Tensor Core | 更快、更省显存，需控制数值误差 | FP16/BF16/autocast/累加实验 |
| DDP | 多卡如何保持同一份模型 | 梯度通信 | individual all-reduce、flatten、overlap |
| Optimizer sharding | AdamW 状态如何不重复存储 | 少状态显存，多广播 | sharded optimizer accounting |
| FSDP | 参数、梯度、状态都如何分片 | 最省显存，需 all-gather/reduce-scatter | FSDP correctness、Nsight |
| DP/TP/FSDP 分析 | 何时通信成为瓶颈 | 用带宽和算力选择并行度 | ring collective 与通信计算不等式 |

## 1. 实验方法：先测准，再解释

### 1.1 CUDA 计时为什么要同步

CUDA kernel 通常是异步提交的；CPU 看到的函数返回时间不等于 GPU 执行时间。每个测量 step 后应调用：

```python
torch.cuda.synchronize()
```

推荐流程：随机初始化模型和 batch；先运行 `w` 个 warm-up step；再测 `n` 个 step，报告均值和标准差。warm-up 用来排除 CUDA context、内存分配、kernel autotuning、缓存建立等一次性成本。没有 warm-up，第一次或前几次通常偏慢且方差大；只做 1～2 次也可能仍未稳定。

### 1.2 三层观测

1. **端到端 benchmark**：forward、backward、optimizer step 的总时间。
2. **Nsight Systems**：把 CPU API、GPU kernel、NVTX 范围和通信放在同一时间线上，定位瓶颈及 overlap。
3. **PyTorch memory snapshot**：查看 active memory timeline，区分参数、激活、梯度、optimizer state 和临时 workspace。

> [!warning] 常见误区
> 不要只看 FLOPs。GPU 性能还受 HBM 读写、kernel launch 次数、矩阵布局、同步和通信影响；一个 FLOPs 很少的 softmax 可能因为读写大量中间张量而耗时明显。

### 1.3 建议实验记录表

| 配置 | warm-up | measured steps | forward ms | backward ms | optimizer ms | peak GiB | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| model/context/dtype | 5 | 10 |  |  |  |  | 记录 GPU、PyTorch、commit |

## 2. 显存基础与 Activation Checkpointing

### 2.1 Autograd 为什么保存大量 residual

反向传播需要 forward 中间值。例如线性层 `y=xW` 的权重梯度需要 `x`，而激活函数、LayerNorm、attention 也各自需要输入或统计量。Transformer 层数、batch、序列长度增大时，保存的 residual 近似线性增加；attention 的 score/probability 矩阵还随序列长度平方增加。

### 2.2 checkpoint 的核心交换

`checkpoint(fn, x)` 只保存 checkpoint 输入，forward 内部中间值丢弃；backward 到达该段时重新执行一次 forward，再用新生成的 residual 求梯度。

- 收益：峰值激活显存下降。
- 代价：增加 forward 重算 FLOPs；调试和随机算子需要保持可重现。
- 关键：优化的是**峰值**，不是总分配量。

### 2.3 单层分段公式

设有 `N` 个 block，每段 `k` 个，单个 block 的临时 residual 成本为 `c`。每段保存一个边界 checkpoint，则：

$$
M(k)=\frac{N}{k}+ck.
$$

第一项是同时存活的段边界；第二项是当前重算段内的 `k` 个 block residual。对连续变量求极值：

$$
M'(k)=-\frac{N}{k^2}+c=0
\Rightarrow k^*=\sqrt{\frac Nc},
\qquad M(k^*)=2\sqrt{cN}.
$$

直觉是两项在最优点相等：边界显存和段内临时显存各占一半。这也可由 AM-GM 不等式得到。实际 `k` 必须取整数并考虑 block 数的因数，故实验应比较相邻候选值，而不是只用连续解。

### 2.4 递归 checkpoint

把 `N` 个 block 分成 `b` 段，并在每段内部继续递归。当前层只有后 `b-1` 段需要新增边界，因为第 0 段输入已经由上层保存：

$$
M(N)=(b-1)+M(N/b).
$$

若 `N=b^L` 且叶子成本为 `c`，展开得到：

$$
M(N)=(b-1)\log_bN+c.
$$

若把最顶层输入也单独计入，写作 `1+(b-1)log_b N+c`；两种写法只是 bookkeeping 约定不同。

递归分支因子满足：

$$
f(b)=\frac{b-1}{\ln b},
\qquad
f'(b)=\frac{\ln b-1+1/b}{(\ln b)^2}>0\quad(b>1).
$$

因此在允许的整数分支中 `b=2` 最优。它用更多重算换更低峰值显存。

### 2.5 计算量如何增长

- 单层 checkpoint：整条网络通常是约 `2× forward`（一次原始 forward + 一次段内重算）。
- 二叉递归：最早的 block 可能被多层重算；平均重算倍数约为 `1+(1-1/b)log_b N`，具体值取决于 checkpoint 调度。

统一的数量级直觉是：若允许约 `m` 次 forward 级别的计算，峰值激活可降到约

$$
M\approx mN^{1/m},\qquad C\approx m\times C_{forward}.
$$

`m` 越大，显存越低但计算越贵。实验验证方式：固定模型和 batch，扫描 checkpoint block size，测 `max_memory_allocated` 与 step time；报告最优点及相邻配置。

## 3. FlashAttention-2：把 IO 而不是 FLOPs 当作目标

### 3.1 普通 attention 的问题

$$
S=QK^T/\sqrt d,\quad P=softmax(S),\quad O=PV.
$$

`S`、`P` 形状为 `B×H×N×N`。训练时 autograd 还要保存它们或相关中间值，显存和 HBM 读写随 `N²` 增长。虽然矩阵乘法占主要 FLOPs，但把 `N²` 张量反复写入/读出 HBM 会成为瓶颈。

### 3.2 Tiling + online softmax

FlashAttention 将 Q、K、V 分块，只在片上 SRAM/register 中处理一个 score tile，不把完整 `N×N` 矩阵写回 HBM。为跨 tile 正确归一化，维护每行运行最大值 `m`、指数和 `l`、输出累加器 `o`：

$$
m_{new}=\max(m,\operatorname{rowmax}(S_j)),\quad
\alpha=e^{m-m_{new}},\quad p_j=e^{S_j-m_{new}},
$$
$$
l\leftarrow\alpha l+\operatorname{rowsum}(p_j),\quad
o\leftarrow\alpha o+p_jV_j.
$$

结束时：

$$
LSE=m+\ln l,\qquad O=o/l.
$$

只保存 `Q,K,V,O,LSE` 等 `O(ND)` 信息，避免保存 `P`。

### 3.3 backward 为什么需要重算

令 `D=rowsum(O\circ dO)`，由 `P=exp(S-LSE)`：

$$
dV=P^TdO,\quad dP=dOV^T,
$$
$$
dS=P\circ(dP-D),\quad dQ=dSK/\sqrt d,\quad dK=dS^TQ/\sqrt d.
$$

因此 backward 重新计算每个 tile 的 `S` 和 `P`，用额外 FLOPs 换取不保存 `N²` 矩阵。实现时 `dQ` 按 query tile 写入；`dK/dV` 按 key tile 累积，避免多个 program 对同一输出写冲突。因果 mask 在 tile 内按全局 query/key 索引判断未来位置。

### 3.4 PyTorch、compile、Triton 实验如何解释

1. **原生 PyTorch**：作为正确性与基线，长序列可能 OOM；记录 forward/backward 时间和 attention 中间矩阵估算。
2. **`torch.compile`**：减少 Python/kernel launch 开销，可能融合点操作；首次运行包含编译成本，必须 warm-up 后再测。
3. **Triton FlashAttention**：核心收益来自 tile、融合和减少 HBM IO；比较 forward、backward、峰值显存，而不是只比较 FLOPs。

## 4. Mixed Precision：Tensor Core 与数值稳定性

### 4.1 dtype 不是一个全局开关

`autocast(dtype=torch.float16/bfloat16)` 会按算子选择精度：矩阵乘通常使用低精度 Tensor Core，归约、归一化等敏感操作可能保持 FP32。模型 master parameters 通常仍为 FP32；线性层输出和 logits 可能是低精度，LayerNorm/损失的具体 dtype 由 PyTorch 算子策略决定，实验应打印实际 dtype，不要凭名称猜。

### 4.2 FP16、BF16、FP32

- FP16：吞吐高但动态范围小，梯度下溢/溢出风险高；训练常配 loss scaling 和 `GradScaler`。
- BF16：指数范围接近 FP32，通常比 FP16 稳定，但有效 mantissa 更少。
- FP32：精度和范围最好，Tensor Core 吞吐通常最低。

### 4.3 累加实验的结论

重复加 `0.01` 1000 次时，FP16 累加会因舍入产生明显误差；“FP16 输入转 FP32 后再加”通常比“FP16 变量持续累加”准确。原则是：乘法和带宽密集部分可低精度，反复累加、归约和 optimizer master state 保持高精度。

实验要同时报告 dtype、误差和速度；只看到更快不代表训练等价。

## 5. DDP：复制模型，切分 batch，同步梯度

### 5.1 数学机制

有 `N_DP` 个 rank，每个 rank 得到不同 batch shard。各 rank 独立 forward/backward，得到局部梯度 `g_i`；训练使用平均梯度：

$$
g=\frac1{N_{DP}}\sum_i g_i.
$$

随后每个 rank 用相同 optimizer 更新自己的完整参数副本，因此参数保持一致。初始时必须 broadcast rank 0 参数；否则各 rank 从不同模型开始，平均梯度也不能保证等价。

### 5.2 三种实现与实验预期

| 实现 | 通信方式 | 主要问题 |
|---|---|---|
| naive | 每个参数单独 synchronous all-reduce | kernel/latency 调用多 |
| flat | 把所有梯度 flatten 后一次 all-reduce | 调用少，但必须等整个 backward |
| overlap | 参数梯度 ready 后立即 asynchronous all-reduce | 通信可与后续 backward 计算重叠，需在 optimizer.step 前 wait |

使用 `register_post_accumulate_grad_hook` 触发通信；保存 request handles，在 `finish_gradient_synchronization()` 中 `wait()`。Nsight 中若通信条带与 backward kernel 同时出现，才是真正 overlap；仅使用 `async_op=True` 不等于已经隐藏通信成本。

### 5.3 ring collective 成本

令总 tensor 字节数为 `S`，设备数 `N`，每设备出带宽 `W`：

$$
T_{all-gather}=T_{reduce-scatter}=\frac{N-1}{N}\frac SW,
$$
$$
T_{all-reduce}=2\frac{N-1}{N}\frac SW.
$$

替代算法每轮发送完整 `S`，共 `N-1` 轮，所以：

$$
T_{alternate}=(N-1)\frac SW,
$$

比 ring reduce-scatter + all-gather 多传输约 `N/2` 倍（大 `N` 时）。

## 6. Optimizer State Sharding

AdamW 每个参数通常有 FP32 的一阶矩 `m` 和二阶矩 `v`，所以 optimizer state 约为参数字节数的 2 倍；再加参数、梯度和激活，峰值很容易受 optimizer step 支配。

简化 sharding：把参数集合分给各 rank，每个 rank 的 AdamW 只更新自己的 shard；step 后 broadcast 更新后的参数，让所有 rank 继续持有一致的完整模型。

- 显存：optimizer state 近似降为 `1/N`，但完整参数和梯度仍复制。
- 通信：每步需要广播更新参数；通信量与参数大小相关。
- 与 ZeRO-1：目标相近，都是只分片 optimizer states；具体实现的分片粒度、参数广播时机和通信组织可能不同。它不是 ZeRO-2/3，因为梯度/参数没有同时分片。

实验应在初始化后、optimizer.step 前后分别采样峰值显存，并拆解参数/梯度/state；再比较每 iteration 时间，解释“省显存但广播增加时间”的 trade-off。

## 7. FSDP：参数、梯度、状态全部分片

每个 rank 只保存每个 weight 的 `1/N` shard。使用某层前 all-gather 得到完整 weight，计算后释放；反向得到完整梯度后 reduce-scatter，只留下本 rank 的梯度 shard。norm 等很小的层通常不值得分片，因为通信 latency 可能超过收益。

典型生命周期：

```text
local weight shard
  -> all-gather
full compute weight (可先 cast 到 BF16/FP16)
  -> forward/backward
  -> 释放 full weight
  -> reduce-scatter gradient
  -> local master FP32 weight + local optimizer state
```

混合精度下，master weight 和 optimizer update 保持 FP32；通信和计算可使用低精度 weight，从而减少带宽。为了不让通信阻塞计算，应提前预取下一层 all-gather，并在当前层计算期间重叠。

FSDP 实验重点：正确性（与非并行 baseline 参数一致）、梯度 shape/dtype、峰值显存、all-gather 是否在使用前完成。若 Nsight 显示 all-gather 晚于计算开始，说明预取或 stream 同步设计有问题。

## 8. DP、FSDP、TP 的通信-计算建模

令 batch `B`、模型宽度 `D`、FFN 宽度 `D_FF`、设备算力 `C` FLOP/s、带宽 `W` byte/s，权重/激活用 FP16（2 bytes）。忽略非 matmul 操作。

### 8.1 FFN 基础 FLOPs

FFN：`xW1`、`xW2`、`zW3` 三个矩阵乘；forward FLOPs：

$$
F_{fwd}=6BD D_{FF}.
$$

backward 中每个 matmul 的输入梯度和权重梯度各需要一次同等规模乘法，因此：

$$
F_{bwd}=12BD D_{FF}.
$$

### 8.2 DP

每 rank batch 为 `B/N_DP`，backward 计算量：

$$
F_{bwd,DP}=\frac{12BD D_{FF}}{N_{DP}}.
$$

三份权重梯度总字节数为 `2(DD_{FF}+DD_{FF}+D_{FF}D)=6DD_{FF}`，all-reduce 通信时间近似：

$$
T_{comm,DP}=2\frac{N_{DP}-1}{N_{DP}}\frac{6DD_{FF}}W.
$$

计算时间为 `F/C`。compute-bound 条件 `T_comm <= T_compute` 给出设备数上限；随着 `N_DP` 增大，计算按 `1/N_DP` 降低，而梯度通信几乎不降，因此最终通信成为瓶颈。

### 8.3 FSDP

batch 仍按 `N_FSDP` 切分，所以 forward/backward FLOPs 分别为：

$$
F_{fwd}=\frac{6BD D_{FF}}{N_{FSDP}},\qquad
F_{bwd}=\frac{12BD D_{FF}}{N_{FSDP}}.
$$

forward 需要 all-gather 权重，backward 需要 all-gather 权重并 reduce-scatter 梯度。按三份权重合计 `6DD_FF` bytes，近似：

$$
T_{fwd,comm}=\frac{N-1}{N}\frac{6DD_{FF}}W,
$$
$$
T_{bwd,comm}=2\frac{N-1}{N}\frac{6DD_{FF}}W.
$$

与 DP 的关键差异：DP 通信主要是梯度 all-reduce；FSDP 通信还包括 forward/backward 的权重 all-gather，但参数、梯度、optimizer state 都能分片，显存大幅下降。

### 8.4 TP

FFN 中 `W1,W2` 按输出维 column shard，`W3` 按输入维 row shard。forward 的局部输出需要一次 activation all-reduce，大小约 `2BD` bytes：

$$
T_{fwd,TP}\approx\frac{N_{TP}-1}{N_{TP}}\frac{2BD}{W}.
$$

backward 还要同步与分片方向对应的输入梯度/权重梯度；写公式时要逐层追踪 `dy -> dz -> dx1,dx2 -> dx`，不要把所有梯度都误认为需要 all-reduce。TP 降低每个 matmul 的 FLOPs，但通信对象是 activation，故 batch 和序列/token 数会直接影响可扩展性。

### 8.5 2D：FSDP + TP

设备网格大小 `N=N_FSDP N_TP`。FSDP 轴通信和 TP 轴通信若可重叠，则 forward 通信时间是：

$$
T_{comm}=\max(T_{FSDP},T_{TP});
$$

若共享网络资源不能重叠，则是 `T_FSDP+T_TP`。选择 `N_FSDP,N_TP` 的原则是让两条轴的通信成本尽量均衡，同时满足：

$$
T_{comm}\le \frac{F_{fwd}}C.
$$

这解释了为什么“更多卡”不必然更快：总设备数增加会降低每卡计算，却让通信占比上升。实际系统还需考虑拓扑、链路带宽、kernel overlap、critical batch size 和显存约束。

## 9. 实验与知识点的闭环

### 9.1 每个实验都写四句话

1. **假设**：例如“flatten 梯度会减少 launch latency”，“FlashAttention 会降低 `N²` 激活显存”。
2. **控制变量**：固定模型、batch、context、dtype、GPU、warm-up 和测量步数。
3. **观测**：时间均值/标准差、峰值显存、kernel/通信时间、正确性误差。
4. **解释与边界**：说明结果是否支持假设，以及是否受编译、拓扑、OOM 或 overlap 影响。

### 9.2 推荐验证顺序

```text
baseline correctness
  -> end-to-end timing
  -> Nsight/memory profile 定位瓶颈
  -> 实现优化
  -> correctness regression
  -> 同配置重新 benchmark
  -> 用公式解释 scaling 与 trade-off
```

### 9.3 最容易被追问的结论

- warm-up 不是“让程序更快”，而是排除一次性初始化，使测量稳定。
- FlashAttention 的核心不是减少理论 attention FLOPs，而是减少 `N²` 中间结果的 HBM IO 和显存。
- asynchronous collective 只有在正确 wait 且时间线上与计算重叠时才产生收益。
- optimizer state sharding 只分片 state；FSDP 才进一步分片参数和梯度。
- 最优 checkpoint 在“边界保存”和“段内重算”相等处；递归二分来自 `(b-1)/ln b` 单调递增。
- 通信瓶颈判断必须比较 `T_comm` 与 `T_compute`，不能只看 FLOPs 或只看带宽。

## 10. 新增实测结果（RTX 3090）

详细原始记录见 [[../exercises/assignment2_writeup_notes|assignment2 writeup notes]]。下面只保留能解释知识点的关键结果；所有时间均为 warm-up 后的平均值，GPU 为 RTX 3090 24GB。

### 10.1 Roofline 与端到端基线

3090 的 FP32 峰值约 `35.6 TFLOP/s`、显存带宽约 `936 GB/s`，ridge point 为：

$$
I_{ridge}=35.6\times10^{12}/936\times10^9\approx38\;\text{FLOP/Byte}.
$$

small 模型（约 128.6M 参数，batch=4，context=512）实测：

| 模式 | 每步 | TFLOP/s | 峰值显存 |
|---|---:|---:|---:|
| forward FP32 | 47.5 ms | 11.9 | 0.76 GB |
| forward+backward FP32 | 148.4 ms | 11.4 | 4.88 GB |
| full step BF16 autocast | 111.2 ms | 15.2 | 4.52 GB |

估算 AI 远高于 38，但实测仍只有 11～15 TFLOP/s，说明模型并非只受理论算力限制：RMSNorm、RoPE、softmax、embedding、kernel launch 和小 batch 都会稀释 GEMM 利用率。BF16 端到端只比 FP32 快约 1.33 倍，原因是只有部分算子进入 Tensor Core。

### 10.2 FlashAttention 的“内存先赢、速度后赢”

纯 attention（B=1，D=64）显示朴素实现的峰值近似 `O(N²)`，FlashAttention 为 `O(N)`：

| N | 朴素 FP32 | Flash FP32 | Flash/朴素 |
|---:|---:|---:|---:|
| 1,024 | 17.2 MB | 0.3 MB | 1.6% |
| 4,096 | 135.3 MB | 1.1 MB | 0.8% |
| 8,192 | 539.0 MB | 2.1 MB | 0.4% |
| 16,384 | 2,151.7 MB | 4.3 MB | 0.2% |

端到端 small、batch=1、context=8192、BF16 forward：朴素 attention 为 672.1 ms、10.64 GB；Triton FlashAttention 为 **169.0 ms、1.04 GB**，分别约 4.0 倍加速和 10.2 倍省显存。context=512 时两者速度接近，说明 attention 占比太小；context 越长，`N²` HBM 流量越容易成为主瓶颈。

反例同样重要：Python 实现的 flash-pytorch 在 context=2048 的 full step 比朴素实现慢 23%。分块数学本身不保证更快；如果每个 tile 都启动多个 Python/小 kernel，launch 开销会抵消 IO 收益。真正的速度收益来自 Triton/CUDA 融合。

### 10.3 `torch.compile` 的作用与边界

官方 attention 网格上，`torch.compile` 将朴素 attention 的速度提高约 1.7～2.1 倍，并把峰值显存约减半（仍是 `O(N²)`）。整模型 small full step 从 170.9 ms 降到 134.8 ms（1.27 倍），medium 从 508.9 ms 降到 417.3 ms（1.22 倍）。

解释：Inductor 融合 softmax、RMSNorm、SwiGLU、残差等小算子并减少 kernel launch；但它无法把带动态 Python `range` 的自定义分块 autograd Function 稳定 trace。因此“compile 朴素实现”和“手写融合 Triton kernel”是互补路线，不是同一件事。

### 10.4 backward 重计算的实测取舍

N=8192、D=64 的 attention backward 对比：存储 `P` 的方案峰值约 1050 MB；FlashAttention 重计算方案约 34 MB，约 30 倍节省，但教学版 FP32 Triton kernel 慢约 3.7 倍。梯度最大误差约 `10^-6`，远小于 `1e-2` 测试容差。

causal 模式下 Triton backward 从 21.4 ms 降到 15.8 ms，因为 query/key tile 可以跳过未来区域；朴素实现反而可能因物化 mask 和 `where` 变慢。这个结果把“算法减少无效 tile”与“kernel 是否真正利用该结构”联系起来。

### 10.5 batch、checkpoint 与分布式 profiling

- batch 从 1 增到 16 时，small full FP32 吞吐从 4.99 提升到 12.06 TFLOP/s；固定 launch 开销被摊薄后趋于饱和。batch=32 在朴素 attention 下 OOM，直接说明 `N²` 激活限制了可训练 batch。
- medium full、context=2048 的 checkpoint 扫描中，`k=1` 峰值约 13.1 GB，`k=2` 约 16.1 GB，`k=4` 约 22.0 GB；因为每个 block 的 attention residual 很大，实际 `c\gg N`，理论最优 `k^*=\sqrt{N/c}<1`，所以最小合法段大小最优。
- 双进程 gloo 模拟中，baseline/DDP/DDP-sharded/FSDP 每步约 342/702/1937/2838 ms。这个排序主要反映通信次数和 gloo 延迟，不能当作 NCCL/NVLink 的绝对性能；它仍验证了“逐参数 collective 的延迟”和“未及时 reshard 的临时 buffer”是实现风险。

> [!tip] 写进 writeup 的方式
> 每张表后只回答三件事：结果支持了哪个公式？性能变化来自算力、IO、launch 还是通信？实验有哪些局限？不要把所有脚本实现细节混在结论段里。

## 11. 代码入口与测试

- `cs336_systems/attention.py`：PyTorch/Triton FlashAttention autograd Function。
- `cs336_systems/distributed.py`：DDP/FSDP 相关容器与通信逻辑。
- `cs336_systems/optimizer.py`：optimizer state sharding。
- `tests/test_attention.py`、`test_ddp.py`、`test_sharded_optimizer.py`、`test_fsdp.py`：正确性约束。
- 官方作业说明：`assignment2_system/assignment2-systems/cs336_assignment2_systems.pdf`。

> [!note] 实验记录边界
> 上述数字来自当前 writeup notes 中记录的 RTX 3090 实验；不同 GPU、CUDA、PyTorch、warm-up 和通信后端会改变绝对时间。报告中应同时保留配置、均值/标准差和 Nsight 截图。
