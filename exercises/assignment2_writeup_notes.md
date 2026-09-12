# Assignment2 学习笔记：通信量与显存核算

> 综合：公式推导、ring all-reduce / ZeRO 论文核对，以及代码实测复核。
> 对应 writeup 题目：`alternate_ring_all_reduce`、`data_parallel_calcs`、`fsdp_calcs`、
> `optimizer_state_sharding_accounting(c)`、`fsdp_accounting(a)`。
>
> ⚠️ 标注【自己推】的小节建议先遮住答案自己推一遍再看。

## 先看结论

这份笔记同时承担“推导草稿”和“实验日志”两个用途，所以细节较多。写正式 writeup 时建议每道题只保留三层：

1. **答案框**：先给最终公式或 1～2 句结论。
2. **一行理由**：说明通信对象/矩阵形状/显存组成。
3. **证据**：只放最能区分假设的表格或截图；脚本细节放附录。

本文后半的 RTX 3090 数据是实验日志，不必全部复制进 writeup。最重要的证据链是：`N²` attention 显存 → FlashAttention 的 `O(N)` 显存；Python 分块不等于快 → Triton 融合后长上下文反超；DDP/FSDP 的理论节省 → 临时 gather buffer 和 gloo 延迟会稀释实测收益。

## 术语速查：术语后面必须跟“它为什么重要”

| 术语 | 白话解释 | 在本实验中的作用 |
|---|---|---|
| **kernel** | GPU 执行的一小段函数；一次矩阵乘、softmax 或逐元素运算通常对应一个或多个 kernel | kernel 越碎，启动和调度的固定成本越明显 |
| **kernel launch** | CPU 把一个 kernel 排入 GPU 队列的动作 | Python 分块会产生很多 launch，解释 flash-pytorch 为什么慢 |
| **HBM / 显存带宽** | GPU 的大容量显存及其每秒搬运数据的速度 | attention 反复读写 `N×N` 矩阵时，带宽而不是乘法次数可能成为瓶颈 |
| **片上存储（SRAM/register）** | GPU 内部、容量较小但速度很快的存储 | FlashAttention 把 tile 留在这里，避免把完整 score 写到 HBM |
| **Tensor Core** | GPU 中专门做低精度矩阵乘的硬件单元 | FP16/BF16 矩阵乘更快，但需要接受精度变化 |
| **autocast** | PyTorch 根据算子自动选择 FP16/BF16 或 FP32 的上下文 | 说明为什么模型参数仍可为 FP32，而部分计算使用低精度 |
| **residual / activation** | forward 中间产生、backward 还要用的张量 | checkpoint 通过不保存它们来降低峰值显存 |
| **FLOPs** | 浮点加法/乘法的次数；TFLOP/s 是每秒万亿次 | 衡量计算工作量，不等于真实运行时间 |
| **计算强度（arithmetic intensity）** | FLOPs 除以搬运的字节数，单位 FLOP/Byte | 判断程序更接近“算力受限”还是“带宽受限” |
| **roofline / ridge point** | 用算力峰值和带宽画出的上限模型；ridge point 是两种限制的交界 | AI 高于 ridge point 只表示理论上偏算力受限，不能保证实现达到峰值 |
| **all-reduce** | 每个 rank 提供一份张量，最后每个 rank 都得到总和/平均值 | DDP 用它同步各卡的梯度 |
| **all-gather** | 每个 rank 提供一片数据，最后每个 rank 都拿到完整数据 | FSDP 在使用分片权重前需要它 |
| **reduce-scatter** | 先把各 rank 的数据相加，再把结果切片分给各 rank | FSDP 只保留本 rank 的梯度片，节省显存 |
| **overlap** | 通信和计算在时间线上同时进行 | DDP/FSDP 只有真正重叠，通信时间才可能被隐藏 |
| **gloo / NCCL** | PyTorch 的分布式通信后端；gloo 常用于 CPU，NCCL 面向 NVIDIA GPU | 本笔记的 gloo 双进程结果是保守上界，不能直接当作 NVLink 性能 |
| **occupancy** | 一个 SM 同时容纳多少并行线程/程序的程度 | tile 太大或共享内存太多会降低 occupancy，导致 Triton 变慢 |

写解释时采用固定句式：**“术语是什么 → 它改变了哪一项 → 表中哪个数字证明了这一点。”** 例如：“HBM 是 GPU 的大显存；朴素 attention 把 `N×N` score 写入 HBM，所以 N 翻倍时流量近似变四倍，表中 8192→16384 的显存正好体现了这一点。”

## 实验结果怎么读（白话版）

下面这段是所有实验的统一解释模板。正式 writeup 不要只写“快了 1.7 倍”或“显存少了 10 倍”，还要说明速度/显存变化究竟由哪一部分造成。

### A. `tl.dot` 精度实验：更快是因为计算方式变了

**预测。** FP32 的普通矩阵乘应该最准确但较慢；TF32、FP16、BF16 会把输入换成较短的数字格式，因此吞吐更高，但误差更大。BF16 的有效数字位比 FP16 少，所以同样的输入下它的误差通常最大。这里的“吞吐”就是一秒钟完成多少次乘加，不是模型最终训练速度。

**结果。** micro-benchmark 中 FP32-ieee 约 16.2 TFLOP/s，TF32/FP16/BF16 约 25～26 TFLOP/s；cuBLAS FP16 可到 61.6 TFLOP/s。误差排序大致为 FP32 < FP16/TF32 < BF16。

**解释。** 手写 Triton kernel 没有充分利用大矩阵的加载和并行能力，所以它只能达到硬件峰值的一部分；cuBLAS 是高度优化的库，能把数据更好地送入专门的矩阵计算单元。这个实验只说明“低精度矩阵乘有潜力更快”，不能直接推断整套 Transformer 也会按同样倍数加速，因为 Transformer 还有归一化、softmax、embedding 等操作。

### B. 朴素 attention 与 FlashAttention：先看显存，再看速度

**预测。** attention 的分数矩阵有 `N×N` 个元素，序列长度翻倍时显存应该接近变成 4 倍。FlashAttention 不把完整分数矩阵写入显存，只在小块内计算，因此显存应该随 `N` 近似线性增长。

**结果。** N=16,384 时，朴素 FP32 attention 约 2,151.7 MB，FlashAttention 约 4.3 MB，比例约 0.2%。朴素实现严格呈平方增长，FlashAttention 呈线性增长。

**解释。** 朴素实现不只是保存一个分数矩阵：softmax 输出和反向所需的临时结果也可能同时存在，所以实际峰值约为多个 `N²` 缓冲区之和。FlashAttention 只保留 Q、K、V、输出和每行一个 log-sum-exp 值，因此把最危险的 `N²` 项删掉了。这个结果首先证明的是“能不能跑更长序列”，其次才是“跑得快不快”。

### C. 为什么 flash-pytorch 可能比朴素实现慢

**预测。** 分块算法减少了大矩阵的存储，但如果每个小块都由 Python 循环调度，kernel 会被切得很碎；短序列上朴素实现反而可能更快。

**结果。** context=2048 的 full step 中，朴素实现约 263 ms，flash-pytorch 约 323 ms；context=512 时两者基本持平。

**解释。** 这里没有违反 FlashAttention 的理论，慢的是实现方式：12 层 × 多个 tile 会产生大量小操作，CPU 需要反复发出指令，GPU 也难以保持满负载。也就是说，“不保存 `N²` 矩阵”是内存策略，“融合成少数几个 GPU kernel”才是速度策略。两者必须同时做到，才能看到完整收益。

### D. Triton FlashAttention 的长上下文结果

**预测。** 短 context 时 attention 只占整步的一小部分，替换 attention 不会明显改变总时间；context 变长后，朴素实现的平方级读写会越来越重，Triton 版本应该逐渐反超。

**结果。** small、batch=1、context=8192、BF16 forward 中，朴素实现 672.1 ms、10.64 GB；Triton FlashAttention 169.0 ms、1.04 GB，约 4 倍加速、10 倍省显存。context=2048 FP32 时两者约 82.0/86.6 ms，但 FlashAttention 显存少约 54%。

**解释。** context=2048 时，FFN 和投影矩阵乘仍然占总时间的大头，attention 即使完全变快，也只能影响总时间的一部分；context=8192 时，attention 分数矩阵的读写已经成为主要负担，FlashAttention 同时得到“少写显存”和“融合操作”两个收益。BF16 还让 Triton 的矩阵乘进入专门的低精度计算单元，所以速度差距进一步放大。

### E. `torch.compile`：减少小操作，但不会改变平方增长

**预测。** 编译器可以把连续的小操作合并，减少每个操作单独启动 GPU kernel 的开销；但如果算法仍然生成 `N×N` 矩阵，显存的平方增长不会消失。

**结果。** attention 网格中，编译后速度提高约 1.7～2.1 倍，显存约减半；整模型 small full step 从 170.9 ms 降到 134.8 ms，medium 从 508.9 ms 降到 417.3 ms。

**解释。** 编译器把 softmax 链、RMSNorm、SwiGLU、残差相加等小操作合并了，所以少了很多“启动一次、读一次、写一次”的重复开销。它只是减少常数，不能把 `O(N²)` 变成 `O(N)`；因此 context=16,384 仍可能 OOM。当前自定义 flash Function 中带有动态 Python 循环，编译器无法稳定追踪，所以不能简单地把“Python 分块 flash”再套一层 compile。

### F. backward 重计算：用时间换显存

**预测。** 保存 `P` 会让反向过程继续持有多个 `N×N` 矩阵；不保存 `P`、反向时重新算它，应该显著降低显存，但会增加矩阵乘次数。

**结果。** N=8192 时，存 P 的方案约 1050 MB，FlashAttention 重计算约 34 MB，约 30 倍省显存；教学版 FP32 Triton backward 约慢 3.7 倍。梯度最大误差约 `10^-6`，明显小于测试容差 `1e-2`。

**解释。** 重计算增加的只是 `S=QK^T` 和 softmax 概率的计算，不会改变最终梯度公式；它把“存大矩阵”换成“再做一次矩阵乘”。教学 kernel 较慢主要是 FP32-ieee、较小 tile 和 dQ/dK/dV 分开的 kernel，不代表生产级 FlashAttention 的最终速度。causal 模式下 Triton 更快，是因为它跳过了确定不会用到的未来 tile；朴素实现仍可能先生成完整 mask，因而得不到同等收益。

### G. batch 扫描：小 batch 先被固定开销拖住

**预测。** batch 增大后，同一次 forward 处理的 token 更多，固定的 kernel 启动和调度时间被摊薄，GPU 利用率提高；但激活显存也会随 token 数增加，最终 OOM。

**结果。** batch=1 到 16 时，small full FP32 的吞吐从 4.99 提升到 12.06 TFLOP/s；batch=32 在朴素 attention 下 OOM。

**解释。** batch=1 时，GPU 还没来得及“吃饱”就已经做完了许多小操作；batch 增大后，大矩阵更适合并行计算，吞吐上升。继续增大 batch 并不会无限变快，因为显存和带宽会先达到上限。batch=32 的 OOM 是 attention 的 `N²` 激活与参数、梯度、AdamW 状态叠加的结果，换成 FlashAttention 可以直接减少其中最大的那一项。

### H. checkpoint 段大小：理论最优要结合实际 residual 大小

**预测。** 单层 checkpoint 的峰值是边界数量 `N/k` 加上重算段临时激活 `c·k`，因此一般呈 U 形；但如果每个 block 的 `c` 很大，最优点会落在最小合法 `k`。

**结果。** medium full、context=2048、朴素 attention 中，k=1/2/4 的峰值约 13.1/16.1/22.0 GB；k=8 OOM。FlashAttention 将 k=1 峰值降到约 8.0 GB，但最优仍是 k=1。

**解释。** 公式中的 `c` 不是抽象常数：朴素 attention 的每个 block 还带着很大的 score 和 softmax residual，实际 `c` 远大于 `N`，所以 `k^*=sqrt(N/c)<1`，只能选 k=1。FlashAttention 砍掉了 attention 的 `N²` residual，但 SwiGLU、归一化和其他中间值仍然很大，`c` 仍可能大于 `N`。因此 checkpoint 与 FlashAttention 是互补优化：前者减少同时保存的 block 数，后者减少每个 block 的内部残差。

### I. DDP/FSDP profiling：理论节省不等于立刻变快

**预测。** DDP 会增加梯度通信；把很多小梯度合并或在 backward 中提前通信，应该减少等待。FSDP 会减少常驻参数，但每层要 all-gather 权重，若实现中临时保存完整 buffer，实测节省会变小。

**结果。** 双进程 gloo 模拟中，baseline/DDP/DDP-sharded/FSDP 每步约 342/702/1937/2838 ms，峰值显存约 5.39/5.39/5.17/5.02 GB。

**解释与限制。** 这个排序说明当前实现中通信次数和单次通信延迟很重要，不能把它当作 NVLink/NCCL 的绝对成绩：两个进程共享一张 GPU，gloo 还经过主机转发，通信代价被严重放大。显存只下降少量，是因为 step 中仍短暂存在完整参数、gather list 或完整梯度；真正的 FSDP 需要逐层 gather、计算后立即释放，并用 reduce-scatter 保持梯度分片。这个实验更适合验证实现是否遵守生命周期，而不是比较生产集群速度。

---

## 0. 变量表与基础公式（所有题的地基）

分析用的 FFN 层（handout §8）：

```
x1 = x·W1     W1: (D, D_FF)
x2 = x·W2     W2: (D, D_FF)
z  = f(x1)*x2 (SwiGLU, elementwise)
y  = z·W3     W3: (D_FF, D)
```

| 符号 | 含义 |
|---|---|
| C | 设备算力 (FLOP/s) |
| W | 每设备出口带宽 (B/s) |
| N | 并行度 (N_DP / N_FSDP) |
| B | 全局 batch（token 数）；每设备拿到 B/N |
| S | 一个设备持有的数据大小 |

| 公式 | 内容 |
|---|---|
| matmul (A,B)×(B,C) | 2ABC FLOPs |
| 环形 all-reduce 时间 | 2(N−1)/N · S/W ≈ 2S/W（与 N 几乎无关！） |
| 环形 reduce-scatter / all-gather | (N−1)/N · S/W ≈ S/W |
| all-reduce 通信量 | 2S（= RS + AG 各 S） |
| **本作业的 Ψ** | 3 个 dW 合计 = 3·D·D_FF·2B(fp16) = **6·D·D_FF 字节** |

注意两套精度设定不要混：**§8 的计算题假设全 fp16（2B）**；我们的实现是纯 fp32（4B），§4/§5 显存核算按 fp32 走。

---

## 1. alternate_ring_all_reduce（1 分）【自己推 ✓ 已完成】

**题目**：变体算法每步 t=1..N−1，设备 i 发送**原始数据分片** x^((i−t+1) mod N) 给下家并累加收到的分片。耗时？

**答案**：每步每个设备发送一份完整原始分片（大小 S），共 N−1 步：

$$T_{variant} = (N-1)\cdot \frac{S}{W}$$

**对比经典环形 all-reduce** = RS + AG 两阶段 = 2(N−1)/N · S/W。比值 = **N/2**——变体多传约一半。

为什么经典环形更快：它每步传的是**累加和分片**（大小 S/N），变体每步传的是**原始分片**（大小 S）。累加和分片每步指数级"浓缩"了更多设备的信息，所以 RS 阶段只需 N−1 步、每步 S/N，就让设备 i 拿到第 i 块的全局和。

### 附：经典 ring all-reduce 图解（N=4, S=4，每块 1 元素）

目标：全员得到 [G0, G1, G2, G3]，其中 Gk = ak+bk+ck+dk。

**阶段一 reduce-scatter（3 轮）**——设备 i 发送块 (i−t) mod 4，接收块 (i−t−1) mod 4 并累加：

| 轮 | 设备0 发→收 | 设备1 发→收 | 设备2 发→收 | 设备3 发→收 |
|---|---|---|---|---|
| t=1 | a3→1, 收d2 | b0→2, 收a3 | c1→3, 收b0 | d2→0, 收c1 |
| t=2 | a2+d2→1, 收d1+c1 | b3+a3→2, 收a2+d2 | c0+b0→3, 收b3+a3 | d1+c1→0, 收c0+b0 |
| t=3 | a1+..+c1→1, **槽0=G0** | b2+..+d2→2, **槽1=G1** | c3+..+a3→3, **槽2=G2** | d3+..+b3→0, **槽3=G3** |

结果：设备 i 的第 i 块是全局和 → 每人一个分片，正是 reduce-scatter。

**阶段二 all-gather（3 轮）**——每轮把"上一轮刚收到的分片"转发给下家：

```
t=1: 0发G0得G3  1发G1得G0  2发G2得G1  3发G3得G2   (各持有2个)
t=2: 各转发新收到的 → 各持有3个
t=3: 再转发一次   → 各持有全部4个 ✓
```

每轮每设备发 S/N，共 2(N−1) 轮 → **时间 2(N−1)/N · S/W ≈ 2S/W，与 N 无关**。

数字验证（N=4, S=4, W=1）：经典 = 2·3/4·4 = 6；变体 = 3·4 = 12 = 1.5× ✓（= N/2 倍）。

---

## 2. data_parallel_calcs（3 分）【自己推 ✓ 已完成】

### (a) backward FLOPs

每设备（batch 切成 B/N_DP），backward 的 6 个 matmul（eq. 24–30，elementwise 忽略）：

| # | 公式 | 形状 | FLOPs |
|---|---|---|---|
| 1 | dz = dy·W3ᵀ | (B/N,D)×(D,D_FF) | 2(B/N)D·D_FF |
| 2 | dx1·W1ᵀ | (B/N,D_FF)×(D_FF,D) | 2(B/N)D_FF·D |
| 3 | dx2·W2ᵀ | 同上 | 2(B/N)D_FF·D |
| 4 | dW3 = zᵀ·dy | (D_FF,B/N)×(B/N,D) | 2(B/N)D_FF·D |
| 5 | dW2 = xᵀ·dx2 | (D,B/N)×(B/N,D_FF) | 2(B/N)D·D_FF |
| 6 | dW1 = xᵀ·dx1 | 同上 | 2(B/N)D·D_FF |

$$\boxed{FLOPs_{bwd} = \frac{12\,B\,D\,D_{FF}}{N_{DP}}}$$

验证：forward 只有 3 个 matmul = 6BD·D_FF/N_DP，backward 正好 2× ✓（与 benchmark 代码里 fwd+bwd≈3×fwd 一致）。

### (b) backward 通信时间

- 只有**权重梯度** dW1/dW2/dW3 出设备（激活梯度不出卡），每个 D·D_FF 元素 ×2B = 2D·D_FF 字节
- dW 尺寸不含 B、不含 N（batch 维被求和消掉）——这是关键
- 总 Ψ = 3·D·D_FF·2 = 6·D·D_FF 字节，all-reduce：

$$\boxed{T_{comm}^{bwd} = \frac{2(N_{DP}-1)}{N_{DP}} \cdot \frac{6\,D\,D_{FF}}{W} \approx \frac{12\,D\,D_{FF}}{W}}$$

### (c) N_DP 上界

瓶颈定义（原话）：通信时间 > 计算时间。

$$\frac{2(N-1)}{N}\cdot\frac{6DD_{FF}}{W} > \frac{12BD D_{FF}}{N\,C} \;\Rightarrow\; 12(N-1)\frac{DD_{FF}}{W} > 12B\frac{DD_{FF}}{C}$$

$$\boxed{N_{DP} > 1 + \frac{B\,W}{C} \;\text{时陷入通信瓶颈}}$$

要点：**12·D·D_FF 整个消掉** → 瓶颈条件与模型大小无关，只看"每卡分到多少数据"。B·W/C 越大（batch 大、网络快、机器慢）能扩展的卡越多。

直觉总结：计算时间 ∝ 1/N 线性下降，通信时间 ≈ 常数 → 卡数翻倍计算减半、通信不变，瓶颈必然到来。

---

## 3. fsdp_calcs（3 分）【参考答案——建议先自己推再对照】

FSDP = 数据也切（B/N_FSDP）+ 权重分片（每片 D·D_FF/N_FSDP）。backward 前每层要 all-gather 权重，backward 后 reduce-scatter 梯度。

### (a) FLOPs

FSDP 不改变计算量公式，只把 B 换成 B/N_FSDP（数据切分与 DP 相同，权重 gather 不算 FLOPs）：

$$FLOPs_{bwd} = \frac{12\,B\,D\,D_{FF}}{N_{FSDP}}, \qquad FLOPs_{fwd} = \frac{6\,B\,D\,D_{FF}}{N_{FSDP}}$$

### (b) 通信时间

每层的权重总量 = 3·D·D_FF 元素 = 6·D·D_FF 字节(fp16) = Ψ（与 Q2 的 Ψ 同值）：

- **forward**：每层前 all-gather 权重（Ψ）：
  $$T_{comm}^{fwd} = \frac{(N-1)}{N}\cdot\frac{6DD_{FF}}{W} \approx \frac{6DD_{FF}}{W}$$
- **backward**：all-gather 权重（Ψ，backward 也要用权重值）+ reduce-scatter 梯度（Ψ）：
  $$T_{comm}^{bwd} = \frac{2(N-1)}{N}\cdot\frac{6DD_{FF}}{W} \approx \frac{12DD_{FF}}{W}$$

**关键观察**：backward 通信量与 Q2 的 all-reduce 完全相同（2Ψ）！all-reduce 本来就是 RS+AG，FSDP 只是把这两半拆到两个时机用。FSDP 省的是显存，不是通信。

### (c) N_FSDP 上界

backward：(2(N−1)/N)·Ψ/W > 12BD D_FF/(N·C) → (N−1) > B·W/C
forward：((N−1)/N)·Ψ/W > 6BD D_FF/(N·C) → (N−1) > B·W/C

$$\boxed{N_{FSDP} > 1 + \frac{B\,W}{C} \;\text{时瓶颈}（forward/backward 同界）}$$

与 DP 的界相同——印证 (b) 的观察：通信总量一样、计算总量一样，瓶颈条件自然一样。FSDP 的纯收益是显存 1/N。

---

## 4. optimizer_state_sharding_accounting(c)（5 分中的分析部分）

**题目**：我们的实现与 ZeRO stage 1 (ZeRO-DP P_os, arXiv 1910.02054 §6.2) 有何不同？

### 三个层面的差异

**① 通信调度（核心）**

论文原话："Instead of an all-reduce, ZeRO only requires a scatter-reduce on the gradients (Ψ)... an all-gather to collect updated parameters (Ψ). Total = 2Ψ, exactly same as baseline."

| 方案 | 梯度同步 | 参数同步 | 合计 |
|---|---|---|---|
| 纯 DDP 基线 | all-reduce 2Ψ | — | 2Ψ |
| 论文 ZeRO-1 | **reduce-scatter Ψ**（自带） | all-gather Ψ | **2Ψ** |
| 我们 + DDP 外层 | all-reduce 2Ψ（外包给 DDP） | all-gather Ψ（step 内） | **3Ψ** |
| 我们 + FSDP 外层 | reduce-scatter Ψ（外包给 FSDP） | all-gather Ψ | **2Ψ** |

论文的 trick = **拆分 all-reduce**：环形 all-reduce 本来就是 RS+AG；既然 rank i 只更新自己那 1/N 参数，它根本不需要完整梯度，把前半段的 RS 单独用掉，省下的 Ψ 挪给参数 AG，总账仍是 2Ψ。
我们的实现把梯度归约外包给外层容器（handout 设计：optimizer 只负责参数同步）——配 DDP 多花 Ψ，配 FSDP 追平。

**② 通信粒度（工程差异）**

论文：参数展平成**一块连续 buffer** 等分 N 段，2 次集合通信搞定（大块传输，带宽打满）。
我们：**逐参数** all-gather（xl 几百个 tensor，每个 10–20µs 延迟起跳，延迟主导）。
另外 round-robin 按参数个数分配，embedding（大）和 bias（小）被平等对待 → 负载不均；论文按字节数等分天然均衡。

**③ 精度设定 → 分片天花板不同（最易漏）**

| 组成 | 论文（混合精度） | 我们的实现（纯 fp32） |
|---|---|---|
| 参数 | 2Ψ (fp16) | 4Ψ (fp32) |
| 梯度 | 2Ψ (fp16) | 4Ψ (fp32) |
| master weights | 4Ψ (fp32) | — |
| Adam m+v | 8Ψ (fp32) | 8Ψ (fp32) |
| **合计** | 16Ψ | 16Ψ |
| **被分片的** | **12Ψ (75%)** | **8Ψ (50%)** |
| **N→∞ 省上限** | **4×** | **2×** |

巧的是总 bytes/param 相同（16），但构成不同 → 论文"up to 4× memory reduction"的前提是混合精度，套在纯 fp32 实现上只能到 2×。

### 交付物模板（2-3 句）

> Our implementation shards the same components as ZeRO stage 1 (only AdamW states), but the communication schedule differs: ZeRO-1 replaces the gradient all-reduce with a reduce-scatter (Ψ) and all-gathers updated parameters (Ψ), totalling 2Ψ — the same as DDP. Our optimizer relies on the outer container for gradient synchronization (2Ψ with DDP, Ψ with FSDP) plus a per-parameter all-gather (Ψ) at step time, and does not flatten parameters into a contiguous buffer, so it issues many small collectives. Finally, our implementation is pure FP32 without master weights, so only 8Ψ of 16Ψ per-rank memory is sharded (2× ceiling), versus 12Ψ of 16Ψ (4× ceiling) under the paper's mixed-precision setup.

---

## 5. fsdp_accounting(a)（5 分中的分析部分）

**题目**：基于第 6 节的分析，FSDP 预计省多少峰值显存？（可忽略 all-gather 预分配缓冲）

### 显存阶梯（每卡，fp32，Ψ = 参数量，A = 激活）

| 方案 | 参数 | 梯度 | 优化器状态 | 合计 |
|---|---|---|---|---|
| DDP 基线 | 4Ψ | 4Ψ | 8Ψ | **16Ψ** |
| ZeRO-1 | 4Ψ | 4Ψ | 8Ψ/N | **8Ψ + 8Ψ/N** |
| ZeRO-2 | 4Ψ | 4Ψ/N | 8Ψ/N | **4Ψ + 12Ψ/N** |
| **FSDP (ZeRO-3)** | 4Ψ/N | 4Ψ/N | 8Ψ/N | **16Ψ/N** |

相对 ZeRO-1 的额外节省：Δ = (4Ψ+4Ψ)(1 − 1/N) = **8Ψ(1 − 1/N)**；N=2 时省 4Ψ。

### 代入本仓库的 xl（已用代码验证）

`benchmark.py` 的 MODEL_SIZES：d_model=2560, d_ff=10240, 32 层, vocab=10000。
参数量 N = 32×(4·2560² + 3·2560·10240) + 2×(10000×2560) ≈ **3.41B**（用 small 验证：公式算出 128.6M = benchmark 实测 ✓；注意元宝给的 2.0B 是别的配置，以代码为准）。

| N=2 | 每卡稳态（不含激活） |
|---|---|
| DDP 基线 16Ψ | ≈ 54.5 GB |
| ZeRO-1 12Ψ | ≈ 40.9 GB |
| **FSDP 8Ψ** | **≈ 27.3 GB** |

→ **FSDP 比 ZeRO-1 基线省 ≈ 13.6 GB（4Ψ），比 DDP 基线省 ≈ 27.3 GB（50%）**。

### 必须写进答案的两个前提

1. **激活不分片**：FSDP 不省 A；batch×context 越大，上述比例被稀释越多（A 的实测值来自 §1 的 profiling）。
2. **逐层 gather-and-free**：若一次性 gather 全模型，瞬时 +4Ψ 等于白干；backward 的 reduce-scatter 也要增量做，否则反向中短暂持有完整梯度 4Ψ。
3. （题目已豁免）forward 期间临时 gather 的完整权重占峰值，但题目明说可忽略预分配缓冲。

### 交付物模板（2-3 句）

> FSDP shards the parameters (4Ψ) and gradients (4Ψ) in addition to optimizer states, so per-rank steady-state memory drops from 8Ψ + 8Ψ/N_FSDP to 16Ψ/N_FSDP — saving 8Ψ(1 − 1/N_FSDP). For the xl model (Ψ ≈ 3.41B) with 2 GPUs this is ≈ 13.6 GB versus the ZeRO-1 baseline (≈ 40.9 GB → 27.3 GB). This assumes layer-by-layer gather-and-free of weights and incremental reduce-scatter of gradients; activations are not sharded, so the relative saving shrinks as activation memory grows.

---

## 6. 等 GPU 的部分（预期 checklist，到时对答案）

**4(a) 三个采样点（fp32, xl, N=2, Ψ≈3.41B）**：

| 时刻 | 构成 | 预期 |
|---|---|---|
| 初始化后 | 参数 4Ψ | ≈ 13.6 GB |
| step 前 | 参数+梯度+激活 | ≈ 27.3 GB + A |
| step 后 | 参数+梯度+m/v | ≈ 40.9 GB（ZeRO-1 下 ≈ 34.1 GB） |

坑：真峰值可能在**第一步 m/v 惰性分配的一瞬间**（参数+梯度+新分配状态同帧存在）——别只报稳态。

**4(b)**：变慢但换显存；2 卡 NVLink 带宽不是瓶颈，**逐参数小包的延迟**才是。

**5(b) Nsight 看三点**：首层 gather 必然暴露（bubble）；中间层 NCCL AllGather 与 GEMM 应并排（prefetch 生效）；若 gather/compute 严格交替 → prefetch 没生效，Q3 的"通信被掩盖"假设是假的。

---

## 7. 翻车点清单

| # | 坑 | 正确姿势 |
|---|---|---|
| 1 | Q1 变体答案写成 2(N−1)S/W | 变体每步发**完整原始分片** S，共 N−1 步 → (N−1)S/W |
| 2 | Q2(b) 忘了 dW 不含 B | 梯度大小 = 6DD_FF 字节，与 batch、卡数无关 |
| 3 | Q2(c) 保留 D·D_FF | 两边相乘后 12·D·D_FF 消掉 → 条件与模型大小无关 |
| 4 | fp16 的 2B 丢掉 | §8 计算题全是 fp16；显存核算才是 fp32 |
| 5 | 4(c) 只说"我们没分片梯度" | 核心是通信调度（3Ψ vs 2Ψ）与精度设定（2× vs 4×） |
| 6 | 论文"4×"直接套 fp32 | 纯 fp32 只能 2×（被分片的只有 8Ψ/16Ψ） |
| 7 | 5(a) 忘了激活不分片 | 节省比例被 A 稀释 |
| 8 | 5(a) 没提 gather-and-free | 一次性 gather 会让结论不成立 |
| 9 | xl 参数量用 2.0B | 本仓库 xl = 3.41B（d_model=2560, 32 层），以 benchmark.py 为准 |
| 10 | 4(a) 只报稳态峰值 | 第一步 m/v 惰性分配瞬间才是真峰值 |

---

## 相关代码

- DDP：`cs336_systems/distributed.py`（钩子异步 all-reduce）
- ShardedOptimizer：`cs336_systems/optimizer.py`（切片 → 本地 step → all-gather 拼回）
- FSDP：`cs336_systems/distributed.py`（dim0 分片 + forward 前 all-gather + finish 里 RS/AR）
- benchmark：`cs336_systems/benchmark.py`（MODEL_SIZES 与参数量验证）

---

## 8. 实验记录（RTX 3090, 2026-09-02）

### 8.0 实验环境与配置汇总

**软硬件环境**（所有实验共用）：

| 项目 | 配置 |
|---|---|
| GPU | RTX 3090, 24GB；fp32 峰值 35.6 TFLOP/s, bf16 峰值 71 TFLOP/s, 带宽 936 GB/s |
| ridge point | 35.6e12 / 936e9 ≈ **38 FLOP/Byte** |
| Python / PyTorch | 3.12.3 / 2.12.1+cu130 |
| CUDA / Triton | 13.0 / 3.7.1 |
| Flash kernel | Triton, BLOCK=64, num_warps=4（受 SM 共享内存 99KB 限制, BLOCK×num_stages 不能超） |
| 计时 | timeit 墙钟 + torch.cuda.Event；每步后 torch.cuda.synchronize() |
| 显存 | torch.cuda.max_memory_allocated（峰值 allocated） |
| 分布式 | 单卡双进程模拟, gloo backend（走主机转发, 计时为悲观上界） |

**各实验配置一览**：

| 实验 | 脚本 | 被测对象 | 模式 | 精度 | 扫描变量 |
|---|---|---|---|---|---|
| 8.1 tl.dot 精度 | /tmp/bench_dot_precision.py | 单个 matmul M=N=K=4096, BLOCK=64 | forward matmul | fp32-ieee / tf32 / fp16 / bf16（cuBLAS 对照） | input_precision / dtype |
| 8.2 峰值显存 | /tmp/bench_memory.py | 纯 attention, B=1, D=64 | forward | fp32 / fp16 | N ∈ {1024, 4096, 8192, 16384} |
| 8.3 端到端 | benchmark.py | small（128.6M）/ medium（423M）, b4 ctx512 | forward / full（fwd+bwd+opt） | fp32 / bf16（autocast） | 模型大小 × 模式 × 精度 |
| 8.4 naive vs flash 端到端 | benchmark.py --attention | small | forward / fwd-bwd | fp32（flash kernel fp32-ieee） | (b4 ctx512, b1 ctx2048) × {naive, flash-pytorch, flash-triton} |
| 8.5 ctx=2048 forward | benchmark.py | small, b1 ctx2048 | forward | fp32 | naive vs flash-triton |
| 8.6 决定性实验 | benchmark.py --mixed-precision --attention | small, b1 ctx8192 | forward | bf16（两边同标准） | naive vs flash-triton |
| 8.7 batch 扫描 | benchmark.py | small, ctx512 | full | fp32 | batch ∈ {1, 2, 8, 16, 32} |
| 8.8 DDP/FSDP profiling | exercises/ddp_fsdp_profile.py | small, b4 ctx512 | full step | fp32 | 2 进程 × {baseline, ddp, ddp-sharded, fsdp} |
| 8.9 backward 三方对比 | exercises/bench_flash_bwd.py | 纯 attention, B=1, D=64 | fwd+bwd | fp32 | N ∈ {1024,4096,8192} × {非causal, causal} × {naive, 存P, flash-pytorch, flash-triton} |

模型尺寸（与 benchmark.py MODEL_SIZES 一致）：small d768/L12 ≈ 128.6M；medium d1024/L24 ≈ 423M。

> 注：/tmp 下两个脚本为临时脚本，环境重置后需重建；benchmark.py 与 ddp_fsdp_profile.py 在仓库/exercises 中。

### 8.1 tl.dot 精度 vs 性能（/tmp/bench_dot_precision.py）

M=N=K=4096, BLOCK=64：

| 变体 | TFLOP/s | 最大误差 | 说明 |
|---|---|---|---|
| fp32-ieee | 16.2 | ~0 | CUDA core |
| tf32（默认） | 25.3 | 0.27 | 10位尾数, 输入端舍入 ~2^-11 |
| fp16 | 26.1 | 0.11 | 同 11 位有效尾数 |
| bf16 | 26.2 | 0.87 | 8 位有效尾数, 误差 ≈ fp16×2³ ✓ |
| cuBLAS fp32 | 23.2 | — | torch.matmul |
| cuBLAS fp16 | 61.6 | — | 大 tile + cp.async, 理论 71 的 84% |

结论：低精度提速 ~1.6×（未调优 kernel）；精度换速度的交易对 attention（tolerance 1e-2）完全可接受。

### 8.2 峰值显存：朴素 vs flash（/tmp/bench_memory.py, B=1, D=64）

| N | S 矩阵(fp32) | 朴素 fp32 | 朴素 fp16 | flash fp32 | flash/朴素 |
|---|---|---|---|---|---|
| 1024 | 4.2MB | 17.2MB | 4.3MB | 0.3MB | 1.6% |
| 4096 | 67.1MB | 135.3MB | 67.6MB | 1.1MB | 0.8% |
| 8192 | 268.4MB | 539.0MB | 269.5MB | 2.1MB | 0.4% |
| 16384 | 1073.7MB | 2151.7MB | 1075.8MB | 4.3MB | 0.2% |

与理论精确吻合：
- 朴素 fp32 = **2×S**（S 与 softmax 输出 P 同时存活, out-of-place）
- 朴素 fp16 = 2×S_fp16 = 1×S_fp32（精度省系数, 不改 O(N²) 渐近）
- flash = o + lse ≈ N×260B 严格线性（16384×260B = 4.26MB ≈ 实测 4.3MB）
- N=16384 时省 **500×**; N 翻倍比值减半（分子线性分母平方）

### 8.3 端到端 benchmark（batch=4, ctx=512, 3090）

| 配置 | 每步 | TFLOP/s | 峰值显存 | 吞吐 |
|---|---|---|---|---|
| small forward fp32 | 47.5ms | 11.9 | 0.76GB | 43.1k tok/s |
| small fwd-bwd fp32 | 148.4ms | 11.4 | 4.88GB | 13.8k tok/s |
| small full bf16 | 111.2ms | 15.2 | 4.52GB | 18.4k tok/s |
| medium full fp32 | 508.9ms | 10.8 | 14.74GB | 4.0k tok/s |

核对：fwd FLOPs 565.5G = 2·N·tokens + 4·B·L²·d·层数 ✓；fwd-bwd = 3×fwd ✓。
显存分解（small fwd-bwd）：参数 0.51 + 梯度 0.51 + AdamW 1.03 = 2.06GB，激活 ≈ 2.8GB。

**核心发现：AI 上界 1099-1648 FLOP/Byte >> ridge point（35.6e12/936e9 ≈ 38），理论 compute-bound，应接近 35 TFLOP/s，实测只有 11-12**。gap 来源：
1. cuBLAS fp32 实测上限也只有 23（65% peak），且模型混合大量非 GEMM 操作（softmax/RMSNorm/embedding/cross-entropy）
2. A1 模型 attention 是朴素 einsum 实现（物化 N²），非 flash
3. batch 小（2048 tokens/步），kernel launch 开销占比高
4. bf16 端到端只提速 1.33×（GEMM 部分 ~2×，被非 GEMM 部分稀释）——对比 §8.1 micro-bench 的 1.6×，端到端更低

#### 8.3.1 术语详解（对照上面四组数据）

- **FLOP / TFLOP/s**：一次浮点运算；矩阵乘 (A,B)×(B,C) = 2ABC FLOPs。TFLOP/s = 每秒 10¹² 次浮点运算。实测 11.4 = 1696.5 GFLOPs ÷ 0.1484s；硬件上限 35.6（3090 fp32）。
- **计算强度 AI**：FLOPs ÷ 显存字节数 = "搬 1 字节数据期间能做几次浮点运算"。
- **Roofline 模型**：可达速度 = min(算力峰值, AI × 带宽) = min(35.6 TFLOP/s, AI × 936 GB/s)。画出来是屋顶线：AI 小 → memory-bound（斜坡）；AI 大 → compute-bound（天花板）。
- **Ridge point**：两条线的交点横坐标 = 35.6e12 ÷ 936e9 ≈ 38 FLOP/Byte。
- **AI 上界 vs 真实**：估算只含参数读写（2×4N），忽略激活值流量。朴素 attention 的 N² 矩阵（§8.2）每步写+读，实际流量远大于 1.03GB → 真实 AI 大幅缩水，可能跌向 memory-bound。
- **kernel launch overhead**：每启动一个 kernel 有几微秒固定开销；一步几百个 kernel，batch 小时占比大 → batch 扫描时 TFLOP/s 会爬升。
- **混合精度**：autocast 把 GEMM 换成 bf16 输入进 tensor core（累加/主权重仍 fp32）。设 GEMM 占比 θ，加速 = 1/(1−θ+θ/2)；实测 1.33× 反解 θ ≈ 60% → 40% 时间在非 GEMM 操作上。
- **peak allocated vs reserved**：allocated 是程序实际在用的；reserved 是 allocator 向 CUDA 批发的库存（含复用空闲块）。
- **显存账本（fp32 + AdamW）**：每参数 16 字节（参数 4 + 梯度 4 + m 4 + v 4）。xl 3.41B → 54GB > 24GB 单卡装不下 → FSDP/ZeRO 的动机。

### 8.4 naive vs flash 端到端（--attention 选项, 3090）

benchmark.py 新增：`--attention {naive,flash-pytorch,flash-triton}`（monkey-patch A1 模型的
`scaled_dot_product_attention`）；AI 估算补上激活流量（隐状态 ~12 次读写/层 + 朴素 attention 的
S/P 各写+读一次 = 16·B·L²/层, flash 该项为 0）。

| 配置 | attention | 每步 | TFLOP/s | 峰值显存 | AI |
|---|---|---|---|---|---|
| b4 ctx512 fwd-bwd | naive | 148.1ms | 11.44 | 4.88GB | 381 |
| b4 ctx512 fwd-bwd | flash-pytorch | 149.2ms | 11.36 | **3.76GB** | 597 |
| b4 ctx512 forward | naive | 47.7ms | 11.83 | 0.76GB | 254 |
| b4 ctx512 forward | flash-triton | 49.9ms | 11.32 | 0.63GB | 398 |
| b1 ctx2048 fwd-bwd | naive | 263.2ms | 7.76 | **8.95GB** | 220 |
| b1 ctx2048 fwd-bwd | flash-pytorch | 322.6ms | 6.33 | **3.76GB** | 720 |

三个结论：
1. **flash 峰值显存与 ctx 无关**：batch×ctx 总 token 数相同（2048）的两组实验里，
   flash 都是 3.76GB（线性项相同, N² 项为零）；naive 从 4.88→8.95GB（N² 项 ×16）。
   ctx=2048 时 flash 省 58%。
2. **flash-pytorch 换内存不换速度**：ctx=2048 反而慢 23%（263→323ms）——Python 层
   for 循环 16 块 × 12 层 = 192 次小 kernel 串行, 输给了 cuBLAS 大 GEMM + 融合 softmax。
   "flash 快"的说法成立前提是**融合 kernel**（Triton/CUDA）, 不是分块算法本身。
3. **ctx=512 时 triton flash 也不占优**（49.9 vs 47.7ms）：N² 项太小, 12 层 kernel
   launch 开销盖过收益 → flash 的性能优势要长 ctx 才显现。

### 8.5 ctx=2048 forward: naive vs flash-triton（决定性实验）

| | naive | flash-triton |
|---|---|---|
| 每步 | 82.0ms | 86.6ms |
| 峰值显存 | 1.37GB | **0.63GB (-54%)** |
| AI 估算 | 147 | 480 |

**速度打平的原因（诚实归因）**：
1. attention 占总 FLOPs 仅 ~23%（155G/681G），FFN/proj 的 GEMM 主导整步时间——attention 再快也只影响 1/4
2. 我们的 kernel 是 fp32 + `input_precision="ieee"`（CUDA core, ~16 TF 上限）, 与 cuBLAS fp32 GEMM（23 TF）同精度; 融合省下的 softmax 访存被 kernel 效率差距抵消
3. 修过一个集成 bug: A1 模型经 rearrange/RoPE 的 Q/K/V 非连续, kernel 要求 `.contiguous()`（stride 契约）

**后续路线**：
- ctx=8192 时 attention FLOPs 占比升到 ~54% → naive 的 N² 访存拖累放大, flash 应反超
- kernel 支持 bf16 输入 → tensor core ~2× → 在相等效率下直接赢（对应 handout 混合精度题）

### 8.6 决定性实验: ctx=8192 + bf16（flash-triton 4× 反超）

kernel 已支持 bf16/fp16 输入（tensor core + fp32 累加）; 两边都跑 --mixed-precision（同标准）。

| ctx=8192 forward, bf16 | naive | flash-triton | 差距 |
|---|---|---|---|
| 每步 | 672.1ms | **169.0ms** | **4.0×** |
| TFLOP/s | 6.82 | **27.10** | 4.0× |
| 峰值显存 | 10.64GB | **1.04GB** | **10.2×** |
| 估算流量 | 27.84GB | 2.07GB | 13.5× |

解读：
1. **naive bf16 比 naive fp32 还慢**（6.82 vs 8.31 TF@2048）: autocast 下 softmax 仍
   提升回 fp32, S(bf16 写)→P(fp32 写) 的 N² 流量反而更大; 27.8GB 流量 @ ~900GB/s
   ≈ 31ms×12 层... 流量彻底主导, AI 165 也说明在带宽墙上
2. **flash-triton 27.1 TF = bf16 tensor core 峰值(71)的 38%**——教学 kernel 未调
   BLOCK/num_stages, 合理; 生产级 (FlashAttention-2) 可到 60-70%
3. **完整故事线（writeup 主线）**: ctx512 打平（attention 占比小）→ ctx2048 显存赢
   时间平 → ctx8192+bf16 时间 4× 显存 10×。"flash 赢在 O(N) 显存 + 融合访存,
   且优势随 ctx 和精度优化放大"

### 8.7 batch 扫描 roofline（small, full 模式, fp32, ctx=512）

| batch | tokens/步 | 每步 | TFLOP/s | 峰值显存 | AI |
|---|---|---|---|---|---|
| 1 | 512 | 85.0ms | 4.99 | 2.61GB | 146 |
| 2 | 1024 | 102.3ms | 8.29 | 3.53GB | 225 |
| 8 | 4096 | 302.7ms | 11.21 | 9.27GB | 381 |
| 16 | 8192 | 562.7ms | 12.06 | 16.86GB | 431 |
| 32 | 16384 | **OOM** | — | >23.5GB | — |

分析：
1. **爬坡→饱和**: 4.99 → 12.06 TFLOP/s, b16 后趋平。线性拟合 b8/b16 得
   固定开销 ≈ 43ms/步（launch + 小 kernel）, 边际速率 ≈ 13 TF——饱和点就是
   "fp32 GEMM(23TF 上限) + memory-bound softmax" 的混合墙, 远够不到 35.6 峰值
2. **b1 的教训**: 512 tokens 时固定开销占 50%, 算力利用率只有 14%——
   "batch 越大越 compute-bound" 的定量证明
3. **b32 OOM 也是结论**: 16K tokens/步 × 朴素 attention 的 N² 激活直接撑爆 24GB
   （fixed 2.06GB + 激活 ~21GB）——同样配置换 flash 就能跑（§8.2/8.4 的直接推论）

### 8.8 DDP/FSDP 单卡双进程 profiling（gloo, small, b4 ctx512, /root/Myllm/exercises/ddp_fsdp_profile.py）

| 变体 | 每步 | 每进程峰值显存 | 集合通信 |
|---|---|---|---|
| baseline | 341.8ms | 5.39GB | 0 |
| ddp | 702.1ms | 5.39GB | ~1×/参数 all-reduce |
| ddp-sharded | 1936.5ms | 5.17GB | + step 内 per-param all-gather |
| fsdp | 2838.0ms | 5.02GB | + forward 阻塞 all-gather |

结论（注意局限: 双进程共享 GPU → baseline 已比单跑慢 2.3×; gloo 走主机转发, 计时悲观上界）:
1. **显存方向正确但被瞬时缓冲抵消**: ddp-sharded 理论省 0.5GB 实省 0.22GB——step 里
   all-gather 临时物化完整参数 + gather_list; fsdp 同理（forward 全量参数 + backward
   全量梯度瞬时存在）→ 实证 flat buffer / bucket / 及时 reshard 的必要性
2. **计时排序 = 通信次数排序**: per-param 小通信延迟主导, 带宽没跑起来——
   生产级三件套: NCCL/NVLink + flat buffer 合并 + overlap（DDP 钩子已 overlap 但被
   gloo 延迟淹没）
3. 双进程 rank 间峰值完全一致（±0.01GB）→ broadcast/分片对称性正确

### 8.9 backward 三方对比: naive vs 存P vs 重计算（exercises/bench_flash_bwd.py, 2026-09-03）

**Triton backward 实现要点**（attention.py, extra credit 完成）:
- 预处理 `delta = rowsum(dO⊙O)`（PyTorch 一行, (B,N), fp32）
- **dQ kernel**: grid 按 Q 块分, 每个 program 独占一个 Q 块 → dq 直接 store, 零写冲突
- **dK/dV kernel**: grid 按 K 块分、沿 Q 块循环（转置分工）→ dk/dv 直接 store, 零写冲突;
  causal 时 Q 循环从对角块起点 `lo=(pid_n·BLOCK_N)//BLOCK_M·BLOCK_M` 开始
- 坑: backward 循环内驻留 tile 比 forward 多（q/do 常驻 + trans 缓冲）,
  BLOCK_N=64 + num_stages=2 仍要 112KB > 3090 的 99KB smem → **backward 用 BLOCK_N=32**
- 算法: P 不存, backward 用 `P = exp(S − lse)` 逐块重建; softmax 反传
  `dS = P ⊙ (dP − delta)`, `dQ = dS·K/√d`, `dK = dSᵀ·Q/√d`, `dV = Pᵀ·dO`

**正确性**: max|grad diff| vs naive（B=1, N=256, D=64）——全部远小于 1e-2 容差:

| 实现 | 非causal | causal |
|---|---|---|
| 存P（不重计算） | 1.79e-7 | 7.22e-7 |
| flash-pytorch（重计算） | 2.38e-7 | 1.19e-6 |
| flash-triton（重计算） | 5.36e-7 | 2.03e-6 |

**fwd+bwd 性能/显存**（B=1, D=64, fp32, iters=5; FLOPs 记账: fwd 4N²·D, bwd 存P/naive 8N²·D, 重计算 10N²·D）:

| 非causal | ms/step | 峰值MB | TFLOP/s | | causal | ms/step | 峰值MB |
|---|---|---|---|---|---|---|---|
| **N=1024** | | | | | | | |
| naive | 0.6 | 33.8 | 1.25 | | naive | 0.8 | 34.8 |
| 存P | 0.5 | 33.5 | 1.54 | | 存P | 0.6 | 33.5 |
| flash-pytorch | 5.1 | 21.3 | 0.18 | | flash-pytorch | 6.0 | 21.3 |
| flash-triton | 1.1 | 18.5 | 0.88 | | flash-triton | 1.2 | 18.5 |
| **N=4096** | | | | | | | |
| naive | 1.8 | 278.2 | 7.23 | | naive | 2.2 | 294.2 |
| 存P | 1.6 | 277.3 | 8.11 | | 存P | 1.8 | 277.3 |
| flash-pytorch | 25.8 | 36.3 | 0.58 | | flash-pytorch | 29.3 | 36.3 |
| flash-triton | 6.0 | 25.3 | 2.50 | | flash-triton | 5.5 | 25.3 |
| **N=8192** | | | | | | | |
| naive | 6.8 | 1052.2 | 7.63 | | naive | 8.1 | 1116.2 |
| 存P | 5.8 | 1050.3 | 8.86 | | 存P | 6.6 | 1050.3 |
| flash-pytorch | 44.6 | 56.3 | 1.35 | | flash-pytorch | 50.3 | 56.4 |
| flash-triton | 21.4 | 34.3 | 2.81 | | flash-triton | 15.8 | 34.3 |

结论:
1. **存P ≈ naive 显存（峰值 ≈ 4×N²）**: backward 里 p、dp、(dp−delta)、ds 四个 N²
   矩阵同时存活（out-of-place 各留一个）→ "少存一个 N²"救不了显存,
   **重计算（不存 P）才是 O(N) 的唯一路线**
2. **重计算的账**: +2N²·D FLOPs（12→14, +16.7%）换 N=8192 时 **30× 显存**
   （1050→34.3MB）; 速度慢 3.7× 是教学 kernel 效率问题（fp32-ieee 走 CUDA core、
   BLOCK_N 被 smem 压到 32、delta+两个 kernel 未全融合）, 不是重计算本身的错——
   生产级 FA2（bf16 tensor core + 调优）在长 ctx 下反超（衔接 §8.6 结论）
3. **causal 只有 flash-triton 受益**: 21.4→15.8ms（dQ 的 hi 裁剪 + dK/dV 的 lo 裁剪
   跳过约一半 KV 块）; naive/存P causal 反而更慢（N² 布尔掩码物化 + where）
4. flash-pytorch 最慢（Python 循环, 每 N 块 5 个小 matmul 走 autograd 调度）,
   flash-triton 融合版快它 2-4×——再次印证 §8.4 的"分块算法≠快, 融合才是"
5. 显存比随 N 单调下降（54.8%→9.1%→3.3%）: 分子线性分母平方, 与 §8.2 forward 同律

## 9. gradient checkpointing（§3, 4 分）

### 9.1 (a) 显存最优策略的理论

记账模型（handout 假设: 单个 block 的 residuals 主导一切簿记开销）:
- 每个 block 的 residuals（backward 需要的激活）= 1 单位; N 个 block
- checkpoint 一个位置 = 存 1 单位"边界激活"（该 block 的输入）
- 重算 = 重跑 forward 恢复该段内部 residuals（一段 k 个 block ≈ c·k 单位, c = 每 block 的 residual 个数, 量级 ~8）

三种策略对比:

| 策略 | 峰值激活 | forward 次数 |
|---|---|---|
| 不 checkpoint | N | 1 |
| 单层分段（每段 k 块） | N/k + c·k | 2 |
| 递归嵌套（每层分 b 份） | (b−1)·log_b N + c = O(log N) | 约 log_b N + 1 |

- 单层分段: 峰值(k) = N/k + c·k → 极值 **k* = √(N/c)**, 峰值 = 2√(cN)
  （两边各贡献一半: 边界 checkpoint 的 N/k 与重算段的 c·k 相等时最优）
- 递归嵌套: 当前层第 0 段输入已由上层保存，新增边界只有 `b−1` 个，因此
  `f(N)=(b−1)+f(N/b)`，`f(1)=c`，解为
  `f(N)=(b−1)log_b N+c`。连续分析 `((b−1)/ln b)` 对 `b>1` 单调递增，
  所以允许的整数分支中最优是 **b=2**，不是 `b≈e`；代价是 block 可能被多层重算。
- 代码草图: 递归函数把当前段二分，保存后半段输入 checkpoint，backward 时递归重算。

(b) 是 (a) 的受限版: 只许重算一遍（不许嵌套）→ 在单层分段里找最优 k,
用 k* = √(N/c) 与实验对照。

### 9.2 (b) checkpoint 段大小实验（exercises/bench_checkpointing.py）

3090 适配声明: xl full step 固定开销 16B/参数 × 3.41B ≈ 54.6GB > 24GB;
large 也实测全 k OOM（固定 21.5GB + 最少激活即超限）→ checkpointing 只省激活、
救不了参数/梯度/优化器状态。full step 用 medium（0.42B, 固定 6.8GB）复现规律,
xl 用 forward-only 验证。

**medium full step, fp32, b4 s2048, naive attention**:

| k(块/段) | 段数 | ms/step | 峰值MB | 备注 |
|---|---|---|---|---|
| 0（基线） | - | - | OOM | 24 块 × ~2GB 内部激活 |
| **1** | 24 | 3662.8 | **13113** | **谷底 = 每块都 checkpoint** |
| 2 | 12 | 3718.3 | 16062 | 重算 2 块, +3.0GB |
| 4 | 6 | 3726.9 | 21965 | 重算 4 块, +5.9GB |
| 8 | 3 | - | OOM | 重算 8 块 ≈ +16GB |

关键分析: 谷底在 k=1 而非理论 k* = √(N/c) ≈ 2, 因为 **c 被低估**——
naive attention 下每 block 的 backward 需保存 attention scores
（b4·h12·2048²·4B ≈ 805MB）+ softmax 输出（再 805MB）,
单个 block ≈ **62 个残差流单位（c≈62 ≫ N/4）** → N/k 项可忽略,
峰值 ≈ 固定 + c·k·C 随 k 单调增, 最小允许 k=1 即最优。
handout 的"next smaller/larger 对比"= k=1 vs k=2 vs k=4（k<1 不存在）。
时间代价: 2× forward ≈ +1.6%~54% 视 k 而定（这里 3.66s vs 3.73s, 差异被
通信/碎片掩盖, 重算本身约 +1 遍 fwd 的算力）。

**flash 对照实测**（medium full step, fp32, `--attention flash-pytorch`）:

| k(块/段) | naive 峰值MB | flash 峰值MB |
|---|---|---|
| 0（基线） | OOM | OOM |
| **1** | 13113 | **7986（-39%）** |
| 2 | 16062 | 8913 |
| 4 | 21965 | 10770 |
| 8 | OOM | 14484 |
| 16 | OOM | 21915 |

修正后的完整结论:
1. flash 只消灭 scores/softmax（~50 单位/块）, 每块 residual 仍 ≈ **25-30 单位**
   （SwiGLU 的 d_ff 中间量 134MB×2 + q/k/v/o + 双 RMSNorm fp32 保存量）,
   ×24 块 ≈ 20GB → **k=0 仍 OOM**; 但 k=1 峰值 13.1→8.0GB（-39%）
2. c≈30 仍 ≫ N=24 → k* = √(24/30) < 1 → **谷底仍在 k=1**。
   U 型曲线的谷底右移需要 c ~ N 量级（更深模型）; c ≫ N 时最小 k 恒最优。
   k* = √(N/c) 公式方向正确, 谷底位置由 c/N 相对大小决定
3. **flash 与 checkpointing 作用在不同项上**（flash 砍 c, checkpoint 砍重算段的
   c·k 项之外的边界项）, 可叠加: flash + k=1 = 8.0GB, 让 medium b4 s2048
   full step 从 naive 的 OOM 降到 8GB 内
4. 递归嵌套（9.1）在此场景可把峰值进一步压向 O(log N), 但 handout (b) 限制
   一次重算, 故不展开实验

**xl forward-only 实测**（fp32, b4 s2048; 无 backward → 无重算, 曲线方向反转）:

| k(块/段) | 段数 | 峰值MB | 备注 |
|---|---|---|---|
| 0 | - | OOM | 32 块 × ~2GB+ 内部激活全保留 |
| 1 | 32 | OOM | 32 边界 × 80MiB = 2.5GB → 顶破 24GB |
| 2 | 16 | 23011 | |
| 4 | 8 | 22371 | |
| 8 | 4 | 22051 | |
| 16 | 2 | 21891 | |
| **32** | 1 | **21811** | 最优 = 整模型一段 |

fwd-only 模型: 峰值 = 13.6GB(参数) + ~8.2GB(单 block forward 瞬时峰值:
scores 2.1GB + softmax 2.1GB 等, **任何 checkpoint 粒度都压不掉**)
+ 80MiB × N/k(边界数)。k 越大段越少 → 边界越少 → 峰值单调降 → k=32 最优。
实测 k=2 与 k=32 的差 1.2GB ≈ 15×80MiB 边界差 ✓。
**结论分叉**: U 型曲线（存在最优中间 k）只在 full step 出现——重算项 c·k
随 k 上升; fwd-only 只有单调下降项。handout (b) 的答案以 medium full step 为准:
**每块都 checkpoint（k=1）最优**。

(注: 脚本"激活MB(估)"列按 full step 的 U 型公式估算, 对 fwd-only 模式不适用,
以上表实测值为准。)

> [!note] 尚未完成的对照
> medium full step + flash-pytorch 的 `k=0` 基线和“谷底是否右移到 `k=2`”尚未实测；不要把这里的预测写成最终结论。

## 10. 官方 attention 网格（4.1.1: naive vs flash, B=8, fp32, 100 次计时）

脚本: sweep_attention.py（默认参数即官方网格; mem_MB = 单次带梯度 forward 的
allocator 峰值净增 = op 激活占用）。d=64 列（其余 d 几乎相同）:

| N | naive fwd ms | naive bwd ms | naive mem MB | flash-py fwd ms | flash-py bwd ms | flash-py mem MB |
|---|---|---|---|---|---|---|
| 256 | 0.29 | 2.16 | 8.0 | 0.49 | 3.64 | 4.5 |
| 512 | 0.43 | 2.22 | 32.0 | 1.18 | 5.55 | 9.1 |
| 1024 | 0.63 | 2.18 | 128.1 | 2.31 | 13.15 | 18.1 |
| 2048 | 2.33 | 8.25 | 512.2 | 4.31 | 23.81 | 36.2 |
| 4096 | 9.26 | 32.36 | 2048.4 | 12.03 | 52.97 | 72.5 |
| 8192 | 36.08 | 125.96 | 8192.8 | 44.07 | 205.39 | 145.0 |
| 16384 | **OOM** | - | (需 32GB+) | 168.17 | 792.95 | 290.0 |

结论:
1. **naive 显存严格 = 16·B·N² 字节**（N=256: 8.0MB, N=8192: 8192.8MB, 逐点吻合:
   S+P+两个中间量, 每个 B·N²·4B）。N=16384 需 ~34GB → OOM, flash 同点仅 266-386MB
   （**~30-60×**）, 完整跑通 —— O(N²) vs O(N) 的直接证据
2. **速度交叉点未到**: fp32+B=8 下 naive (cuBLAS 大 GEMM) 在 fwd 上一直更快
   （8192: 36.1 vs 44.1ms）, flash-pytorch 的速度劣势随 N 缩小;
   naive 的败因是 N=16384 直接 OOM —— flash 的价值在"能跑", 其次才是快
3. **d 不变性**: 时间/显存对 d∈{16..128} 几乎不敏感（N² 项主导, d·N 项小）
   —— 4 张 d 表可合并成一张

## 11. torch.compile 对照（4.2a: naive±compile, B=8, fp32, d=64, 100 次计时）

| N | fwd ms (原/编译) | 提速 | bwd ms (原/编译) | 提速 | mem MB (原/编译) |
|---|---|---|---|---|---|
| 256 | 0.28 / 0.25 | 1.2× | 1.62 / 1.06 | 1.5× | 8.0 / 4.5 |
| 1024 | 0.63 / 0.31 | **2.1×** | 2.51 / 2.10 | 1.2× | 128.1 / 66.1 |
| 4096 | 9.19 / 5.31 | 1.7× | 32.15 / 17.63 | 1.8× | 2048.4 / 1032.5 |
| 8192 | 35.86 / 21.12 | 1.7× | 125.15 / 68.33 | 1.8× | 8192.8 / **4113.0** |

结论:
1. **速度 +70%~110%**（inductor 融合了 softmax 链的小算子 + 减少 kernel launch）
2. **显存精确减半**（8192.8→4113.0, 128.1→66.1）: naive 的 4 份 B·N²·4B 缓冲
   （S + softmax 输出 + 2 中间量）被 inductor 融合成 2 份 —— 仍是 O(N²), 但常数减半
3. **编译后的 naive 反超 flash-pytorch**: fwd 21.1 vs 44.1ms @8192 ——
   flash-pytorch 的 Python 循环开销是硬伤; flash 的相对优势只剩显存（145MB vs 4.1GB）
4. **flash 系无法被 compile**: dynamo 无法 trace 带逐块 Python 循环的自定义
   autograd Function（`range(0, n_k, block)` 的动态边界 → ConstantVariable 断言失败）。
   "编译"与"Python 级分块"在当前工具链下不可兼得 —— 生产实现走 kernel 融合
   （我们的 Triton 路线）正是为了绕开这个限制

## 12. flash 官方大表（4.2.2: B=1, causal, bf16, 20 次计时, d=64 列）

| N | naive fwd/bwd ms | flash-triton fwd/bwd ms | triton 提速(fwd) | naive mem | triton mem |
|---|---|---|---|---|---|
| 512 | 0.53 / 2.36 | 0.31 / 0.82 | 1.7× | 2.3MB | 0.1MB |
| 1024 | 0.69 / 2.60 | 0.31 / 1.31 | 2.2× | 9.0MB | 0.1MB |
| 4096 | 0.84 / 2.70 | 0.31 / 1.32 | 2.7× | 144MB | 0.5MB |
| 8192 | 2.82 / 8.43 | 0.32 / 1.31 | **8.9×** | 576MB | 1.0MB |
| 16384 | 11.18 / 33.29 | **1.00 / 3.30** | **11×** | 2304MB | **2.1MB（1100×）** |

(flash-pytorch 全程最慢: N=16384 fwd 44.4ms, 比 naive 还慢 4× —— Python 循环开销
在 bf16 下更加显眼。)

结论:
1. **速度首次反超且随 N 扩大**: 1.7×(512) → 11×(16384)。bf16 让 naive 也吃到
   tensor core, 但 naive 的 N² 显存流量成为瓶颈; triton 不物化 S/P, 纯算力比拼
2. **fwd 达成率 ≈ 96%**: N=16384 causal fwd FLOPs = 2·N²·d = 34.4 GFLOP,
   / 1.004ms = **34.3 TFLOP/s ≈ 3090 bf16+fp32-accum 峰值(35.5) 的 96%** ——
   kernel 已从 launch/访存受限进入 compute-bound 饱和区
3. **N ≤ 8192 时 triton fwd 恒为 0.31ms**（launch/占用率地板）: B=1 时 grid 只有
   N/64 个 program, N=4096 才 64 个 < 82 SM → 未填满 GPU。B>1 的真实训练里
   这个地板会被并行度掩盖
4. **显存 1100×**: 2304MB vs 2.1MB @16384（bf16 下 naive 只物化 2 份 N²,
   flash 恒为 O(N)）
5. **d 的代价**: d=128 在大 N 变慢（fwd 2.29 vs 1.00ms）—— smem 压力掉占用率,
   与 8.6 的教训一致

## 13. torch.compile 整模型端到端（4.2b: benchmark.py --compile, full step, fp32, b4 ctx512）

| 配置 | 每步 | TFLOP/s | 峰值显存 |
|---|---|---|---|
| small 非 compile（实测基线） | 170.9ms | 9.98 | 5.40GB |
| small --compile | **134.8ms（1.27×）** | **12.59（+26%）** | **4.63GB（-0.77GB）** |
| medium 非 compile（8.3 基线） | 508.9ms | 10.8 | 14.74GB |
| medium --compile | **417.3ms（1.22×）** | **13.18（+22%）** | **12.71GB（-2.0GB）** |

结论:
1. 端到端 +22% 算力 / 1.22× 提速: 整模型 fused 的大头在逐层小算子
   （RMSNorm/RoPE/SwiGLU/残差加), attention 的 N² 部分只占一步的 ~8%
2. 显存 -2.0GB(-14%): 8.3 的激活 ≈ 9.6GB 里, inductor 融合省出 2GB
   （attention 缓冲减半 + 逐层中间量合并）
3. 注意 compile 用时 ~40s（逐 shape 编译）, 推理/长训场景才划算;
   TF32 未开启（inductor 警告）—— 若允许 set_float32_matmul_precision('high'),
   GEMM 部分还能再快, 但精度语义变化需在 writeup 声明

## 14. 计时规范矩阵（2.1.3b: warmup5 + measure10, fp32, b4 ctx512, 含 std）

| size | mode | ms/step (std) | TFLOP/s | 峰值显存 |
|---|---|---|---|---|
| small | forward | 47.2 (±0.0) | 11.97 | 0.76GB |
| small | forward-backward | 148.1 (±0.3) | 11.45 | 4.88GB |
| small | full | 169.0 (±0.2) | 9.97 | 5.40GB |
| medium | forward | 145.7 (±1.7) | 12.55 | 2.02GB |
| medium | forward-backward | 439.4 (±1.1) | 12.51 | 13.04GB |
| medium | full | 508.0 (±0.5) | 10.83 | 14.74GB |
| large | forward | 298.4 (±1.7) | **13.94** | 4.40GB |
| large | forward-backward | **OOM**（22.1GB 处 silu 分配失败） | - | - |
| large | full | **OOM**（20.0GB 处 einsum 失败） | - | - |

结论:
1. std 0.0-1.7ms（<0.6%）—— warmup 后计时非常稳; 所有对比实验的均值差
   都远大于 std, 结论可信
2. TFLOP/s 随模型变大爬升（fwd: 11.97→12.55→13.94）—— 与 §8.7 batch 扫描
   同一机理: 计算占比变大, launch 开销/小算子占比变小
3. forward→forward-backward: 时间 ≈ ×3（bwd ≈ 2× fwd ✓ 理论）; 
   fwd-bwd→full 只多 ~13%（optimizer step 被算力更强的部分掩盖）
4. **large fwd-bwd 都装不下**: naive attention 的 N² 激活（每层 2×84MB 的 
   S/P, 36 层 ≈ 6GB）+ SwiGLU 中间量把前向顶到 22GB。**换 flash 可解**
   （S/P 消失, 预计 ~16GB）—— 下面的 bonus 实验验证
5. CPU/GPU 时间差 < 0.12%: 两种钟测同一段墙钟（step 内含 sync）, 差异是
   事件时间戳的 enqueue 延迟噪声, 无语义（见 2.1.3a 讨论）

## 15. 无 warmup 对照 + 混合精度（2.1.3c / 2.1.5c）+ flash 救回 large

**无 warmup 对照**（medium full fp32）:

| 配置 | ms/step | std |
|---|---|---|
| warmup=5 | 508.0 | ±0.5ms |
| warmup=0 | 535.1 | **±91.6ms（×183）** |

无 warmup 时首步包含 cudnn 算法选择、allocator 首次 cudaMalloc、kernel 自动调优
等一次性开销, 均值被拉高 5.3%、方差爆炸 → **计时必须 warmup**（2.1.3 的核心结论）。

**混合精度 full step**（2.1.5c, b4 ctx512; large/xl 因 16B/参数硬地板在 24GB 不可行）:

| size | fp32 ms / TF / GB | bf16 ms / TF / GB | 提速 | 显存 |
|---|---|---|---|---|
| small | 169.0 / 9.97 / 5.40 | 110.8 / 15.30 / 4.52 | **1.53×** | -0.9GB |
| medium | 508.0 / 10.83 / 14.74 | 312.7 / 17.61 / 12.43 | **1.62×** | -2.3GB(-16%) |

- 用加速比反解 GEMM 时间占比 θ（加速 = 1/(1−θ/2)）: small θ≈0.69, medium θ≈0.77
  —— 模型越大 GEMM 占比越高, bf16 收益越大（与 §14 结论 2 一致）
- est bytes/step 减半（5.48→2.74GB）→ AI 翻倍（310→619）, 但实测 TF 只 +53-63%:
  bf16 GEMM 峰值（3090 tensor core fp32-accum ≈ 35.6 TF）与 fp32 CUDA core 相同,
  收益来自**激活流量减半 + tensor core 对 GEMM 的执行效率**, 不是峰值翻倍
- 显存: 激活减半, 但参数/梯度/AdamW 仍 fp32（-16% 而非 -50%）

**bonus: flash 把 large fwd-bwd 从 OOM 救回**:

| 实现 | 结果 |
|---|---|
| naive | OOM（前向 22.1GB 处 silu 分配失败） |
| flash-triton | **19.95GB 可跑**: 868.9ms, 14.37 TF |

"flash 的价值排序"最终版: ① 让装不下的配置可跑（large fwd-bwd, N=16384）
② 长序列 bf16 提速 11×（§12）③ 短序列 fp32 略慢但可接受。

## 16. memory_profiling（2.1.6; small 模型适配, b4, 10 步 full step fp32）

**峰值表**（peak = 分配峰值, resident = 步后常驻[参数+优化器状态]）:

| ctx | mode | dtype | peak MB | resident MB | 激活峰值 |
|---|---|---|---|---|---|
| 128 | forward | fp32 | 262 | 240 | 22MB |
| 128 | forward | bf16 | 358 | 240 | 118MB |
| 128 | full | fp32 | 1204 | 712 | 492MB |
| 128 | full | bf16 | 1160 | 712 | 448MB |
| 2048 | forward | fp32 | 2397 | 249 | 2148MB |
| 2048 | forward | bf16 | 1967 | 249 | 1718MB |
| 2048 | full | fp32 | 20847 | 713 | **20134MB** |
| 2048 | full | bf16 | 15939 | 712 | 15227MB |

(b) fwd vs full 之差 = 为 backward 保留的激活:
- 激活 = 线性项（B·s·d 量级, 随 ctx 线性: 残差/SwiGLU/ln 保存量）
  + **attention 平方项**（S+P = 2·B·h·s²·4B/层）
- 分解验证: ctx128 激活 492MB（其中 s² 项 2·4·8·128²·4B·12层 ≈ 50MB, 线性 ≈ 442MB）
  → ctx2048: 442×16 + 2·4·8·2048²·4B·12层(12.9GB) ≈ 7.1+12.9 = 20.0GB ✓ 实测 20.1GB
- **ctx×16 激活 ×41 倍**——平方项主导, naive attention 的 O(s²) 激活是长上下文
  显存爆炸的元凶（flash 直接消灭该项）

(c) 混合精度: ctx2048 full 省 4.9GB(-24%): 激活/S/P 减半, 但参数/梯度/AdamW
仍 fp32; fwd-only bf16 只省 20%（logits fp32 + autocast 缓存 + RMSNorm fp32 上转
使实际省不到一半）

(d) 激活张量大小推导: 单个残差张量 = B·s·d·4B（ctx2048: 16.8MB）;
每层保存 ≈ 30 个残差单位（§9.2 的 c）: S/P 各 537MB(64 单位) + SwiGLU 3×67MB
+ q/k/v/o/ln 等 → 1.68GB/层 × 12 = 20GB ✓

(e) 最大分配块: top 全是 ~20MB（= embedding/lm_head 权重 10000×512×4B = 20.5MB
及其梯度/动量）——**没有巨型单块**, 显存由数万个小激活块构成（这也是
expandable_segments 这类防碎片配置存在的原因）

(f) Nsight/NVTX 残差统计: 见 nsys 小节（依赖环境）

(a) Active memory timeline 截图: 上传 /tmp/mem_ctx128.pkl 与 /tmp/mem_ctx2048.pkl
到 https://pytorch.org/memory_viz 生成（手动步骤）

## 17. Nsight Systems 分析（2.1.4; small full step fp32 b4 ctx512, 8 步）

工具链: benchmark.py 已加 NVTX 标注(benchmark_step 区间) → nsys profile
(-t cuda,nvtx) → export sqlite → nsys_analyze.py（ns 单位坑 + shortName 需
JOIN StringIds）。

**(a) kernel 覆盖率**: 每个 step 的墙钟里 96.5% 至少有一个 kernel 在跑
（8 步范围 96.3-96.6%, 极稳）—— **空闲只有 3.5%**

**(d) 单步构成**: 墙钟 171.6ms, kernel 总时长 165.6ms; 29242 kernel / 8 步
≈ 3657 个 kernel/步, 平均单个仅 ~45µs

**(b) 最长无缝 kernel 链**: 仅 1.62ms（占 step 的 0.9%）—— 没有任何长连续
计算段, 全是短 kernel 无缝接力

**top kernel 累计**（8 步合计 ≈1323ms）:
| kernel | 累计 | 占比 | 归类 |
|---|---|---|---|
| vectorized_elementwise | 431.7ms | 33% | 逐元素（silu/残差/掩码/RoPE…） |
| Kernel2 (cuBLAS 内部) | 196.8ms | 15% | GEMM 变体 |
| ampere_sgemm ×5 种 | 488.3ms | 37% | GEMM |
| elementwise | 160.1ms | 12% | 逐元素 |
| reduce_kernel | 35.7ms | 3% | 归约 |

**核心解读（回答 8.3 的 gap 之谜）**: GPU 96.5% 时间"在忙", 但忙的一半是
逐元素小算子（45%, FLOP/Byte ≈ 0.25, 远低于 ridge 38）—— **瓶颈不是 GPU
空闲, 而是 GPU 在跑算力效率极低的 kernel**。这一个视角统一解释了:
- torch.compile +26%（融合 elementwise, §11/§13）
- model size / batch 变大 TFLOP/s 爬升（GEMM 占比升高, §14）
- naive attention 慢在 softmax/mask 链（elementwise）而非矩阵乘

(c) compile attention 的 kernel 对比 + (e) batch 1 vs 8 对比: 见下一节

## 18. Nsight 对照实验（2.1.4c/e）

**(c) attention N=4096 B=8 fp32, naive vs compile（23 次 fwd + 23 次 bwd 的 kernel 累计）**:

| 指标 | naive | compile |
|---|---|---|
| fwd ms/次 | 9.19 | **4.31（2.1×）** |
| kernel 总数 | 861 | 1310（更多!） |
| eager elementwise 累计 | 588.7ms（65%） | **7.6ms（~0%）** |
| triton fused softmax 链 | - | 347.1ms |
| GEMM 累计 | 280.6ms | 273.5ms（不变, GEMM 无法再融合） |
| mem_MB/次 | 2048.4 | 1032.5（减半） |

解读: compile 的本质不是"kernel 变少"（反而多了）, 而是**把 softmax/mask 的
逐元素链（5 次全量 N² 读写）融成 4 个 triton kernel（2 次读写）**——
省的是显存流量, 与 §11 的显存减半一致。GEMM 时间纹丝不动（280.6→273.5ms）,
因为 GEMM 本来就是单一算子无可融合; **提速上限 = 消灭 elementwise**:
9.19 - (588.7-347)/23 ≈ 4.4ms ≈ 实测 4.31ms ✓ 定量吻合

**(e) batch 1 vs 8（small full step fp32）**:

| 指标 | b=1 | b=8 |
|---|---|---|
| 墙钟/步 | 110.0ms | 303.0ms |
| **kernel 覆盖率** | **57.4%（空闲 42.6%!）** | **97.9%（空闲 2.1%）** |
| kernel 数/步 | ~3680 | ~3660（几乎相同） |
| 平均单 kernel | ~17µs | ~81µs |
| TFLOP/s | 3.87 | 11.18 |

解读: kernel 数量与 batch 无关 → b=1 时每个 kernel 只有 1/8 的工作量,
但 launch 间隔/调度开销不变 → **GPU 42.6% 时间在等活干**（launch-bound）。
这解释了 §8.7 batch 扫描 TFLOP/s 爬升和 §14 的 size 爬升: 都是"同样的
kernel 数, 更多的每-kernel 功作量"。

## 19. all-reduce 通信 benchmark（gloo 单机多进程模拟）

**原始数据**（20 次均值±std, bus_GB/s = 2(n-1)/n × alg）:

| world | 1MB ms (bus GB/s) | 10MB | 100MB | 1000MB |
|---|---|---|---|---|
| 2 | 1.30 (0.75) | 8.95 (1.09) | 82.0 (1.19) | 817.9 (1.19) |
| 4 | 2.29 (0.64) | 13.4 (1.09) | 154.3 (0.95) | 1274.7 (1.15) |
| 6 | 3.04 (0.54) | 17.4 (0.94) | 196.6 (0.83) | 1905.4 (0.85) |

**结论**:
1. **大消息带宽受限**: t ∝ S ✓（w2: 10→100→1000MB 时间精确 ×10/×10）,
   饱和带宽 ≈ 1.19 GB/s
2. **小消息延迟受限**: 1MB 时 alg GB/s 掉到 0.32-0.75——延迟项 2(n-1)·α 主导,
   且随 world 线性增长（1.30→2.29→3.04ms ✓）。
   **这正是 DDP 里 gradient bucket 存在的原因**（把小梯度聚成桶摊薄延迟）
3. ring 的"总线带宽与卡数无关"预言: NCCL 下成立; gloo 下 w4/w6 略降
   （0.85-1.15）——每 rank 的 CPU 转发与主机内存竞争有额外开销, 部分成立
4. **绝对值局限**: gloo CUDA 路径 = GPU→CPU→共享内存 ring→CPU→GPU,
   实测 1.2 GB/s 远低于真 NCCL（PCIe P2P ~16GB/s, NVLink 更高）。
   但**流量公式 2(n-1)/n·S、延迟结构、bucket 动机**全部得到验证——
   定性结论可迁移, writeup 声明即可
