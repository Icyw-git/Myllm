## Discovery · 2026-07-29 · A1 再挖（超参 / 优化 / 过拟合）

### Context
- 已收: BPE 压缩+affected；§7.3 组件；RoPE eval/train×ctx；silu 公平性；post 稳定性；L/d scaling；encode 878×
- 未吃透但已有跑次（相对 `tinystories_ba` valid≈**1.482** @20k）:
  | run | knobs | valid_loss |
  | --- | --- | --- |
  | lr1 | max_lr=1e-4 | 1.617 |
  | tinystories_ba | 3e-4 | 1.482 |
  | lr6 | 6e-4 | 1.451 |
  | warmup500_20k | warmup=500 | 1.487 |
  | bs16_20k | bs=16 | 1.567 |
  | bs64_20k | bs=64 | 1.383 |
  | wd0_20k | wd=0 | 1.485 |
  | rope500k_20k | θ=5e5 | 1.481 |
- 另有 `overfit_*`；尚无干净的 **恒定 lr vs cosine** 对照（笔记概念缺口）
- Obsidian: off

### Blindspot candidates (3–5)
1. **学习率扫描**：1e-4 / 3e-4 / 6e-4 的 valid 怎么讲？更高 lr 一直更好吗？和 warmup/cosine 怎么叠？
2. **batch size**：bs16→64 valid 变好，是「大 batch 更好」还是 **同 step 下 token 预算更大**？如何公平比？
3. **weight decay / RoPE θ**：wd0、rope500k 几乎打平 ba——你还敢说它们「很关键」吗？证据边界在哪？
4. **过拟合套件**：`overfit_*` 能否在几步内把 train loss 压到近 0？证明优化器/数据管线没写炸？
5. **cosine vs 恒定 lr**：概念上 schedule×Adam；缺一条恒定 lr 曲线就仍是口头知识。

### Selected
1–5 全部（用户「全部」）

### Cards（预写）

#### Q1 · P5 · LR
- 追问: 指着 1.617 / 1.482 / 1.451，你的 lr 怎么选？
- 实验: 汇总已有三跑 CSV（≤15m）；可选再扫 1e-3 看是否崩溃（overnight）
- overnight_worthy: yes（表立刻；外推可选）

#### Q2 · P5 · Batch / token 公平
- 追问: bs64 更低 loss，是不是证明大 batch 优越？
- 预期方向: 报 tokens_seen；按 token 或按 wall 对齐再比
- 实验: 从 log 抽 tokens_seen vs valid；或短跑 token-matched bs（overnight）
- overnight_worthy: yes

#### Q3 · P4 · 负结果（wd / θ）
- 追问: wd=0 与 θ=5e5 几乎无差，消融「没效果」怎么讲？
- 实验: 一张负结果表 + 适用范围（TinyStories 20k）
- overnight_worthy: yes（短）

#### Q4 · P4 · Overfit
- 追问: 你怎么证明训练代码没写错？
- 实验: 读/跑 overfit 日志，报 step→train_loss
- overnight_worthy: partial

#### Q5 · P5 · 恒定 lr
- 追问: cosine 关掉会怎样？
- 实验: 新跑 constant lr=3e-4 vs ba cosine（通宵 P1）
- overnight_worthy: yes

### Overnight shortlist
| 推荐 | 理由 |
| --- | --- |
| 1+2 | 超参面试最高频；已有数，缺公平叙事 |
| 2+5 | batch 公平 + cosine 因果 |
| 3+4 | 负结果 + 冒烟证明，快 |
