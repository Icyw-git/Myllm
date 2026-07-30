## Discovery · 2026-07-29 · A1 Transformer LM / §7.3（接续 BPE 轮）

### Context
- claim / hypothesis: §7.3 架构消融与 TinyStories 训练已有跑次，但面试弹药可能停在「跑过」而非「能指着 Δvalid_loss 解释组件贡献 / 长上下文 RoPE / 日程与 Adam 分工」。
- implementation notes:
  - 消融入口: `scripts/run_ablation_train.py` + `scripts/ablation_model.py`
  - 已有 `Myllm-runs/experiments/ablation73/{baseline,no_rmsnorm_*,post_norm,no_rope,silu_ffn}`，约 8k steps
  - 末步粗表（valid_loss @8k）:
    | variant | valid_loss | valid_ppl |
    | --- | --- | --- |
    | baseline | 1.597 | 4.94 |
    | post_norm | 1.692 | 5.43 |
    | silu_ffn | 1.691 | 5.42 |
    | no_rope | 1.710 | 5.53 |
    | no_rmsnorm @3e-4 | 2.073 | 7.95 |
    | no_rmsnorm @1e-4 | 2.825 | 16.9 |
  - 另有 scaling/ctx/rope 等 checkpoint 目录（本轮 Discover 先不默认重训）
  - 上一轮 BPE Q1/Q2 已收；本轮**换模块焦点 → LM / 消融**
- skip_discover: false
- Obsidian: off（沿用上轮，除非你改口）

### Blindspot candidates (3–5)
1. **§7.3 表怎么讲**：五个变体里谁伤最重？`no_rmsnorm` 换 lr 后仍差，说明是「欠训」还是「缺归一化」？你能用自己的 valid_loss 差口述吗？
2. **RoPE 只在短 ctx 上看似小伤**：`no_rope` Δ≈0.11 vs baseline；若把 **eval context 拉长**（或对照 `rope500k` / `ctx128` 跑次），差距是否放大？面试常打「短序列看不出位置编码」。
3. **pre vs post norm**：你 post_norm 略差；训练稳定性（早期 loss 爆炸/梯度）有没有日志证据，还是只有终点一张表？
4. **SwiGLU vs SiLU**：参数量 / d_ff 约定是否公平？`silu_ffn` 的 d_ff=4d 而 SwiGLU 用 8/3 d——Δloss 有多少来自容量而非激活？
5. **Cosine schedule vs AdamW 自适应**：概念上你笔记写过「叠加」；有没有 **恒定 lr vs cosine** 的对照曲线可指？

### Selected
- Q1 ← candidate #1（§7.3 Δ 表）
- Q2 ← candidate #2（RoPE × eval context）

### Cards（预写；选定后进 Freeze）

#### Q1 · P5 · maps 候选 #1
- 追问问题: 指着 §7.3 表：去掉 RMSNorm / RoPE / 改 post-norm / 换 SiLU，valid_loss 各变多少？哪一个最伤？`no_rmsnorm` 降 lr 后仍差说明什么？
- 为什么这是个好问题: 作业核心消融；只会报「跑过 baseline」不够。
- 预期答案方向: 引用上表 Δ；归一化伤害 ≫ 其余；lr 扫描不能把 no_rmsnorm 救回 baseline。
- what_it_tests: 组件贡献是否能量化口述。
- expected_if_matters: 排序大致 no_rmsnorm ≫ no_rope ≈ post/silu > 0。
- 最小实验方案: **解析已有** `ablation73/*/log.jsonl` → 汇总 CSV（终值 + 可选 step 曲线关键点）；ETA ≤30m；一般**不必重训**。
- 验收标准: 一张 Δ 表 + 能解释 no_rmsnorm 双 lr。
- overnight_worthy: yes（短；弹药硬）
- suggested_knobs: step_checkpoint ∈ {2k,4k,8k}

#### Q2 · P5 · maps 候选 #2
- 追问问题: 短上下文上 `no_rope` 只差 ~0.1 valid_loss，能否说 RoPE 不重要？加长 eval/ctx 后会怎样？
- 为什么这是个好问题: 打「短序列假阴性」。
- 预期答案方向: 位置依赖随距离变强；应有更长 ctx 的对比或引用已有 ctx/rope 跑次。
- what_it_tests: RoPE 主张的证据是否完备。
- expected_if_matters: 长 ctx 下 no_rope 相对 baseline 的 gap 变大（或生成明显错位）。
- 最小实验方案: 优先 **读已有** `ctx128_*` / `rope500k_*` 日志做对照；若不足则 `run_ablation_train --variant no_rope` 与 baseline 在更长 `context_length` 各跑短步（需 Freeze 写死 steps/ctx）；ETA 1–4h。
- 验收标准: 至少一张「短 vs 较长 ctx」的 gap 表。
- overnight_worthy: yes
- suggested_knobs: context_length ∈ {128,256,512}；steps 与 baseline 对齐

#### Q3 · P4 · maps 候选 #4
- 追问问题: `silu_ffn` 与 baseline 的 Δ 有多少来自激活，多少来自 d_ff/参数量不匹配？
- 为什么这是个好问题: 消融公平性；面试爱问。
- 预期答案方向: 先报参数量；若未匹配容量，结论要降级为「联合改动」。
- what_it_tests: 消融是否 confounded。
- expected_if_matters: 参数量不同则不能单说「SiLU 更差」。
- 最小实验方案: 从 manifest/ckpt 读 n_params；可选匹配 param 的 silu 重跑（overnight）。
- 验收标准: 参数量表 + 一句公平性声明。
- overnight_worthy: partial
- suggested_knobs: match_params yes/no

### Overnight shortlist（推荐）
| Q | overnight_worthy | why_P1 | rough_ETA |
| --- | --- | --- | --- |
| Q1 | yes | 已有数，缺的是可口述 Δ 表 | ≤30m |
| Q2 | yes | RoPE 短序假阴性，面试高频 | 1–4h |
| Q3 | partial | 公平性补强 | ≤1h 或通宵重跑 |

### Handoff to Freeze
- proposed metric: `valid_loss`（lower better）；可选 `valid_ppl`
- proposed eval: 解析 `log.jsonl` 终值；Q2 另锁 context_length
- editable_scope: `experiments/eval_ablation_*.py`；默认不改 `cs336_basics/linear.py`
- forbidden: 中途改 valid 定义 / 换数据划分；Obsidian off
