## Discovery · 2026-07-29 · A1 续挖（BPE + §7.3 已收之后）

### Context
- 已收弹药:
  - BPE: 跨域 bytes/token；merge `affected` 曲线
  - LM: §7.3 Δ 表；RoPE×eval-ctx（64–256 Δ≈0.11 平坦；512 外推不可用）
- claim / hypothesis（本轮）: 仍有「公平性 / 稳定性 / 规模 / encode」类追问缺口，不能只靠终值一张消融表。
- 已知线索:
  - `baseline` params≈**22.70M**；`silu_ffn`≈**19.94M**（更少参数，valid 仍差 ~0.09）
  - post_norm 早期 valid 略高于 baseline，未见「炸训练」级别证据
- Obsidian: off

### Blindspot candidates (3–5)
1. **SiLU 消融是否公平**：少了 ~2.8M 参数还差 0.09，结论是「激活更差」还是「容量不够」？要不要做 **匹配参数量** 的 silu（调 d_ff）？
2. **pre vs post 稳定性**：终点只差 0.09；早期曲线有没有「post 更不稳」？若没有，面试还能怎么说 post-norm？
3. **深度/宽度 scaling**：你有 L2/L6、d384/d768 等目录的话，valid_loss 是否近似按参数/计算量可讲？缺表就是缺口。
4. **encode 吞吐**：作业 `Tokenizer.encode` vs `FastEncoder` 在 TinyStories/OWT 上差几个数量级？（上一轮 BPE Discover Q3）
5. **干净的长 ctx RoPE**：在 **训练 ctx=512** 下重跑 baseline vs no_rope（避免外推混淆）——上一轮假说未测到。

### Selected
- Q1 ← #1 SiLU 公平性
- Q2 ← #2 pre/post 稳定性
- Q3 ← #3 L/d scaling 表
- Q4 ← #4 encode 吞吐
- Q5 ← #5 train+eval ctx=512 RoPE（通宵 P1）

### Cards（预写）

#### Q1 · P5 · 候选 #1
- 追问问题: silu_ffn 参数更少却只差 0.09，你的「SiLU 不如 SwiGLU」结论站得住吗？
- 预期答案方向: 先报 22.7M vs 19.9M；要么承认 confound，要么补匹配容量跑。
- 最小实验: 读 manifest 出参数表（已有）；可选 `--d-ff` 抬到接近 baseline 参数再短训/全训。
- overnight_worthy: yes（表立刻有；匹配重跑 1–2h）
- suggested_knobs: d_ff；steps=8000 对齐

#### Q2 · P4 · 候选 #2
- 追问问题: post-norm 相对 pre-norm，你有稳定性证据还是只有终点 Δ？
- 预期答案方向: 画/报 early valid；若早期也只是平行略差，则弱化「不稳定」叙事，改成「最终略差」。
- 最小实验: 解析 log.jsonl → early/mid/late 表（≤20m）。
- overnight_worthy: yes（短）

#### Q3 · P5 · 候选 #5
- 追问问题: 上一轮 eval-ctx 扫不清 RoPE×长度；若 **train+eval 都用 512**，no_rope gap 会怎样？
- 预期答案方向: 同预算 tokens，比 Δ；避开外推。
- 最小实验: `run_ablation_train --variant baseline/no_rope --context-length 512 --steps …`
- overnight_worthy: yes（通宵级）
- suggested_knobs: context_length=512；steps 与 8k 对齐或按 token 预算对齐

#### Q4 · P3 · 候选 #4
- 追问问题: encode 为什么慢？Fast 路径快多少？
- overnight_worthy: partial（≤1h）

### Overnight shortlist（推荐）
| Q | why | ETA |
| --- | --- | --- |
| Q1 公平性 | 消融面试高频；参数量已暴露 | 表≤15m；匹配重跑可选 |
| Q2 稳定性曲线 | 补强 post-norm 叙事 | ≤20m |
| Q3 训练态长 ctx | 上一轮未决假说 | 通宵 |
