## Discovery · 2026-07-29 · A1 再挖（lm4：warmup / 缩放律 / 生成）

### Context
- claim / hypothesis: BPE、§7.3、RoPE×ctx、超参表、cosine vs const 已有弹药；面试下一刀常打 **warmup 是否必要、lr×batch 缩放、复合放大怎么归因、生成采样会不会讲**。
- implementation notes（已有未吃透跑次）:
  | run | knobs | valid@20k |
  | --- | --- | --- |
  | tinystories_ba | warm=200, lr=3e-4, bs=32 | **1.482** |
  | warmu | **warmup=0** | **1.489** |
  | lr6 | lr=6e-4, bs=32 | 1.451 |
  | bs64_20k | lr=3e-4, bs=64 | 1.383 |
  | bs64_lr6e4_20k | lr=6e-4, bs=64 | **1.347** |
  | L6_20k / d768_20k | 单轴放大 | 1.430 / 1.435 |
  | L6_d768_bs64_lr6e4_20k | L6+d768+bs64+lr6e-4 | **1.263** |
  | const_lr_3e4_20k | 恒定 lr | 1.502 |
- 生成入口已有: `scripts/run_generate.py`（T / top-k / top-p）
- 已收勿重复: §7.3 组件表、encode 878×、token-matched bs、wd/θ 负结果、cosine vs const
- skip_discover: false · Obsidian: off

### Blindspot candidates (3–5)
1. **Warmup 必要吗**：`warmu`(0) 1.489 vs ba(200) 1.482——几乎打平。你还能背「必须 warmup」吗？早期曲线有没有差点炸？
2. **lr × batch 线性缩放**：2×bs 该不该 2×lr？对比 ba / bs64 / lr6 / bs64_lr6e4；token 预算仍纠缠时怎么讲？
3. **复合放大归因**：L6_d768_bs64_lr6e4→1.263，相对单轴 L6/d768，增益来自哪？缺因子分解就只是「更大更好」。
4. **生成采样**：greedy vs T/top-p 文本差在哪？valid_loss 低是否等于故事可读？（作业常考，现有表几乎空白）
5. **宽度×d_ff 公平重跑**（lm2 遗留）：d384/d768 曾共用 d_ff=1344；按 8/3 d 对齐后再比，scaling 叙事是否翻盘？

### Selected
- Q1 ← #1 Warmup
- Q2 ← #2 lr×batch
- Q3 ← #3 复合放大归因
- Q5 ← #5 d_ff 对齐 scaling（通宵 P1）
- （跳过 Q4 生成）

### Cards（预写）

#### Q1 · P5 · Warmup
- 追问问题: warmup=0 几乎打平 ba，warmup 还重要吗？
- 为什么这是个好问题: 教材口头禅 vs 你自己的 TinyStories 证据。
- 预期答案方向: 报 1.489 vs 1.482；谈 **适用边界**（本设定温和）；可看 early train/valid 有无尖峰。
- what_it_tests: 负结果/弱效应怎么讲，而不是死背必须 warmup。
- expected_if_matters: Δ很小 ⇒ 「本预算下非关键」；若 early 抖 ⇒ 「保早期稳定」。
- 最小实验方案: 解析 `warmu` vs `tinystories_ba` early 曲线（≤20m）；一般不必重训。
- 验收标准: 一张 early/mid/late + 终值 Δ 表。
- overnight_worthy: no（短测）；除非补「更大 lr + no warmup」压测。
- suggested_knobs: warmup∈{0,200}；可选 max_lr=6e-4×warmup=0

#### Q2 · P5 · lr×batch
- 追问问题: bs 加倍时 lr 该不该加倍？指着四格表怎么说？
- 为什么这是个好问题: 经典线性缩放；和 lm3「token 公平」叠在一起最容易被追问穿。
- 预期答案方向: 同 step 下 bs64_lr6e4 最好；但须同时报 tokens；是否「线性缩放成立」要限定同 step 或同 token。
- what_it_tests: 多旋钮交互 + 公平对比意识。
- expected_if_matters: bs64_lr6e4 < min(bs64, lr6) 同 step ⇒ 有正交互迹象；token 对齐后可能减弱。
- 最小实验方案: 从四跑 log 出 2×2 表 + token-matched 切片（≤30m）。
- 验收标准: `q2_lr_batch_grid.csv` + 一句公平性免责。
- overnight_worthy: yes（表立刻；可选 token-matched 补跑）。
- suggested_knobs: batch_size, max_lr

#### Q3 · P5 · 复合放大归因
- 追问问题: 1.263 的模型，你怎么证明不是「只加了 batch/lr」？
- 为什么这是个好问题: 面试官讨厌黑箱更大更好。
- 预期答案方向: 对照 L6、d768、bs64_lr6e4 逐步差分；承认未做全因子设计。
- what_it_tests: 归因 / 实验设计诚实度。
- expected_if_matters: 单轴增益有限，复合后进一步降；缺 leave-one-out 则说「上界/堆叠」而非精确分解。
- 最小实验方案: 汇总已有五跑差分表（≤30m）；通宵可选 leave-one-out 一刀（如 L6_d768 但 bs32/lr3e-4）。
- 验收标准: 归因表 + 「不能唯一分解」声明。
- overnight_worthy: yes（表短；干净 leave-one-out 通宵）。
- suggested_knobs: num_layers, d_model, batch_size, max_lr

#### Q4 · P4 · 生成采样
- 追问问题: temperature / top-p 你怎么调？greedy 为啥常更糟或更刻板？
- 为什么这是个好问题: 有 `run_generate.py` 却几乎零定性弹药。
- 预期答案方向: 同一 ckpt 出几条样例对比；不把 valid_loss 当成可读性。
- what_it_tests: 解码与训练指标的边界。
- expected_if_matters: T↑ 更发散；top-p 截尾；greedy 重复/呆板（用样例说话）。
- 最小实验方案: `run_generate.py` 扫 T∈{0,0.7,1.0} × top-p∈{1,0.9}（≤30m）。
- 验收标准: 样例表 + 一句机制方向（非散文生成比赛）。
- overnight_worthy: no（≤1h 轨 A 也可）；通宵不优先。
- suggested_knobs: temperature, top_k, top_p

#### Q5 · P4 · d_ff 对齐 scaling（遗留）
- 追问问题: 上次 d384/d768 的 d_ff 没跟宽度走，scaling 结论还算数吗？
- 预期答案方向: 承认 confound；对齐 8/3 d 后重比或降级主张。
- 最小实验方案: 重跑 d384/d768（或只跑缺的一侧）d_ff=⌊8/3 d⌋，20k 或短训对齐。
- overnight_worthy: yes
- suggested_knobs: d_model, d_ff

### Overnight shortlist
| 推荐 | 理由 |
| --- | --- |
| 2+3 | 缩放面试最高频；已有数，缺网格叙事/归因 |
| 1+2 | warmup 负结果 + lr×bs，快出表 |
| 4 | 生成空白，但 overnight_worthy=no → 更适轨 A |
| 5 | 还债 lm2；通宵级但偏「补洞」 |

### Handoff to Freeze（选定后填）
- proposed metric + direction: valid_loss（越低越好）；生成题用定性样例
- proposed eval_command: 解析已有 `log.jsonl`；生成用 `scripts/run_generate.py`
- editable_scope hint: 原则上不改 `cs336_basics/`；新跑走 `scripts/run_train.py`
- forbidden: 改 valid 定义 / 换数据划分刷分
