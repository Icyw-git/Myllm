## Discovery · 2026-07-30 · A1 OWT 续挖（§7.4 后 → leaderboard / 容量）

### Context
- claim / hypothesis: §7.4 主实验与口述已硬（跨域 loss 不可裸比；OWT 需轻量重调；生成差归因域风格）。下一刀面试/作业常打 **§7.5 leaderboard**：45min 墙钟、只许 OWT train、naive 基线 valid&lt;5.0——你有 4.25/3.94，但 **还缺「墙钟预算叙事 + 还能怎么榨」**。
- implementation notes（已有）:
  | 资产 | 状态 |
  | --- | --- |
  | OWT npy | train 2.73B / valid 66M tok |
  | owt_ba_20k | valid **4.248** @20k / 1.64e8 tok / ~39min |
  | owt_bs64_lr6e4 | valid **3.939** @20k / 3.28e8 tok / ~77min（同 step 最优） |
  | **墙钟反推（已有 log，未重跑）** | 45min：ba 可跑满 20k→**4.248**；bs64_lr6 仅 ~11.7k step→**~4.066** |
  | token-matched | 组合格 @1.64e8 → **4.122** 仍优于 ba |
  | 生成 Q3 | TS vs OWT 样例已有 |
  | 未动 | L/d/ctx 放大；&gt;20k steps；weight tying；真正「45min 墙钟」协议跑 |
  | 硬件 | 8×4090 空闲；多卡=独立 job 并行 |
- 已收勿重复: TS 超参表、§7.3、OWT ba/hparam/生成口述
- handout §7.5: ≤45min B200；只 OWT train；交 wall-clock 横轴曲线；建议可试 weight tying（先小集试）
- skip_discover: false · Obsidian: off
- AGENTS: 本轮 P1 **优先现有 `run_train.py` 旋钮**（L/d/bs/lr/steps/ctx）；架构改动（tying 等）标为需你先改代码，agent 不代写核心实现

### Blindspot candidates (3–5)
1. **墙钟协议空洞**：leaderboard 要 **wall-clock&lt;45min** 曲线；你表全是 20k step。已有 log 反推：**ba 39min 跑满 20k；bs64_lr6 45min 只到 ~11.7k step**——同 step 最优（3.939）在预算内够不着。
2. **同 step 最优 ≠ 墙钟最优**：20k 表说 bs64_lr6 赢 0.31；45min 截断后差距缩到 **4.066 vs 4.248**（仍赢，但叙事要换）。面试会问你怎么交横轴。
3. **容量还没在 OWT 上动过**：TS 上 L6/d768 有收益；OWT 挂 bs64+lr6e-4 加 L/d 还值吗？
4. **Weight tying（handout 点名）**：需改代码；短训是否降 valid？
5. **生成是否随 valid 变好**：3.939 ckpt vs 4.248 样例对比（短测）

### Selected（按面试杀伤力；**待你确认进 Freeze**）
- **Q1** ← #1+#2 墙钟预算（**可从已有 log 先口述一版；正式 Freeze 仍建议各跑一截 45min 留 SwanLab 曲线**）
- **Q2** ← #3 容量 L6（P1）；d768 可选 P2
- **Q3** 并入 Q1（ba vs bs64_lr6 同墙钟）
- Q4/Q5 不进通宵 P1（tying 阻塞 / 生成短测）

### Cards

#### Q1 · P5 · 45min 墙钟预算（leaderboard 口径）
- 追问问题: 你的最优 OWT 在 **45 分钟墙钟**里 valid 多少？同 step 表还能用吗？
- 为什么这是个好问题: §7.5 硬约束；**已有 log 已能预演翻转**（见下）。
- 预期答案方向:
  - ba：~8.5 step/s，39min 跑满 20k → **4.248**；45min 可略超 20k
  - bs64_lr6：~4.3 step/s，45min 仅 **~11.7k step** → valid **~4.066**（非 3.939）
  - 墙钟下仍赢 ba，但 **Δ 从 0.31 缩到 0.18**；3.939 是「跑满 77min」的数字
  - 4090≠B200，本地报实测 wall + 免责
- what_it_tests: step 协议 vs 墙钟协议；leaderboard 叙事。
- expected_if_matters: 需重画 **wall_s 横轴** 曲线；可能改 leaderboard 配方（偏吞吐）。
- 最小实验方案:
  - **Phase A（≤30m）**：从已有 `log.jsonl` 出 `q1_wallclock_derived.csv`（截断 @2700s）— 可立刻口述
  - **Phase B（正式）**：两跑 `steps` 设上限使 wall≈2700s（ba ~23k / bs64_lr6 ~12k）或 `timeout 2700` 包训练
  | GPU | run | 设定 |
  | --- | --- | --- |
  | 0 | `owt_wc45_ba` | bs32 lr3e-4，wall≤45min |
  | 1 | `owt_wc45_bs64_lr6` | bs64 lr6e-4，wall≤45min |
- 验收标准: `results/owt2/q1_wallclock*.csv` + SwanLab 时间轴截图
- overnight_worthy: yes（Phase B）
- suggested_knobs: steps cap / wall stop；不改 valid

#### Q2 · P5 · OWT 容量一刀（L 或 d，挂最优超参）
- 追问问题: OWT 上只调了 lr/bs——加层/加宽还有收益吗？还是数据难到「超参到顶」？
- 为什么这是个好问题: 衔接 TS scaling 弹药；证明域迁移后容量假设要重测。
- 预期答案方向: 相对 `owt_bs64_lr6e4` 报 Δ；承认未做全因子。
- what_it_tests: 容量 × 难域交互。
- expected_if_matters: L6 或 d768（d_ff=⌊8/3 d⌋）再降 valid ⇒ 容量仍有空间；打平/变差 ⇒ 「先榨吞吐/步数」。
- 最小实验方案（与 Q1 可同晚不同卡）:
  | GPU | run | 改动 | 对照 |
  | --- | --- | --- | --- |
  | 2 | `owt_L6_bs64_lr6e4_20k` | L=6，其余同最优超参，20k | vs 3.939 |
  | 3 | `owt_d768_dff2048_bs64_lr6e4_20k`（可选 P2） | d=768 d_ff=2048 | 对齐 lm4 d_ff |
  - ETA: ~1–1.5h/卡（bs64 更大模型可能更慢）
- 验收标准: `q2_capacity.csv`；口述 Δ 与显存/墙钟代价
- overnight_worthy: yes
- suggested_knobs: num_layers, d_model, d_ff（跟 8/3）

#### Q3 · P4 · 墙钟公平：多 step×小 bs vs 少 step×大 bs
- 追问问题: 同 45min 里，bs32 多迭代是否打得过 bs64？
- 为什么这是个好问题: 把 lm3/lm4「token 公平」升级成 leaderboard「时间公平」。
- 预期答案方向: 两格同墙钟比 valid；吞吐（tok/s）决定谁吃更多数据。
- what_it_tests: 时间预算下的 batch 选择。
- expected_if_matters: 可能与同 step 结论不一致 → 面试加分点。
- 最小实验方案: 并入 Q1 两跑即可（ba vs bs64_lr6 同墙钟）；不必再开第三协议除非 Q1 不够看。
- 验收标准: 同 `q1_wallclock.csv` 加 tok/s 列
- overnight_worthy: yes（依附 Q1）
- suggested_knobs: batch_size, max_lr

#### Q4 · P4 · Weight tying（需改代码）
- 追问问题: handout 点名的 embedding–LM head tying，在你的 OWT 短训上有用吗？
- 为什么这是个好问题: leaderboard 经典招；也考 init std 是否跟着改。
- 预期答案方向: 有/无 Δ + 参数量变化；提 init 注意事项（方向即可）。
- what_it_tests: 架构改动 vs 纯超参。
- expected_if_matters: 降 valid 或加速收敛 ⇒ 进 leaderboard 配方；无收益 ⇒ 「点名招不万能」。
- 最小实验方案: **你先改** `TransformerLM`/训练脚手架支持 tying → 再短训对照 ba 或最优超参（≥1 跑）
- 验收标准: 对照表 + 参数量
- overnight_worthy: **条件 yes**（代码就绪后）
- suggested_knobs: tie_embeddings flag；embedding init std

#### Q5 · P3 · 更好 ckpt 生成是否好转（短测）
- 追问问题: valid 3.94 的模型生成是否明显好于 4.25？
- 预期答案方向: 同 decode 并排；可能仍「新闻腔」——强化域风格故事。
- 最小实验方案: `eval_owt_q3_generate.py` 换 ckpt 路径（≤20m）
- overnight_worthy: no
- suggested_knobs: ckpt 选择

### Overnight shortlist
| 推荐组合 | 内容 | 理由 |
| --- | --- | --- |
| **A（默认）** | Q1+Q3 墙钟两跑 + Q2 L6 一跑 | leaderboard 口径 + 容量；3–4 卡一晚 |
| B | Q2 两跑（L6+d768）跳过墙钟 | 只挖容量 |
| C | Q4 tying（你改代码后） | handout 点名，但阻塞在实现 |
| D | 只 Q5 | 不进通宵 |

### Handoff to Freeze（选定后填）
- proposed metric + direction: `valid_loss`↓；墙钟题额外记录 `wall_s`（**不替代** valid 作主 metric）
- proposed eval_command: 解析 `log.jsonl` 末 valid + wall；墙钟跑可用预先 step cap 或外部 timeout
- editable_scope hint: `scripts/run_train.py` 启动参数；tying 仅当用户授权改脚手架
- forbidden: 改 valid 定义 / 换数据；通宵改 metric；把 4090 分钟直接宣称等于 B200
- SwanLab: project=`cs336-owt`；新 group 建议 `owt2-leaderboard` 或 `owt2-capacity`
