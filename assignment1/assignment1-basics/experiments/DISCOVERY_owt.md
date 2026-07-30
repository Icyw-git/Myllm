## Discovery · 2026-07-29 · A1 OWT LM（§7.4 main_experiment）

### Context
- claim / hypothesis: 作业要求 **同架构 + 同迭代数** 在 OpenWebText 上训 LM；相对 TinyStories，valid_loss 会明显更高，且生成更差——但面试刀口常在 **「loss 数字能否跨域直比」** 与 **「同算力为何更差」**，不是再刷 TinyStories 表。
- implementation notes:
  | 资产 | 状态 |
  | --- | --- |
  | OWT raw | `owt_train.txt` 12G / `owt_valid.txt` 277M |
  | OWT BPE 32k | **done** → `Myllm-runs/tokenizers/openwebtext-v2` |
  | OWT tokenized `.npy` | **缺失**（`tokenized/` 仅 TinyStories） |
  | OWT LM runs | **零**（`experiments/` 无 owt*） |
  | 训练入口 | `scripts/run_train.py`（单卡；默认 vocab=10k + TinyStories npy） |
  | 硬件 | 8×4090 全空 → **多卡 = 多独立 job 并行**（非 DDP；本阶段不引入分布式实现） |
  | TinyStories 对照锚点 | `tinystories_ba` valid≈**1.482** @20k；L4 d512 h16 ff1344 ctx256 bs32 lr3e-4 |
- handout 锚点: §7.4 `main_experiment` — 同 arch + 同 steps；交学习曲线 + 生成样例 + 解释为何更差。
- skip_discover: false · Obsidian: off

### Blindspot candidates (3–5)
1. **数据门槛还没过**：OWT LM 之前必须先有 uint16 memmap；没 tokenize 就谈「开训」是空转。吞吐/ETA 你估过吗？
2. **同协议迁移**：把 `tinystories_ba` 超参原样搬到 OWT（只换 data + vocab=32k），valid 会落在哪一档？相对 1.482 的 Δ 怎么口述才不算「跨域瞎比」？
3. **超参是否要重调**：handout 明示可能要调 lr/bs；TinyStories 上的 lr×bs 结论在 OWT 上还成立吗？
4. **同 step ≠ 同难度**：OWT 熵更高 / 重复更少 → 同 20k step 更差；还是因为 32k vocab / 更长文档把有效上下文吃穿了？
5. **生成交付**：同 compute 生成更差的机制（数据 vs 欠训 vs 解码）——目前零 OWT ckpt，属交付缺口。

### Selected（用户确认 2026-07-29：Q1+Q2+Q3）
- Q0 ← #1 tokenize 门槛（**前置阻塞**，必跑）
- Q1 ← #2 同协议 OWT baseline
- Q2 ← #3 超参小网格并行
- Q3 ← #5 生成对照（Harvest；overnight_worthy=no）
- （#4 并入 Q1 口述口径，不单独重训）
- Freeze: `experiments/FREEZE_owt.md`

### Cards

#### Q0 · P5 · Tokenize OWT（前置）
- 追问问题: 你要训 OWT LM，token ID 序列在哪？uint16 为何合适？全库 tokenize 要多久？
- 为什么这是个好问题: 作业 2.7(d) 显式要求；没 npy = 没实验。
- 预期答案方向: vocab≤32k ⇒ uint16 够用；报墙钟与 tok/s；train/valid 路径写死。
- what_it_tests: 数据管线是否闭环，而不是只会训 TinyStories。
- expected_if_matters: 缺文件 ⇒ 一切 OWT LM 卡 NO-GO。
- 最小实验方案:
  - `scripts/run_tokenize.py` 指到 `openwebtext-v2` + `owt_{train,valid}.txt` → `tokenized/owt_{train,valid}.npy`
  - 建议：先 valid（~277M）估吞吐，再开 train；ETA 粗估数小时量级（视 FastEncoder）
  - 并行：train 是大头，通常 **1 进程吃满 CPU**；GPU 闲置留给 Q1/Q2（tokenize 不必占 GPU）
- 验收标准: 两文件存在 + `dtype=uint16` + meta（vocab_dir、counts）；valid 可先 smoke encode 对齐。
- overnight_worthy: yes（墙钟长，但是 **阻塞项**）
- suggested_knobs: tokenizer_dir, out paths；可选 chunk/并行仅当自己已有分片脚本

#### Q1 · P5 · 同协议 OWT baseline
- 追问问题: 同 L4/d512/20k/bs32/lr3e-4，OWT valid 相对 TinyStories 1.482 差多少？这个差能直接说「模型变差」吗？
- 为什么这是个好问题: §7.4 主交付；面试最爱打「数字跨域可比性」。
- 预期答案方向: 报 OWT final valid + 曲线形态；解释应落在 **数据熵/多样性/任务难度**，不是「同一标尺上变笨了」的裸结论。
- what_it_tests: 同 compute 迁移叙事 + 公平对照意识。
- expected_if_matters: OWT valid ≫ 1.482；曲线仍下降但不像 TS 那么「好讲故事」。
- 最小实验方案:
  ```bash
  # GPU0 · 协议对齐 tinystories_ba，仅换数据与 vocab
  CUDA_VISIBLE_DEVICES=0 SWANLAB_GROUP=owt-main SWANLAB_EXP_NAME=owt_ba_20k \
  SWANLAB_TAGS=owt,baseline,ba-protocol \
  uv run --no-sync python scripts/run_train.py \
    --train-data /data1/wcz/datasets/myllm/tokenized/owt_train.npy \
    --valid-data /data1/wcz/datasets/myllm/tokenized/owt_valid.npy \
    --vocab-size 32000 \
    --steps 20000 --batch-size 32 --max-lr 3e-4 --min-lr 3e-5 --warmup 200 \
    --context-length 256 --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 \
    --out-dir /data1/wcz/projects/Myllm-runs/experiments/owt_ba_20k
  ```
  - ETA: 参考 TS 20k 单卡 ~0.5–1.5h（vocab 32k 略增算力/显存）；通宵裕度足够
- 验收标准: `log.jsonl` 终值 + 与 `tinystories_ba` 对照一行表；SwanLab 曲线可截。
- overnight_worthy: yes（P1 必跑）
- suggested_knobs: 冻结架构与 steps；只允许 data/vocab 与 TS 不同

#### Q2 · P4 · OWT 超参小网格（多卡并行）
- 追问问题: handout 说可能要重调 lr/bs——你在 OWT 上试了哪几格？相对 `owt_ba` 有无稳定增益？
- 为什么这是个好问题: 证明不是「只会抄 TinyStories 超参」；又和 lm3/lm4 的 lr×bs 弹药衔接。
- 预期答案方向: 2–3 个并行变体 vs ba；讲清同 step 对比；若增益小则说「迁移优先于细调」。
- what_it_tests: 域迁移后超参敏感性。
- expected_if_matters: 某一格明显低于 owt_ba ⇒ 有重调价值；否则「同协议已够交差，leaderboard 再动刀」。
- 最小实验方案（**多卡并行，一卡一 run**；Q0 完成后与 Q1 可同晚启动）:
  | GPU | run_id | 相对 ba 的改动 | 目的 |
  | --- | --- | --- | --- |
  | 0 | `owt_ba_20k` | （Q1） | 锚点 |
  | 1 | `owt_lr6e4_20k` | max_lr=6e-4, min_lr=6e-5 | 更高 lr |
  | 2 | `owt_bs64_20k` | batch_size=64 | 更大 bs（tokens/step×2） |
  | 3 | `owt_bs64_lr6e4_20k` | bs64 + lr6e-4 | 线性缩放一格 |
  | 4 | `owt_L6_20k`（可选 P2） | num_layers=6 | 容量一刀 |
  - 其余 GPU 预留失败重跑 / leaderboard 试探
  - 每 run：`SWANLAB_GROUP=owt-hparam`，`SWANLAB_EXP_NAME=<run_id>`，`out-dir=.../experiments/<run_id>`
  - ETA: 墙钟 ≈ 最慢单卡 20k（并行不叠加）；通宵 1 轮够
- 验收标准: `experiments/results/owt/q2_hparam_grid.csv`（final valid + tokens_seen）；口述「哪格赢、是否同 token」。
- overnight_worthy: yes（通宵 P1 网格）
- suggested_knobs: max_lr, min_lr, batch_size；（可选）num_layers

#### Q3 · P4 · 生成对照（交付短测）
- 追问问题: 同模型同步数，OWT 生成为何读起来更差？你有并排样例吗？
- 为什么这是个好问题: §7.4 第二交付；不能只交 loss 曲线。
- 预期答案方向: 同一解码设置下 TS vs OWT 样例；归因 **数据分布/欠拟合网页文**，避免只骂 temperature。
- what_it_tests: 指标 vs 可读性边界。
- expected_if_matters: OWT 样例更碎、更不像连贯叙事；与高 valid_loss 同向但不等于。
- 最小实验方案: Q1 ckpt 出来后 `scripts/run_generate.py`（换 OWT tokenizer + ckpt）；对照已有 TS 生成（≤30m）
- 验收标准: 各 ≥3 条样例表 + 一句机制方向。
- overnight_worthy: no（≤1h；挂在 Harvest）
- suggested_knobs: temperature, top_p；固定 prompt 集合

### Overnight shortlist
| Q | overnight_worthy | why_P1 | rough_ETA | GPU 策略 |
| --- | --- | --- | --- | --- |
| Q0 | yes | 阻塞一切 | tokenize 数小时（CPU） | 不占 GPU |
| Q1 | yes | §7.4 主交付 | ~1×20k 单卡 | GPU0 |
| Q2 | yes | 重调证据 / 抗追问 | 与 Q1 同墙钟 | GPU1–3（+可选4） |
| Q3 | no | 交付但短 | ≤30m | 任意空卡或 CPU 解码 |

推荐默认：`Q0 →（完成后）Q1+Q2 四卡并行`；Q3 收割时做。

### Handoff to Freeze
- proposed metric + direction: `valid_loss` 越低越好（**OWT valid 内比较**）；跨域只报并列数字 + 不可比免责
- proposed eval_command:
  - 训练：上表 `run_train.py` 固定协议
  - 汇总：`experiments/results/owt/*.csv`（Freeze 时写解析脚本路径）
  - 生成：`run_generate.py` + 固定 prompt 文件
- editable_scope hint: `scripts/run_tokenize.py` / `scripts/run_train.py` 启动参数与 out-dir；**不改** `cs336_basics/` 核心；本轮不做 DDP
- forbidden: 改 valid 定义 / 换数据划分刷分；通宵中途改 steps 或 metric；用 TinyStories 权重冒充 OWT
- budget_wall_clock 建议: 一晚（tokenize + 并行 20k×4）
- 多卡约定: **一卡一独立进程**（`CUDA_VISIBLE_DEVICES=k`）；不是 `torchrun` DDP
