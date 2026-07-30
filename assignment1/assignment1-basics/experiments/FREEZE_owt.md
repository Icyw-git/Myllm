## Freeze · 2026-07-29 · A1 OWT LM（用户选 Q1+Q2+Q3）

- claim: 同 TinyStories 架构/步数在 OWT 上可交出 §7.4 曲线与生成；跨域 loss 不可裸比；OWT 上可能需轻量重调 lr/bs。
- hypothesis:
  - Q1: `owt_ba_20k` valid ≫ `tinystories_ba` 1.482；曲线仍降；归因数据难度/熵，非「同尺变笨」。
  - Q2: 相对 owt_ba，lr/bs/bs×lr 至少一格有可见 Δ，或证明「迁移优先于细调」。
  - Q3: 同解码设置下 OWT 生成流畅度明显弱于 TS；与高 valid 同向但不等于。
- discovery_refs: `DISCOVERY_owt.md` Q1, Q2, Q3（**Q0 tokenize 为阻塞前置，必先完成**）
- metric + direction:
  - 主 metric: OWT `valid_loss` **越低越好**（仅 OWT valid 内比）
  - 辅: `data/tokens_seen`、墙钟；跨域只并列报 TS 1.482 + 不可比免责
  - Q3: 定性样例（非刷分）
- eval_command（通宵中不可改）:
  - Q0: tokenize → 固定产物路径（见 data_version）
  - Q1/Q2: `scripts/run_train.py` 下表协议；解析各 `out-dir/log.jsonl` 末条 `valid_loss`
  - 汇总（跑完后）: `experiments/results/owt/q1_baseline.csv` / `q2_hparam_grid.csv`
  - Q3: `scripts/run_generate.py` + 固定 prompt；出样例表（Harvest）
- data_version / split:
  - tokenizer: `/data1/wcz/projects/Myllm-runs/tokenizers/openwebtext-v2`（vocab=32000）
  - train: `/data1/wcz/datasets/myllm/tokenized/owt_train.npy`（uint16）
  - valid: `/data1/wcz/datasets/myllm/tokenized/owt_valid.npy`（uint16）
  - 对照锚点（只读）: `tinystories_ba` valid≈1.482
- editable_scope:
  - 启动参数 / out-dir / SwanLab env
  - `scripts/run_tokenize.py` 的 `--corpus` / 路径覆盖（已支持 `owt`）
  - 不改 `cs336_basics/`
- forbidden_edits:
  - valid 定义 / 换 split 刷分
  - 通宵中改 steps、metric、eval 协议
  - DDP / 改动训练数值语义刷墙钟
  - 用 TS 权重冒充 OWT
- budget_wall_clock: 一晚（Q0 tokenize 数小时 + Q1/Q2 并行 20k）
- wall_clock_per_trial: ~0.5–2h / 20k 单卡（vocab 32k 可能偏慢）
- stop_condition:
  - Q0: 两 npy + meta 落盘
  - Q1/Q2: 各 run `log.jsonl` 有 step≈20000 的 valid；不中途改超参重开
  - Q3: baseline ckpt 后短测完成（可次日）
- expected_output_shape:
  - `Myllm-runs/experiments/owt_{ba,lr6e4,bs64,bs64_lr6e4}_20k/log.jsonl`
  - `experiments/results/owt/q1_baseline.csv` | `q2_hparam_grid.csv`
  - Q3 样例 markdown/csv
- git_branch / baseline_commit: `dff9a77`（作业树；跑前可再记）
- planner: **skipped**（旋钮已由 Discover 写死；不做 autoresearch 改代码环）
- Obsidian: off

### Q0 前置（阻塞）

`run_tokenize.py` 已支持 `--corpus owt`（默认 vocab=`openwebtext-v2`，输出 `owt_{split}.npy`）。

目标命令:

```bash
cd /data1/wcz/projects/Myllm/assignment1/assignment1-basics
# 先 valid 估吞吐，再 train
uv run --no-sync python scripts/run_tokenize.py --corpus owt --splits valid
uv run --no-sync python scripts/run_tokenize.py --corpus owt --splits train
```

### Q1 / Q2 并行表（一卡一进程）

公共协议（相对 `tinystories_ba`，仅 data + vocab 必变）:

| 项 | 值 |
| --- | --- |
| arch | L4 d512 h16 d_ff1344 ctx256 |
| steps | 20000 |
| warmup | 200 |
| vocab | 32000 |
| seed | 0 |

**SwanLab（四跑统一）**

| 项 | 值 |
| --- | --- |
| project | **`cs336-owt`**（覆盖 `.env` 的 tinystories 项目，避免混盘） |
| workspace / mode | 沿用 `.env`（`07011812138` / cloud） |
| group | `owt-hparam`（同屏对比） |
| exp | = `run_id`（`owt_ba_20k` 等） |
| tags | 自动 `cs336,owt` + `SWANLAB_TAGS`（q1/q2…） |
| description | `CS336 A1 OpenWebText · maps_to Q1/Q2` |
| 指标键 | `train/loss` `train/lr` `valid/loss` `time/wall_s` `data/tokens_seen` |

| GPU | run_id | 相对 owt_ba | maps_to |
| --- | --- | --- | --- |
| 0 | `owt_ba_20k` | max_lr=3e-4 min_lr=3e-5 bs=32 | Q1 |
| 1 | `owt_lr6e4_20k` | max_lr=6e-4 min_lr=6e-5 bs=32 | Q2 |
| 2 | `owt_bs64_20k` | lr=3e-4 bs=64 | Q2 |
| 3 | `owt_bs64_lr6e4_20k` | lr=6e-4 bs=64 | Q2 |

模板（替换 `GPU` / `RUN` / lr / bs）:

```bash
CUDA_VISIBLE_DEVICES=0 \
SWANLAB_PROJ_NAME=cs336-owt SWANLAB_GROUP=owt-hparam SWANLAB_EXP_NAME=owt_ba_20k \
SWANLAB_TAGS=q1,ba-protocol,baseline SWANLAB_DATASET_TAG=owt \
SWANLAB_DESCRIPTION='CS336 A1 OpenWebText · Q1/Q2' \
uv run --no-sync python scripts/run_train.py \
  --train-data /data1/wcz/datasets/myllm/tokenized/owt_train.npy \
  --valid-data /data1/wcz/datasets/myllm/tokenized/owt_valid.npy \
  --vocab-size 32000 \
  --steps 20000 --batch-size 32 --max-lr 3e-4 --min-lr 3e-5 --warmup 200 \
  --context-length 256 --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 \
  --out-dir /data1/wcz/projects/Myllm-runs/experiments/owt_ba_20k
```

### Q3（Harvest，不占通宵 P1 墙钟）

- ckpt: `owt_ba_20k` 最终 ckpt
- tokenizer: `openwebtext-v2`
- 对照: 已有 TinyStories 生成设定（同 T/top-p）
- 验收: ≥3 条并排样例 + 一句机制方向

### registry 计划行（status=planned → running → done）

`owt_tokenize`, `owt_ba_20k`, `owt_lr6e4_20k`, `owt_bs64_20k`, `owt_bs64_lr6e4_20k`（notes 带 `maps_to=Q0|Q1|Q2`）

### Phase 2 Stop

```text
eval / metric / SwanLab 已冻结。流水线已启动：tmux `owt-overnight`（Q0→四卡 Q1/Q2）。
```
