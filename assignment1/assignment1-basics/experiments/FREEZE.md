## Freeze · 2026-07-29 · A1 Tokenizer / BPE

- claim: OWT 32k byte-BPE 已训完；可用压缩率跨域表 + 训练日志 profile 抗追问。
- hypothesis:
  - Q1: 同 tokenizer 上，OWT valid 的 bytes/token 优于 TinyStories；中文明显更差。
  - Q2: 单步墙钟与 `affected` 正相关；末期加速由 affected 变小驱动，而非 heap「突然变快」。
- discovery_refs: Q1, Q2（`experiments/DISCOVERY.md`）
- metric + direction:
  - Q1: `bytes_per_token`（**higher = better compression**；越低表示更碎、更差），按 domain 分列；并列 `chars_per_token`
  - Q2: 分桶/相关 — 报告 `seconds_per_merge` vs `affected`（early/mid/late 汇总表）；**不改**训练代码刷速度
- eval_command:
  ```bash
  cd assignment1/assignment1-basics
  # Q1
  uv run --no-sync python experiments/eval_q1_compression.py \
    --tokenizer-dir /data1/wcz/projects/Myllm-runs/tokenizers/openwebtext-v2 \
    --max-bytes 5000000 \
    --out experiments/results/q1_compression.csv
  # Q2
  uv run --no-sync python experiments/eval_q2_merge_profile.py \
    --log /data1/wcz/projects/Myllm-runs/logs/owt-bpe-train-v2.log \
    --out experiments/results/q2_merge_profile.csv
  ```
- data_version / split:
  - Q1 domains（冻结）:
    - `owt`: `/data1/wcz/datasets/myllm/openwebtext/owt_valid.txt` 前 `--max-bytes`
    - `tinystories`: `/data1/wcz/datasets/myllm/tinystories/TinyStoriesV2-GPT4-valid.txt` 前 `--max-bytes`
    - `zh`: `experiments/fixtures/zh_sample.txt`（固定文件，不改）
  - Q2: 上述 train log（只读）
- editable_scope: `experiments/eval_*.py` 的展示/分桶；`experiments/results/`；**不改** `cs336_basics/train_bpe.py`；不重训 OWT
- forbidden_edits: metric 定义（必须用 utf-8 字节数 / token 数）；domain 文件与 max-bytes 默认；重训改 vocab；改 merge 平局
- budget_wall_clock: 2h
- wall_clock_per_trial: Q1 ≤40m；Q2 ≤20m
- stop_condition: Q1 三域表写出；Q2 early/mid/late 三行汇总写出；超时停
- expected_output_shape:
  - Q1 CSV: `domain,bytes,tokens,bytes_per_token,chars_per_token,runtime_s`
  - Q2 CSV: `bucket,n_steps,mean_s_per_merge,mean_affected,mean_freq` + 可选 per-step 明细
- git_branch / baseline_commit: （运行前 `git rev-parse --short HEAD` 填入 registry）
- Obsidian: off

## Phase 2 Stop gate

```text
eval 与 metric 已冻结（对齐 Discover Q1+Q2）。确认后进入 ablation-planner？
```
