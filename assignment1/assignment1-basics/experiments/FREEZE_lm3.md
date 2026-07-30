## Freeze · 2026-07-29 · A1 lm3（用户选 1–5）

- discovery_refs: DISCOVERY_lm3 Q1–Q5
- 轨B约束: 通宵重训只开 **Q5** 为 P1；Q1–Q4 解析已有跑次，不改训练代码循环
- metric:
  - Q1: `valid_loss` @20k for lr∈{1e-4,3e-4,6e-4}（+ warmup500 旁证）
  - Q2: `tokens_seen` vs `valid_loss`（bs16/32/64 同 step；按 token 对齐叙事）
  - Q3: 负结果表 Δ(wd0−ba)、Δ(θ500k−ba)
  - Q4: overfit `step → train_loss`（目标近 0）
  - Q5: constant `min_lr=max_lr=3e-4` vs ba cosine `valid_loss`
- SwanLab（Q5）:
  - 读 `.env`：`SWANLAB_API_KEY` / `SWANLAB_PROJ_NAME=cs336-tinystories` / `SWANLAB_WORKSPACE` / `SWANLAB_MODE=cloud`
  - `SWANLAB_EXP_NAME=const_lr_3e4_20k`
  - `SWANLAB_GROUP=lm3-schedule`
  - `SWANLAB_TAGS=lm3,schedule,constant-lr`
  - 指标键：`train/loss` `train/lr` `valid/loss` `time/wall_s` `data/tokens_seen`
- Obsidian: off
- budget: Q1–Q4 ≤1h；Q5 通宵 20k steps（1 GPU）
