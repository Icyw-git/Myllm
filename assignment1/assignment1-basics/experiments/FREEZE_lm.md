## Freeze · 2026-07-29 · A1 LM / §7.3（Q1+Q2）

- claim: §7.3 组件贡献可用 valid_loss 量化；RoPE 在短 ctx 上的小 gap 可能掩盖长依赖需求。
- hypothesis:
  - Q1: no_rmsnorm 伤害最大；降 lr 不能拉回 baseline。
  - Q2: 同一对 baseline/no_rope ckpt，加长 eval ctx 后 Δvalid_loss 增大（或至少不缩小）。
- discovery_refs: Q1, Q2（`experiments/DISCOVERY_lm.md`）
- metric + direction: `valid_loss`（**lower better**）；并列 `valid_ppl`；Q2 另报 `delta = no_rope - baseline`
- eval_command:
  ```bash
  cd assignment1/assignment1-basics
  # Q1
  uv run --no-sync python experiments/eval_q1_ablation_table.py \
    --root /data1/wcz/projects/Myllm-runs/experiments/ablation73 \
    --out experiments/results/q1_ablation_table.csv
  # Q2
  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python experiments/eval_q2_rope_ctx.py \
    --baseline-ckpt /data1/wcz/projects/Myllm-runs/experiments/ablation73/baseline/ckpt_step8000.pt \
    --no-rope-ckpt /data1/wcz/projects/Myllm-runs/experiments/ablation73/no_rope/ckpt_step8000.pt \
    --contexts 64,128,256,512 \
    --out experiments/results/q2_rope_ctx.csv
  ```
- data_version / split: 与消融训练相同 valid npy（脚本内读 run manifest / run_train defaults）
- editable_scope: `experiments/eval_q*.py`；不改 `cs336_basics/linear.py` / 不重训除非 Q2 失败
- forbidden_edits: valid_loss 定义；换 train/valid 文件；静默改 variant 权重
- budget_wall_clock: 4h
- wall_clock_per_trial: Q1 ≤20m；Q2 ≤1h（纯 eval）
- stop_condition: 两张 CSV 写出
- expected_output_shape:
  - Q1: `run,variant,max_lr,step,valid_loss,valid_ppl,delta_vs_baseline`
  - Q2: `variant,context_length,valid_loss,valid_ppl,delta_vs_baseline`
- Obsidian: off
