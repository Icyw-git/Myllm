## Freeze · 2026-07-29 · A1 lm2（用户选 1–5）

- discovery_refs: DISCOVERY_lm2 Q1–Q5
- 轨B约束: 通宵重训只开 **Q5** 为 P1；Q1–Q4 为冻结测量（解析/bench），不占 autoresearch 改代码循环
- metric:
  - Q1: `n_params` + `valid_loss`（已有 silu vs baseline）
  - Q2: early/mid/late `valid_loss`（baseline vs post_norm）
  - Q3: scaling 表 `valid_loss` / params（L2/L6/d384/d768 vs ba）
  - Q4: `tokens_per_sec`（ref vs fast encode）
  - Q5: `valid_loss` @ train+eval ctx=512，Δ(no_rope−baseline)
- Obsidian: off
- budget: Q1–Q4 ≤2h；Q5 通宵（两卡并行 baseline / no_rope）
