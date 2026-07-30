## Ablation / Track-B Plan · 2026-07-29 · A1 BPE

Injected from Discover Q1+Q2（非模型组件消融；为**测量实验**）。未调 Codex MCP：卡面已足够，避免空转。

### P1 only（maps_to）

| # | Name | maps_to | What It Tests | Expected If Matters | Priority |
| --- | --- | --- | --- | --- | --- |
| 1 | merge-log profile early/mid/late | Q2 | 训练加速是否由 `affected` 驱动 | early mean_affected ≫ late，且与 s/merge 同向 | 1 |
| 2 | cross-domain bytes/token | Q1 | OWT 32k 跨域压缩 | owt ≤ tinystories ≪ zh（bytes/token） | 1 |

### Unnecessary（本轮不跑）
- 重训 8k/16k vocab
- GPT-2 merges 对照
- encode Fast vs slow 吞吐（DISCOVERY Q3）

### Run order
1. Q2（只读日志，分钟级）→ 早出口述弹药  
2. Q1（5MB×2 域 encode，可能十几–几十分钟）

### Autoresearch brief
- **模式**: 测量-only（**不**开参数搜索改 `train_bpe`）
- target: 无代码优化目标；执行 `FREEZE.md` 两条 `eval_command`
- metric: 见 Freeze；keep/discard 不适用
- stop: 两张 CSV 写出即停

### Estimated compute
- CPU only；≤2h wall（Freeze 预算）
