# Diy-llm repo hooks（项目专用）

本文件只给 **本仓库** 的 interrogation tutor 用；全局 `minimal-ablation-proposer` 不读死路径。
发 card / 调 plugin 时，把下面模板填进 `command_template` 与 `param_range`。

## A1 §7.3 architecture ablations

```bash
cd assignment1/assignment1-basics
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/run_ablation_train.py \
  --variant {variant} \
  --steps {steps} \
  --out-dir {out_dir}/{variant}
```

- `param_range` 常用：`variant ∈ {baseline, no_rmsnorm, post_norm, no_rope, silu_ffn}`
- 默认先 `baseline`，再选与 `what_it_tests` 最相关的单变体

## Numeric / systems sweeps (A2 等)

无统一入口时：只用 card 里用户已有的 bench 命令；不要新建 kernel/训练脚本。

## Obsidian

- 默认目录：`CS自学/Diy-llm/抗追问/`
- tags：`cs336`, `diy-llm`, `anti-interrogation`, `<module-slug>`
