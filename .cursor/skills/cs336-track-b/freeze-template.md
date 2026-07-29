# Track B · Freeze template（替代 experiment-logbook）

在模块或仓库根维护：

1. `EXPERIMENT_LOG.md` — 追加叙事条目（见 [../cs336-track-a/log-template.md](../cs336-track-a/log-template.md)）
2. `experiments/registry.csv` — 机器可读账本（没有就创建）

## 运行前冻结清单（Phase 1 必填）

写入 `EXPERIMENT_LOG.md` 开篇，或单独 `experiments/FREEZE.md`：

```markdown
## Freeze · YYYY-MM-DD · <module>

- claim:
- hypothesis:
- metric + direction:          # e.g. lower val_bpb / higher tok/s — 通宵中不可改
- eval_command:                # frozen harness — 通宵中不可改
- data_version / split:
- editable_scope:              # 允许改的路径
- forbidden_edits:             # eval / 数据划分 / metric 定义
- budget_wall_clock:
- wall_clock_per_trial:
- stop_condition:
- expected_output_shape:       # 表头或关键数字位置
- git_branch / baseline_commit:
```

## registry.csv 表头

```csv
run_id,git_commit,command,config,seed,dataset,metric,result,runtime_s,status,notes
```

规则：

1. 先建基线行（`status=baseline`），再开 autoresearch。
2. 代码变更前先 commit；无改进则 reset，仍记一行 `status=reject`。
3. **绝不许**静默改 `eval_command` / metric / 数据划分。
4. 输出目录建议：`train_outputs/`（可信）、`explore_outputs/`（探索）、`debug_outputs/`（失败）。

## Phase 1 Stop gate

向用户确认：

```text
eval 与 metric 已冻结。确认后进入 ablation-planner？
```
