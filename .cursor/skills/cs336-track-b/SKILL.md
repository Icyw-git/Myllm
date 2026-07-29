---
name: cs336-track-b
description: >-
  CS336 Track B (通宵实验弹药): freeze via built-in freeze-template (no
  experiment-logbook), plan P1 with ablation-planner, overnight run with
  autoresearch-agent, then harvest oral defense + Obsidian. No labrat. Not for
  daily ≤1h anti-interrogation (cs336-track-a). Use when the user says 轨B,
  Track B, 通宵实验, claim已硬, 实验弹药, or planner→autoresearch.
---

# CS336 Track B · 通宵实验弹药

一句话：Claim 已硬时，**内置冻结模板** → `ablation-planner`（只 P1）→ `autoresearch-agent` → 本 skill 内收割。**不用 labrat / experiment-logbook。**

**本 skill 不含日常盲点扫描 / ≤1h 抗追问主循环。** 那是轨 A：`cs336-track-a`。

## Dependency skills（用户自行全局安装）

| 角色 | Skill | Install |
| --- | --- | --- |
| 冻结 | **内置** [freeze-template.md](freeze-template.md) | 无需安装 |
| 规划 | `ablation-planner` | `npx skills add wanshuiyin/auto-claude-code-research-in-sleep --skill ablation-planner -g -y` |
| 通宵跑 | `autoresearch-agent` | `npx skills add alirezarezvani/claude-skills@autoresearch-agent -g -y` |

可选外部记账（一般不需要）：`eyadsibai/ltk@experiment-tracking`（~83 installs）或 `galaxy-dawn/claude-scholar@obsidian-experiment-log`（~50，偏 Obsidian）。默认用内置冻结即可。

启动 planner / autoresearch 前：在 `~/.agents/skills/` **Read 对应 SKILL.md**。缺则打印安装命令并停止；禁止 labrat。

模板：

- 冻结：[freeze-template.md](freeze-template.md)
- 日志条目：[../cs336-track-a/log-template.md](../cs336-track-a/log-template.md)
- Obsidian：[../cs336-track-a/obsidian-note-template.md](../cs336-track-a/obsidian-note-template.md)
- 命令钩子：[../cs336-track-a/repo-hooks.md](../cs336-track-a/repo-hooks.md)

## Hard constraints

1. 无 claim + frozen eval → **NO-GO**（可建议改开轨 A）。
2. 顺序：Freeze → Plan(P1) → Run → Harvest。
3. 只跑 **P1**（最多 +1 个 P2）。
4. 通宵中禁止改 metric / 数据划分 / eval。
5. 不调用 labrat / experiment-logbook；不跑轨 A 盲点全流程。
6. 遵守作业 `AGENTS.md`；单模块；可 git 回退。

## Inputs

```text
claim: …
module: …
components_or_knobs: …
metric + direction: …
eval_command: …           # frozen
editable_scope: …
budget_wall_clock: …
out_dir / results_path: …
```

## Pipeline

### Phase 0 — Gate

- [ ] claim 可证伪
- [ ] eval 已写死
- [ ] editable_scope + 可回退
- [ ] budget
- [ ] `ablation-planner` + `autoresearch-agent` 可 Read

→ `Track B GO` 或 `NO-GO`。

### Phase 1 — Freeze（内置，不调外部 logbook）

1. 按 [freeze-template.md](freeze-template.md) 写冻结块 + 确保 `experiments/registry.csv`。
2. **Stop：** 用户确认 eval 冻结。

### Phase 2 — Plan（`ablation-planner`）

按其 SKILL 出表 → **只留 P1（可选一 P2）** → Autoresearch brief。  
**Stop：** 用户确认或「直接跑」。

### Phase 3 — Run（`autoresearch-agent`）

按其 SKILL 循环；注入 brief；无改进则回退；结果写入 registry + `EXPERIMENT_LOG.md`。

### Phase 4 — Harvest（本 skill 内）

1. 1–2 张追问卡（指向表上数字；不给满分答案）。
2. 口述 + 引导环。
3. Append 日志；写 Obsidian。

## Triggers

- `开始轨B` / `Track B`
- `通宵实验弹药` / `claim已硬`

## Anti-patterns

- 再去装 / 调用 `experiment-logbook`
- planner 全表通宵；改 eval 刷分；跳过口述

## Checklist

See [checklist.md](checklist.md).

## 另一条途径

日常 ≤1h 抗追问 → **`cs336-track-a`**（另开会话）。
