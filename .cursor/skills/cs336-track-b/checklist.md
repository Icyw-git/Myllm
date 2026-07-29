# Track B checklist

```text
Track B Progress:
- [ ] Deps: ablation-planner, autoresearch-agent（冻结用内置 freeze-template，无需 logbook）
- [ ] Phase 0 GO
- [ ] Phase 1 freeze + user confirmed
- [ ] Phase 2 P1 trimmed + user confirmed
- [ ] Phase 3 autoresearch done; registry + EXPERIMENT_LOG updated
- [ ] Phase 4 harvest oral + Obsidian
```

## Install（仅规划 + 通宵跑）

```bash
npx skills add wanshuiyin/auto-claude-code-research-in-sleep --skill ablation-planner -g -y
npx skills add alirezarezvani/claude-skills@autoresearch-agent -g -y
```

## 可选外部记账（通常不需要）

```bash
npx skills add eyadsibai/ltk@experiment-tracking -g -y
# 或偏 Obsidian：
# npx skills add galaxy-dawn/claude-scholar@obsidian-experiment-log -g -y
```

## Start

```text
开始轨B。
claim: …
module: …
metric + direction: …
eval_command: …
editable_scope: …
budget_wall_clock: …
```
