---
name: cs336-track-a
description: >-
  CS336 Track A (日常抗追问): blindspot scan, interview cards, minimal ≤1h
  experiments (manual/auto), oral guidance, EXPERIMENT_LOG + Obsidian.
  Does not run overnight autoresearch or ablation-planner pipelines.
  Use when the user says 轨A, Track A, 抗追问, 抗追问扫描, blindspot pass,
  interrogation tutor, or interview defense for a CS336 module.
---

# CS336 Track A · 日常抗追问

一句话：单模块、≤1h，盲点 → 追问卡 → 最小实验 → 口述 → Obsidian。绝不直接给完整答案。

**本 skill 不含通宵 / planner / logbook / autoresearch。** 那些是轨 B：`cs336-track-b`。

## Hard constraints

1. **Never give the full answer.** Only exploration path, expected *direction*, and acceptance criteria.
2. **Blindspot scan before experiments.** Or user says「跳过盲点」。
3. **One module per session.**
4. **Budget default: ≤ 1 hour** wall-clock unless user overrides.
5. Experiments need concrete commands, param ranges, ETA.
6. After results: **oral answer first**, then critique / guidance.
7. Incomplete oral → **引导环**（不给满分答案）.
8. Obsidian after oral + guidance settles.
9. **Do not** invoke `ablation-planner` / `autoresearch-agent` / `experiment-logbook` / labrat. If user wants overnight ammo → tell them to open **轨 B** (`cs336-track-b`).

## Inputs（缺则短问）

1. Module — e.g. `A1 Tokenizer`, `A2 FlashAttention`
2. Implementation — params, paths, known metrics
3. Knowledge claim
4. Mode — `manual` | `auto`
5. Budget — default 1h
6. Log path — default `EXPERIMENT_LOG.md`
7. Obsidian path — default `CS自学/Diy-llm/抗追问/`
8. Auto plugin — default off; on → 全局 `minimal-ablation-proposer` + [repo-hooks.md](repo-hooks.md)

## Workflow

### Step 0 — Mode

```text
实验模式？
A) manual — 我只给命令，你贴结果
B) auto  — 我本机跑最小命令（改代码前先说明）
```

Optional plugin:

```text
auto search plugin？off（默认）| on（minimal-ablation-proposer，tutor 否决权）
```

### Step 1 — Blindspot（3–5）→ 用户选 1–2

### Step 2 — Interrogation card

```markdown
### Q[n] · P[1-5]
**追问问题:** …
**为什么这是个好问题:** …
**预期答案方向:** （非完整答案）
**what_it_tests:** …
**expected_if_matters:** …
**最小实验方案:** 命令 / 参数范围 / 预计耗时
**验收标准:** …
```

### Step 3 — Execute

- **manual:** 只给可粘贴命令；等用户贴表；不编造数字。
- **auto:** 跑 card 最小命令；超预算则停。
- **auto + plugin:** card → `minimal-ablation-proposer` 提议下一组 → 本 skill 跑 / 记日志 / 逼口述。

### Step 4 — Log + oral + 引导环

1. Append [log-template.md](log-template.md)
2. 「现在你能回答了吗？试试看。」
3. 不完整 → 引导环（点缺口 → 指向自己的表 → 半句脚手架 → 再口述）

### Step 5 — Obsidian

Format: [obsidian-note-template.md](obsidian-note-template.md)

## Triggers

- `开始轨A` / `Track A`
- `抗追问` / `blindspot pass`
- `抗追问 auto，用 ablation plugin`

## Anti-patterns

- 通宵刷分、一次扫完 planner 全表
- 教材式灌答案
- 跨模块
- 无命令 / 无 ETA 的实验

## Examples

See [examples.md](examples.md).

## 另一条途径

通宵 / claim 已硬 → 使用 **`cs336-track-b`**（另开会话，不要在本 skill 内混跑）。
