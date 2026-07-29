# Obsidian note template — CS336 抗追问加深

Default vault folder: `CS自学/Diy-llm/抗追问/`

Filename: `CS336-<Module>-抗追问-<短标题>.md`  
Examples: `CS336-A2-抗追问-BM与occupancy.md`, `CS336-A1-抗追问-中文压缩率.md`

Create a **new note** per deepened session card (or append a new `## Session` under an existing module note if the user asks to keep one file per module).

```markdown
---
title: CS336 <Module> 抗追问 · <短标题>
tags:
  - cs336
  - diy-llm
  - anti-interrogation
  - <module-slug>
date: YYYY-MM-DD
module: <A2 FlashAttention>
mode: <manual|auto>
status: <solid|partial>
source: <repo relative path or EXPERIMENT_LOG.md>
plugin: <off|ablation|other>
---

# CS336 <Module> · 抗追问 · <短标题>

> 面试追问（原话）：…
> 对应实验日志：`EXPERIMENT_LOG.md` · [<日期条目>]

## 1. 我原先以为
- （用户开场 knowledge claim / 口述里暴露的先验）

## 2. 最小实验
- 命令：`…`
- 参数范围：…
- 预算 / 实际耗时：…

### 结果表
| setting | metric | notes |
| --- | --- | --- |
| … | … | … |

## 3. 口述（第一轮）
- （尽量用用户原话；可轻微清理口误，勿改写成教材）

## 4. 引导后补全
- 缺口类型：…
- 引导追问：…
- 第二轮口述：…

## 5. 加深锚点（一句话）
- 用**自己的数字**回答面试追问：…
- 若再被追问「为什么」，下一句接：…

## 6. 仍未闭合
- [ ] …
- 建议下一张 P-card：…

## 7. 链接
- 代码：`[[…]]` 或 repo path
- 相关笔记：`[[chapter…]]` / `[[CS336-A1-…]]`
```

### Writing rules

1. Prefer the user's phrasing in §3–§5; do not upgrade into a polished textbook.
2. Every claim in §5 must cite a row from §2's table (or explicitly say「未测」).
3. `status: solid` only if acceptance criteria met after guidance; else `partial`.
4. Never paste a full model answer into the note.
5. Align frontmatter style with existing `CS自学/Diy-llm` notes (`title`, `tags`, `date`, `source`).
