# EXPERIMENT_LOG template

Create `EXPERIMENT_LOG.md` at the module or repo root if missing. Append entries; never overwrite history.

```markdown
## [YYYY-MM-DD] Module: <A2 FlashAttention> · Mode: <manual|auto> · Plugin: <off|ablation|…>

### Blindspot selected
- ...

### Interview question
- ...

### Hypothesis / expected_if_matters
- ...

### Command
\`\`\`bash
# exact command(s)
\`\`\`

### Plugin proposals (optional)
| round | proposed command | accepted? | reason |
| --- | --- | --- | --- |
| 1 | ... | yes/no | budget / scope / spoiler |

### Results
| setting | metric | notes |
| --- | --- | --- |
| ... | ... | ... |

### Oral answer attempt (user)
- (paste or paraphrase user's words)

### Guidance loop (if incomplete)
- gap type:
- scaffold prompt:
- second oral attempt:

### Gaps still open
- ...

### Obsidian note
- path: `CS自学/Diy-llm/抗追问/...`
- status: solid | partial

### Next P-item (optional)
- ...
```
