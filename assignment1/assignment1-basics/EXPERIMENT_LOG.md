# EXPERIMENT_LOG

## [2026-07-29] Module: A1 Tokenizer / BPE · Mode: auto · Plugin: off

### Blindspot selected
- Q1 跨域压缩率（OWT / TinyStories / 中文）
- Q2 训练越来越快与 affected 的关系

### Interview question
- 见 `experiments/DISCOVERY.md` Q1 / Q2

### Hypothesis / expected_if_matters
- Q1: OWT valid 压缩最好（更高 bytes/token）；中文更碎（更低 bytes/token）
- Q2: early affected/s/merge ≫ late；二者正相关

### Command
```bash
uv run --no-sync python experiments/eval_q2_merge_profile.py \
  --log /data1/wcz/projects/Myllm-runs/logs/owt-bpe-train-v2.log \
  --out experiments/results/q2_merge_profile.csv

uv run --no-sync python experiments/eval_q1_compression.py \
  --tokenizer-dir /data1/wcz/projects/Myllm-runs/tokenizers/openwebtext-v2 \
  --max-bytes 5000000 \
  --out experiments/results/q1_compression.csv
```

### Results

**OWT train baseline:** done 8700.3s | vocab=32000 merges=31743 → `Myllm-runs/tokenizers/openwebtext-v2`

**Q2** `experiments/results/q2_merge_profile.csv`:

| bucket | mean_s_per_merge | mean_affected | steps |
| --- | --- | --- | --- |
| early (10%) | 1.22 | 8187 | 1–3174 |
| mid | 0.083 | 241 | 15872–19045 |
| late | 0.029 | 139 | 28570–31743 |

补充（口述更锋利）：first50 mean_s≈13.0、mean_aff≈1.7e5；last50 mean_s≈0.024、mean_aff≈104；affected vs s/merge Pearson≈0.89（每 100 step 抽样）

**Q1** `experiments/results/q1_compression.csv`（max-bytes=5e6；zh=全量夹具）:

| domain | bytes_per_token | chars_per_token | tokens |
| --- | --- | --- | --- |
| owt | 4.4288 | 4.3911 | 1.13e6 |
| tinystories | 4.0040 | 4.0023 | 1.25e6 |
| zh | 1.1767 | 0.4179 | 962 |

注：`bytes_per_token` **越高压缩越好**。zh 的 chars/token≈0.42 ⇒ 平均每字 >2 token。

### Oral answer attempt (user)
- 第一轮：方向对但变量表述偏笼统（「pair 数」未落到 affected；中文「几乎无压缩」偏绝对）
- 第二轮：看过对照表示例后表示理解（status: solid 口径已对齐：affected 驱动墙钟；bytes/token 越高越好；zh≈1.18 / chars/token≈0.42）

### Guidance loop (if incomplete)
- gap type: 变量命名不准；压缩指标方向与中文碎片化表述
- scaffold: 指向 first50/last50 与 Q1 三行表
- second oral attempt: 用户确认已明白

### Gaps still open
- [ ] 可选闭卷重述一遍（不看示例）
- [ ] 可选 P2：GPT-2 对照 / encode 吞吐

### Obsidian note
- skipped

---

## [2026-07-29] Module: A1 LM / §7.3 · Mode: auto · Plugin: off

### Blindspot selected
- Q1 §7.3 消融 Δvalid_loss
- Q2 RoPE × eval context_length

### Results

**Q1** `experiments/results/q1_ablation_table.csv`（Δ vs baseline @8k）:

| run | valid_loss | Δ |
| --- | --- | --- |
| baseline | 1.597 | 0 |
| silu_ffn | 1.691 | +0.094 |
| post_norm | 1.692 | +0.095 |
| no_rope | 1.710 | +0.114 |
| no_rmsnorm @3e-4 | 2.073 | +0.477 |
| no_rmsnorm @1e-4 | 2.825 | +1.229 |

**Q2** `experiments/results/q2_rope_ctx.csv`（同 ckpt，改 eval ctx）:

| ctx | baseline | no_rope | Δ (nr−base) |
| --- | --- | --- | --- |
| 64 | 1.863 | 1.969 | +0.105 |
| 128 | 1.685 | 1.794 | +0.108 |
| 256 | 1.638 | 1.745 | +0.107 |
| 512 | 2.064 | 1.952 | **−0.112** |

注：64–256 上 Δ≈0.11 几乎不随长度放大；512 超出训练 ctx=256，baseline 因 RoPE 外推变差，**不能**当成「长上下文更需要 RoPE」的干净证据。

### Oral answer attempt (user)
- Q1: no_rmsnorm 最伤，Δ≈0.48/1.23；降 lr 更差 — 正确
- Q2: 512 不能当证据 — 正确；补答：64–256 平坦 Δ 否定「序列越长 RoPE 效果越明显」— solid（限定在训练长度内 eval）

### Obsidian note
- skipped

---

## [2026-07-29] Module: A1 lm2 · Mode: auto · Plugin: off

### Blindspot selected
- Q1–Q5（全选）；通宵重训仅 Q5

### Results

**Q1 silu fairness** `experiments/results/lm2/q1_silu_fairness.csv`
- baseline 22.70M valid=1.597
- silu_ffn 19.94M (−2.75M) valid=1.691 Δ=+0.094
- 结论方向：参数更少仍更差 ⇒ 不能单用「容量不够」开脱，但仍是联合改动（激活+d_ff 约定）

**Q2 norm stability** `experiments/results/lm2/q2_norm_stability.csv`
- early mean valid: baseline 2.82 / post_norm 2.92 / no_rmsnorm 4.24
- late mean: 1.61 / 1.71 / 2.09
- post 相对 baseline 全程平行略差，未见「炸训」级不稳定

**Q3 scaling** `experiments/results/lm2/q3_scaling.csv`（20k steps）
| run | L | d | params_M | valid_loss |
| --- | --- | --- | --- | --- |
| L6_20k | 6 | 512 | 28.9 | 1.430 |
| d768_20k | 4 | 768 | 37.2 | 1.435 |
| tinystories_ba | 4 | 512 | 22.7 | 1.482 |
| d384_20k | 4 | 384 | 16.2 | 1.527 |
| L2_20k | 2 | 512 | 16.5 | 1.592 |
| ctx128_20k | 4 | 512 | 22.7 | 1.669 |

注：d384/d768 的 d_ff 同为 1344，宽度扫不纯。

**Q4 encode** `experiments/results/lm2/q4_encode_throughput.csv`（50KB, tinystories tok）
- ref Tokenizer: ~802 tok/s
- FastEncoder: ~7.0e5 tok/s
- speedup ≈ **878×**

**Q5** 进行中：tmux `a1-q5-rope`；ctx=512 bs=16 steps=8000（token 对齐原 8k×32×256）；out `ablation73_ctx512/{baseline,no_rope}`

### Oral / Obsidian
- Q1–Q4 口述第一轮：见下；Q5 ~3.6k/8k，中期 Δ(valid)≈0.21（1.75 vs 1.96）
- Obsidian: skipped

### Oral answer attempt (user) · Q1–Q4
- Q1: 用户要示例答案（见对话）；口径：少参仍差 ⇒ 不能只甩锅容量，但仍是联合改动
- Q2: 无炸训证据 — solid
- Q3: 上下文减半 + token 减半（修正「训练缓慢」）— solid
- Q4: 全量扫 merge ~878× — solid

### Q5 final（train+eval ctx=512, bs=16, 8k, token-matched）
| variant | valid_loss | valid_ppl | Δ vs baseline |
| --- | --- | --- | --- |
| baseline | 1.584 | 4.88 | 0 |
| no_rope | 1.735 | 5.67 | **+0.151** |

对照原 ablation73（train+eval ctx=256）：Δ≈**+0.114**  
⇒ 在**同长度训练**下，加长 ctx 后 RoPE gap **略增大**（0.11→0.15），与「只改 eval、外推到 512」的假信号相反。  
CSV: `experiments/results/lm2/q5_rope_ctx512.csv`

### Oral Q5
- 用户：gap 变大；因上次 train/eval ctx 不一致 — solid
- 可补半句：上次是 RoPE **外推**伤 baseline，不是干净的「长上下文更需要 RoPE」

---

## [2026-07-29] Module: A1 lm3 · Mode: auto · Plugin: off

### Blindspot selected
- Q1–Q5（全选）；通宵重训仅 Q5 constant lr

### SwanLab format（Q5）
- project=`cs336-tinystories` group=`lm3-schedule` exp=`const_lr_3e4_20k`
- tags=`lm3,schedule,constant-lr`
- metrics: `train/loss` `train/lr` `valid/loss` `time/wall_s` `data/tokens_seen`
- run: https://swanlab.cn/@07011812138/cs336-tinystories/runs/hqd8mc40
- `run_train.py` 已对齐 ablation：日志含 `tokens_seen`，SwanLab 记 `data/tokens_seen`；支持 `SWANLAB_GROUP`

### Results（Q1–Q4 已有跑次）

**Q1** `experiments/results/lm3/q1_lr_sweep.csv`
| run | max_lr | valid |
| --- | --- | --- |
| lr1 | 1e-4 | 1.617 |
| tinystories_ba | 3e-4 | 1.482 |
| lr6 | 6e-4 | 1.451 |
| warmup500_20k | 3e-4 (warmup=500) | 1.487 |

**Q2** token 公平（同 20k step 时 bs64 tokens=2×ba、bs16=½ba）
- 同 step：bs16 1.567 / ba 1.482 / bs64 1.383
- token-matched @ ba final tokens（1.638e8）：bs64@10k → **1.487** ≈ ba；bs16 达不到该预算（终 8.19e7）
- CSV: `q2_batch_tokens_*.csv`

**Q3** 负结果 `q3_negative_results.csv`
- wd0 Δ=+0.003；rope500k Δ≈−0.001（相对 ba）

**Q4** overfit：`overfit_002` first≈9.29 → last≈0.002（ok）；`overfit_001/smoke` 缺 log

**Q5** 完成：tmux `a1-lm3-constlr`；`min_lr=max_lr=3e-4` warmup=200；out `Myllm-runs/experiments/const_lr_3e4_20k`

| run | schedule | final valid | final lr | Δ vs ba |
| --- | --- | --- | --- | --- |
| tinystories_ba | cosine → 3e-5 | **1.482** | 3e-5 | 0 |
| const_lr_3e4_20k | constant 3e-4 | **1.502** | 3e-4 | **+0.020** |

曲线要点：中期两者接近（5k: 1.72 vs 1.73）；后期 cosine 继续降，const 在 ~1.50–1.54 抖动、终值略差。  
CSV: `experiments/results/lm3/q5_const_vs_cosine*.csv`  
SwanLab: https://swanlab.cn/@07011812138/cs336-tinystories/runs/hqd8mc40

### Oral prompts（待你答）
- Q1: 更高 lr 一直更好吗？边界在哪？
- Q2: bs64 更好是不是大 batch 优越？token 对齐后怎么说？
- Q3: wd/θ「没效果」的证据边界？
- Q4: 怎么证明训练代码没写炸？
- Q5: cosine 关掉会怎样？用 1.482 vs 1.502 怎么讲？

### Oral answer attempt (user) · Q5
- 用户：训练不稳定，效果不如余弦退火 — **partial→可过**
- 证据对齐：终值 1.502 vs 1.482（Δ+0.02）支持「不如」；late valid 抖动 const 更大（≥15k range≈0.055 vs ba≈0.027）支持「更不稳」
- 宜补半句：不是炸训/发散，是**后期缺 decay** → 收敛更吵、略差；中期（~5k）两者仍接近

### Obsidian
- skipped

---

## [2026-07-29] Module: A1 lm4 · Mode: auto · Plugin: off

### Blindspot selected
- Q1 warmup / Q2 lr×batch / Q3 复合归因 / Q5 d_ff 对齐（跳过 Q4 生成）
- 通宵 P1 = Q5 两跑并行

### Freeze
- `experiments/FREEZE_lm4.md`；metric=`valid_loss`；planner skipped

### Results（Q1–Q3）

**Q1** `experiments/results/lm4/q1_warmup.csv`
| run | warmup | final valid | early_v | late_v | Δ |
| --- | --- | --- | --- | --- | --- |
| tinystories_ba | 200 | 1.482 | 2.264 | 1.471 | 0 |
| warmu | 0 | 1.489 | 2.275 | 1.478 | **+0.007** |

**Q2** 同 step 2×2：ba 1.482 / lr6 1.451 / bs64 1.383 / bs64_lr6e4 **1.347**  
token-matched @1.638e8：ba 1.482 / lr6 1.451 / bs64@10k 1.487 / bs64_lr6e4@10k **1.463**  
⇒ 同 token 下「只加大 bs」优势消失；**bs×2+lr×2** 仍略好于 ba。

**Q3** `q3_compound_attribution.csv`（Δ vs ba）
| run | Δ |
| --- | --- |
| L6 | −0.052 |
| d768（d_ff confound） | −0.047 |
| bs64 | −0.099 |
| lr6 | −0.031 |
| bs64_lr6e4 | −0.135 |
| L6_d768_bs64_lr6e4 | **−0.219** (valid **1.263**) |

不能唯一分解；复合 > 任一单轴。

**Q5** 完成（SwanLab group=`lm4-dff-align`）CSV: `experiments/results/lm4/q5_dff_align.csv`

| run | d | d_ff | params_M | valid | Δ vs ba | vs 旧同宽 |
| --- | --- | --- | --- | --- | --- | --- |
| d384_20k（旧） | 384 | 1344 | 16.24 | 1.527 | +0.045 | — |
| d384_dff1024 | 384 | **1024** | **14.76** | **1.506** | +0.025 | **−0.020** |
| tinystories_ba | 512 | 1344 | 22.70 | 1.482 | 0 | — |
| d768_20k（旧） | 768 | 1344 | 37.19 | 1.435 | −0.047 | — |
| d768_dff2048 | 768 | **2048** | **43.68** | **1.379** | **−0.103** | **−0.056** |

对齐链：1.506 → 1.482 → 1.379（随宽单调降）。旧表低估了加宽收益（d768 钉死 1344）；d384 旧跑 FFN 过大，对齐后参数更少反而略好。

### Oral prompts
- Q1: warmup=0 几乎打平，还重要吗？
- Q2: 2×bs 该不该 2×lr？同 step vs 同 token 怎么讲？
- Q3: 1.263 怎么证明不是只靠更大 batch/lr？
- Q5: d_ff 对齐后，宽度 scaling 结论翻盘了吗？用 1.506/1.482/1.379 怎么讲？

### Obsidian
- skipped

---

## [2026-07-29] Module: A1 OWT LM · Mode: track-b discover+freeze · Plugin: off

### Blindspot selected
- 用户确认: **Q1 + Q2 + Q3**（Q0 tokenize 为阻塞前置）

### Freeze
- `experiments/FREEZE_owt.md`
- metric=`valid_loss`↓（OWT 内比）；planner skipped
- 并行: GPU0 `owt_ba` / GPU1 `lr6e4` / GPU2 `bs64` / GPU3 `bs64_lr6e4`
- 缺口: `run_tokenize.py` 已加 `--corpus owt`（产物 `owt_{train,valid}.npy`）
- SwanLab: project=`cs336-owt` group=`owt-hparam`；`run_train` 按 data 路径打 `owt` tag
- **Done:** tmux `owt-overnight` EXIT=0；四跑均 step=20000

### Results

**Q0** tokenize: train=2.727e9 tok（5.1G）/ valid=6.64e7；meta=`tokenized/owt_tokenize_meta.json`

**Q1** `experiments/results/owt/q1_baseline.csv`（同 arch/20k）

| run | domain | valid | ppl | tokens |
| --- | --- | --- | --- | --- |
| tinystories_ba | TS | **1.482** | — | 1.638e8 |
| owt_ba_20k | OWT | **4.248** | 70.0 | 1.638e8 |

**Q2** 同 step 网格 `q2_hparam_grid.csv`；token-matched `q2_token_matched.csv`

| run | bs | lr | valid@20k | Δ vs owt_ba | tokens | wall_s |
| --- | --- | --- | --- | --- | --- | --- |
| owt_ba_20k | 32 | 3e-4 | **4.248** | 0 | 1.638e8 | 2349 |
| owt_lr6e4_20k | 32 | 6e-4 | 4.182 | −0.066 | 1.638e8 | 2454 |
| owt_bs64_20k | 64 | 3e-4 | 3.994 | −0.255 | 3.277e8 | 4625 |
| owt_bs64_lr6e4_20k | 64 | 6e-4 | **3.939** | −0.310 | 3.277e8 | 4618 |

token-matched @1.638e8：ba 4.248 / lr6 4.182 / bs64@10k **4.168** / bs64_lr6@10k **4.122**

SwanLab: https://swanlab.cn/@07011812138/cs336-owt

**Q3** `experiments/results/owt/q3_generate.{csv,md}`（T=0.8 top-k=50 top-p=0.9 max_new=128 seed=0）
- 同 decode 三 prompt：TS 故事腔连贯；OWT 重复套话 / 新闻案情腔 / 一处早 EOS

### Oral answer attempt (user) · Q1
- 第一轮：不可以直接比；语料难度和大小完全不同；OWT loss≈TS 三倍 — **partial**
- 第二轮：网页数据相对故事 **多样性/难度更高** — **solid**（去掉了易被打穿的「大小=训得少」）

### Oral answer attempt (user) · Q2
- 第一轮：试了 lr6e-4、bs64；同 step bs64 更明显 — **partial**
- 第二轮：最好是 **bs64+6e-4**；同 tokens 下 bs64 仍优于 ba — **solid**
- 可顺带记：同 token 组合格 4.122 < 单 bs64 4.168 < ba 4.248

### Oral answer attempt (user) · Q3
- 用户：训数不同；OWT 偏新闻风，与故事不符；易重复、短句
- 判定: **solid**（对齐样例：套话重复 / 早 EOS / 案情腔；归因数据分布而非解码）
- 可选半句：同算力下 OWT 更难拟合（valid≈4.25），流畅度差与高 loss 同向

### Gaps still open
- [ ] Obsidian: off
- [ ] 下一轮 Discover: `experiments/DISCOVERY_owt2.md` — **墙钟反推已写入 derived CSV；待选 Q1+Q2 进 Freeze**

### Obsidian
- skipped
