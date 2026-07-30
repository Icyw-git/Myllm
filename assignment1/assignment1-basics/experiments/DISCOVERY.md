## Discovery · 2026-07-29 · A1 Tokenizer / BPE

### Context
- claim / hypothesis: OWT byte-level BPE（vocab=32k）已训完；早期 merge 慢、后期加速可解释；下一步应用侧（压缩率 / encode 吞吐）与训练侧（affected 曲线）应用数字抗追问。
- implementation notes:
  - 训练脚本（加速路径）: `scripts/run_train_bpe.py --fast`
  - 产出: `/data1/wcz/projects/Myllm-runs/tokenizers/openwebtext-v2`（vocab=32000, merges=31743）
  - 总墙钟: `done 8700.3s`；merge 段约 `1:43:17`；早期 ~12–19 s/merge，末期 ~40–57 merge/s
  - types≈6.6e6；日志含 `affected=` / `freq=` / `pair=`
  - encode: `cs336_basics/get_tokenizer.py` + `scripts/run_tokenize.py`（FastEncoder）
  - 对照语料: `owt_valid.txt`；`tests/fixtures/tinystories_sample_5M.txt`；可选 TinyStories tokenizer 目录
- skip_discover: false
- Obsidian: off（本轮）

### Blindspot candidates (3–5)
1. 训练墙钟：早期慢 / 后期快是否真由 `affected` 主导？能否用日志画出 step→s/merge、affected 曲线？
2. OWT 32k tokenizer 在 **域外**（TinyStories / 中文）压缩率掉多少？是经验还是测过？
3. 作业 `Tokenizer.encode` vs `run_tokenize.FastEncoder` 在 OWT valid 上的 tok/s 差几个数量级？
4. 同数据 vocab 8k/16k/32k 的 bytes/token 边际收益是否饱和？（需重训或已有 tinystories 对照）
5. 与 GPT-2 merges 在同一段英文上的压缩对比：自己的 32k 赢在哪、输在哪？

### Selected (按面试杀伤力自选 1–2)
- Q1 ← candidate #2（压缩率跨域）
- Q2 ← candidate #1（训练时间–affected 证据链）
-（P2 候选）Q3 ← candidate #3（encode 吞吐）— overnight_worthy 视预算

### Cards

#### Q1 · P5
- 追问问题: 你的 OWT 32k byte-BPE 在 TinyStories / 中文上压缩率相对 OWT valid 差多少？你报的数字是测的还是猜的？
- 为什么这是个好问题: 训练完只会报「vocab=32k」不够；面试必打跨域压缩与 CJK 碎片化。
- 预期答案方向: 应用自己的表：`bytes/token` 或 `chars/token`；英文域内 vs 故事域 vs 中文；能解释「未见过的字符组合 → 更短 merge 命中」。
- what_it_tests: 训好的 tokenizer 的域泛化 / 压缩度量是否落地。
- expected_if_matters: OWT→TinyStories 略差或接近；中文明显更差（更高 bytes/token）。
- 最小实验方案:
  - 固定样本：`owt_valid` 截断 N MB；`tinystories_sample_5M.txt`；自备中文段落（或作业 german/其他非拉丁若无中文则用非英文 fixture + 自写中文）
  - 命令方向: 用已有 `scripts/run_tokenize.py` 或短脚本加载 `openwebtext-v2`，统计 `total_bytes / n_tokens`
  - ETA: ≤30–60 min（含写表）
- 验收标准: 一张表 ≥2 个域，能指着数字回答追问；禁止只说「中文不友好」。
- overnight_worthy: yes（可扩展到多域 + GPT-2 对照 = P2）
- suggested_knobs: sample_bytes ∈ {1e6, 5e6, 2e7}；tokenizer_dir；可选对比 `tokenizers/tinystories`

#### Q2 · P4
- 追问问题: 为什么你的 OWT BPE 训练「越来越快」？早期 15s/merge、末期几十 merge/s，瓶颈变量是什么？
- 为什么这是个好问题: 区分「我看过进度条」vs「能用 affected/freq 解释复杂度」。
- 预期答案方向: 每步成本 ∝ `affected`（含该 pair 的 unique types）；早期高频 pair 触达大量 type；后期稀有 pair；不是「heap 突然变快」。
- what_it_tests: 训练复杂度直觉是否与实现一致。
- expected_if_matters: 从 `owt-bpe-train-v2.log` 抽样 step 的 s/merge 与 affected 正相关。
- 最小实验方案:
  - 解析日志：`BPE merge:` 行 → `(step, s/merge 或时间差, affected, freq, pair)`
  - 输出：前 50 / 中段 / 末 50 的汇总表 + 散点或分桶均值
  - ETA: ≤30–45 min
- 验收标准: 能指着自己的表说「早期 affected~1e5–4e5 → 秒级；末期 affected~1–1e3 → 毫秒级」。
- overnight_worthy: yes（短，但弹药硬）
- suggested_knobs: log path；sample_every ∈ {1, 10, 100}

#### Q3 · P3（可选 P2）
- 追问问题: 作业版 `Tokenizer.encode` 在 OWT 上有多慢？你的 FastEncoder 快多少？快在哪一步？
- 为什么这是个好问题: A1 常见追问 encode 路径；有现成 `run_tokenize.py`。
- 预期答案方向: 对比同一文本墙钟 / tok/s；指向 pretoken 后的逐 word merge vs rank+cache。
- what_it_tests: encode 实现瓶颈是否可复述。
- expected_if_matters: FastEncoder ≫ 作业 encode（数量级差需自己测）。
- 最小实验方案: `run_tokenize.py` 对 `owt_valid` 截断；可选再跑作业 Tokenizer 小样本对照。
- 验收标准: 并列表 + 一句「瓶颈在哪」。
- overnight_worthy: partial（≤1h 也可轨 A）
- suggested_knobs: encoder ∈ {ref, fast}；input_bytes

### Overnight shortlist
| Q | overnight_worthy | why_P1 | rough_ETA |
| --- | --- | --- | --- |
| Q1 | yes | 训完后最缺的应用侧数字；跨域压缩面试高频 | 0.5–1h |
| Q2 | yes | 把本次 8700s 跑次变成可口述的证据链 | ≤45m |
| Q3 | partial | encode 弹药；可作 +1 P2 | ≤1h |

### Handoff to Freeze
- proposed metric + direction:
  - Q1: `bytes_per_token`（lower better）按 domain
  - Q2: 相关（Pearson 或分桶）`affected` vs `seconds_per_merge`；或报告分位表
  - Q3: `tokens_per_sec`（higher better）
- proposed eval_command: （Freeze 时写死具体脚本调用）
- editable_scope hint: `scripts/` 测量脚本 + `experiments/` 分析；默认不改 `cs336_basics/train_bpe.py`
- forbidden: 静默重训改 vocab 定义 / 改 merge 平局规则刷数；Obsidian off
