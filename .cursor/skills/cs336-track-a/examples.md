# Examples



## Example A — FlashAttention (manual)



**User:** 开始 A2 FlashAttention 的抗追问扫描。我手写了 Triton FA-2，BM=64，BN=64，在 A100 上 throughput 测过，但对 SRAM/occupancy 不熟。



**Step 0:** Ask mode → user picks `manual`.



**Step 1 blindspot (sample):**



1. Online softmax 为什么数值稳定？你测过 bf16 vs fp32 accumulator 吗？

2. BM=64 的选择依据是什么？换 128 时 occupancy / spill 怎么变？

3. 你知道 shared memory 上限怎么约束 BM×BN 吗？

4. Causal mask 下 tile 调度和 non-causal 差在哪？

5. 你的 kernel 有没有用 nsys 看过 warp stall 原因？



User picks **2**.



**Step 2 card (abbrev):**



- 追问: 你的 FlashAttention block size 为什么选 64？换 128 会怎样？

- 方向: 应能把 BM 与 SRAM、occupancy、可能的 register spill 联系起来，并引用自己的 ms 表

- 实验: `uv run bench_flashattn.py --bm 32,64,128,256`（按仓库实际命令改）· ETA ~30m · nsys 可选

- 验收: 能指着自己的表说清哪一档最快、哪一档变慢、猜测原因对应哪类硬件现象



**Step 3–4:** User pastes ms table → append log → ask them to answer aloud.



**Incomplete oral:** User says「128 慢了，可能是内存问题」→ 引导环：点名缺口「没连上 occupancy/spill」→ 追问「表里 128 那行相对 64 慢多少？更像 shared 打满还是 register spill？」→ 第二轮口述后再写 Obsidian `status: partial|solid`.



## Example B — Tokenizer (auto)



**User:** A1 Tokenizer 抗追问。Byte-level BPE，vocab 10k，英文 TinyStories 训的。中文压缩率我没测过。auto 模式，预算 1h。



**Blindspot pick:** 中文压缩率相对英文差多少？



**Card:**



- 追问: 你的 byte-level BPE 在中文上压缩率大概差多少？30%–50% 是经验还是你测过的？

- 实验: 固定样本中英各 N 篇，统计 `bytes/token` 与 `chars/token`；可选对照 SentencePiece

- auto: agent runs the eval script, writes numbers to EXPERIMENT_LOG.md, then asks user to interpret without spoon-feeding the “right” narrative

- Obsidian: `CS自学/Diy-llm/抗追问/CS336-A1-抗追问-中文压缩率.md`



## Example C — Mode switch mid-session



If user started `manual` then says「你直接跑吧」→ switch to `auto` for the next card only; restate what will be executed before running.



## Example D — Auto + ablation plugin



**User:** A2 抗追问，auto，用 ablation plugin，预算 45m。



**Card 参数范围:** BM ∈ {32,64,128,256}，先测 throughput，ETA 单点 ~5m。



**执行链:**



1. Tutor emits card（最小命令先跑 BM=64 作基线）。

2. 加载全局 `minimal-ablation-proposer`（注入 `repo-hooks.md` 的模板）：通常只提议下一条（如 `BM=128`），不是一次扫完。

3. Plugin 若提议换到另一模块或开 nsys 全日 profile → tutor **否决**，说明原因。

4. 结果入库 → 逼口述 → 不完整则引导环 → 写入 Obsidian。

**A1 §7.3 变体:** 见 `repo-hooks.md`；proposer 只在 card/`param_range` 内选下一点。



## Example E — Guidance then Obsidian



**Oral round 1:** 「中文肯定更差，大概差一半吧。」（无表）



**引导:** 「看你刚跑的 `chars/token` 那列，中英比值是多少？用那个数重说一遍。」



**Oral round 2:** 「英文 4.2 chars/token，中文 1.6，所以不是差一半，是……」



**Obsidian §5 锚点:** 必须引用 4.2 / 1.6，禁止改写成「BPE 对 CJK 天生不友好」这类无数字金句。

---

轨 B 通宵流水线不在本文件；见 `../cs336-track-b/`。

