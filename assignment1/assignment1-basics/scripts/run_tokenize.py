"""
独立脚本：把文本编成 uint16 .npy，带进度条。
加速做在本脚本内（ranks + pretoken cache），不改 get_tokenizer.py。

用法：
  cd assignment1/assignment1-basics
  uv run python scripts/run_tokenize.py

---------------------------------------------------------------------------
为什么作业里的 Tokenizer.encode 很慢？
---------------------------------------------------------------------------
慢路径（get_tokenizer.Tokenizer.encode）对每个 pre-token 大致是：

    for merge in merges:          # 约 1 万条
        扫一遍当前字节序列，能合就合

代价 ≈ (#pre-token) × (#merges) × (词长)。
多数 merge 轮根本合不上，却仍要空跑完整表 → 全库 tokenize 会到「几天」量级。

本脚本两条加速，语义与慢路径对齐（启动时会对照 Tokenizer.encode 做 sanity）：

1) merge ranks（改算法，不是偷工减料）
   - 预训练 merges 是有序的：越靠前越先学到，优先级越高。
   - 建成 ranks[(a,b)] = 序号后，对一个 word 只反复做：
       「在当前相邻 pair 里，选 rank 最小的那一对合掉」
     直到没有可合 pair。
   - 这与「按 merges 顺序依次扫」等价（BPE 标准实现 / tiktoken 思路），
     但复杂度从 O(#merges × 词长) 变成大约 O(词长²)。
     TinyStories 的 pre-token 很短，所以快几个数量级。

2) pretoken cache（利用语料重复）
   - 同一字符串（如 " the"、" once"）在语料里出现极多次。
   - 第一次算完 id 列表后放进 dict；之后命中直接复用。
   - 不改变任何结果，只避免重复做 BPE。
---------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import regex as re
from tqdm import tqdm

from cs336_basics.get_tokenizer import Tokenizer

DEFAULT_VOCAB_DIR = Path("/data1/wcz/projects/Myllm-runs/tokenizers/tinystories")
DEFAULT_DATA_DIR = Path("/data1/wcz/datasets/myllm/tinystories")
DEFAULT_OUT_DIR = Path("/data1/wcz/datasets/myllm/tokenized")

SPLITS = {
    "train": "TinyStoriesV2-GPT4-train.txt",
    "valid": "TinyStoriesV2-GPT4-valid.txt",
}

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class FastEncoder:
    """与 Tokenizer.encode 同语义；用 merge ranks + word cache 加速。"""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.special_tokens = special_tokens or []
        # ranks：pair → 训练时的合并优先级（越小越先合）。
        # 慢路径是 for merge in merges 顺序扫；这里用查表 + 每次取最小 rank，等价且更快。
        self.ranks = {pair: i for i, pair in enumerate(merges)}
        # pretok cache：str → 已算好的 token id 列表。命中则跳过整段 BPE。
        self.cache: dict[str, list[int]] = {}
        self.pat = re.compile(GPT2_PAT)

        # special 按长度从长到短，避免短的 <|endoftext|> 抢先匹配重叠串
        sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
        if sorted_specials:
            self.special_pattern = "(" + "|".join(re.escape(t) for t in sorted_specials) + ")"
        else:
            self.special_pattern = None

    def _encode_word(self, word: str) -> list[int]:
        # --- 加速 2：相同 pre-token 只 BPE 一次 ---
        cached = self.cache.get(word)
        if cached is not None:
            return cached

        # 与慢路径相同：先拆成单字节 bytes
        parts = [bytes([b]) for b in word.encode("utf-8")]

        # --- 加速 1：ranks 贪心合并 ---
        # 慢：for 每一条 merge（~1e4），扫整词看能不能合（大量空转）。
        # 快：每轮只看「当前词里真实存在的相邻 pair」，选 ranks 最小的合一次。
        #     因为 merges 本身按学习顺序编号，最小 rank = 本该最先应用的那条，
        #     所以结果与按表顺序应用一致，但不会对用不上的 merge 空跑。
        while len(parts) >= 2:
            best_i = -1
            best_rank = None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            # 当前没有任何 pair 出现在 merges 里 → 该 word 的 BPE 结束
            if best_i < 0:
                break
            # 不重叠合并：合 best_i 与 best_i+1，拼成新 bytes token
            parts = parts[:best_i] + [parts[best_i] + parts[best_i + 1]] + parts[best_i + 2 :]

        ids = [self.reverse_vocab[p] for p in parts]
        self.cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        # 流程对齐 Tokenizer.encode：先按 special 切开，再 GPT-2 regex 出 pre-token
        if self.special_pattern:
            parts = re.split(self.special_pattern, text)
        else:
            parts = [text]
        tokens: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                tokens.append(self.reverse_vocab[part.encode("utf-8")])
            else:
                for word in self.pat.findall(part):
                    tokens.extend(self._encode_word(word))
        return tokens

    def encode_lines(self, lines):
        for line in lines:
            yield from self.encode(line)


def load_fast_encoder(vocab_dir: Path, special_tokens: list[str]) -> FastEncoder:
    vocab = pickle.load(open(vocab_dir / "vocab.pkl", "rb"))
    merges = pickle.load(open(vocab_dir / "merges.pkl", "rb"))
    max_id = max(vocab.keys())
    if max_id > np.iinfo(np.uint16).max:
        raise SystemExit(f"vocab id {max_id} 超出 uint16，不能存 .npy")
    print(f"tokenizer: vocab={len(vocab)} merges={len(merges)} max_id={max_id}")
    enc = FastEncoder(vocab, merges, special_tokens)

    # 与作业 Tokenizer 对照一小段，防止加速写歪
    ref = Tokenizer(vocab, merges, special_tokens)
    probe = "Once upon a time, there was a little girl.<|endoftext|>Hello!\n"
    a, b = enc.encode(probe), ref.encode(probe)
    if a != b:
        raise SystemExit(f"fast encode 与 Tokenizer 不一致:\nfast={a}\nref ={b}")
    print("sanity: FastEncoder == Tokenizer.encode OK")
    return enc


def tokenize_file(encoder: FastEncoder, input_path: Path, out_path: Path, chunk_tokens: int) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".bin.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    if out_path.exists():
        out_path.unlink()

    total_bytes = input_path.stat().st_size
    n_tokens = 0
    n_lines = 0
    chunk: list[int] = []
    t0 = time.time()

    print(f"\nencode {input_path.name} ({total_bytes / 1e9:.2f} GB) -> {out_path.name}")
    print(f"(fast: ranks + pretok cache; cache starts empty)")
    with open(input_path, "r", encoding="utf-8") as f_in, open(tmp_path, "wb") as f_out:
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc=input_path.name)
        for line in f_in:
            n_lines += 1
            pbar.update(len(line.encode("utf-8")))
            for token_id in encoder.encode(line):
                chunk.append(token_id)
                n_tokens += 1
                if len(chunk) >= chunk_tokens:
                    np.asarray(chunk, dtype=np.uint16).tofile(f_out)
                    chunk.clear()
        if chunk:
            np.asarray(chunk, dtype=np.uint16).tofile(f_out)
            chunk.clear()
        pbar.close()

    src = np.memmap(tmp_path, dtype=np.uint16, mode="r")
    assert src.shape[0] == n_tokens
    dst = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.uint16, shape=(n_tokens,))
    for i in tqdm(range(0, n_tokens, chunk_tokens), desc="write .npy", unit="chunk"):
        dst[i : i + chunk_tokens] = src[i : i + chunk_tokens]
    dst.flush()
    del dst, src
    tmp_path.unlink(missing_ok=True)

    elapsed = time.time() - t0
    meta = {
        "input": str(input_path),
        "output": str(out_path),
        "num_tokens": n_tokens,
        "num_lines": n_lines,
        "dtype": "uint16",
        "seconds": round(elapsed, 2),
        "tokens_per_sec": int(n_tokens / elapsed) if elapsed > 0 else 0,
        "cache_size": len(encoder.cache),
        "encoder": "FastEncoder(ranks+cache)",
    }
    print(
        f"done: tokens={n_tokens:,} lines={n_lines:,} cache={len(encoder.cache):,} "
        f"in {elapsed:.1f}s ({meta['tokens_per_sec']:,} tok/s) -> {out_path}"
    )
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Tokenize corpus to uint16 .npy (fast, with progress)")
    p.add_argument("--vocab-dir", type=Path, default=DEFAULT_VOCAB_DIR)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--splits", nargs="+", default=["train", "valid"], choices=list(SPLITS.keys()))
    p.add_argument("--chunk-tokens", type=int, default=1_000_000)
    p.add_argument("--special-token", default="<|endoftext|>")
    args = p.parse_args()

    encoder = load_fast_encoder(args.vocab_dir, [args.special_token])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_meta = {}
    t0 = time.time()
    for split in args.splits:
        inp = args.data_dir / SPLITS[split]
        if not inp.is_file():
            raise SystemExit(f"missing: {inp}")
        out = args.out_dir / f"tinystories_{split}.npy"
        all_meta[split] = tokenize_file(encoder, inp, out, args.chunk_tokens)

    all_meta["total_seconds"] = round(time.time() - t0, 2)
    all_meta["vocab_dir"] = str(args.vocab_dir)
    meta_path = args.out_dir / "tinystories_tokenize_meta.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\nmeta -> {meta_path}")
    print(f"all done in {all_meta['total_seconds']}s")


if __name__ == "__main__":
    main()
