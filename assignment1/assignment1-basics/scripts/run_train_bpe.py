"""
独立脚本：用你的 BPE 实现训词表，带 tqdm 进度条。
不改 cs336_basics/train_bpe.py。

用法：
  cd assignment1/assignment1-basics
  uv run python scripts/run_train_bpe.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import regex as re
from tqdm import tqdm

from cs336_basics.train_bpe import BPE


class BPEWithProgress(BPE):
    """与 BPE.train 同逻辑，仅多了阶段日志 + merge 进度条。"""

    def train(self, input_path: str | Path, vocab_size: int, **kwargs):
        print(f"[1/4] 读入 {input_path} ...", flush=True)
        t0 = time.time()
        with open(input_path, "r") as f:
            text = f.read()
        print(f"      文本 {len(text) / 1e6:.1f}M chars，{time.time() - t0:.1f}s", flush=True)

        print("[2/4] pretokenize ...", flush=True)
        t1 = time.time()
        parts = re.split(self.special_pattern, text)
        words = []
        for part in parts:
            words.extend(re.findall(self.regex, part))
        print(f"      {len(words):,} words，{time.time() - t1:.1f}s", flush=True)

        vocab = {}
        for i in range(256):
            vocab[i] = bytes([i])
        for token in self.special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")

        print("[3/4] 建 word_freq / pair 表 ...", flush=True)
        t2 = time.time()
        word_freq = Counter()
        for word in tqdm(words, desc="count words", unit="w"):
            tokens = [bytes([b]) for b in word.encode("utf-8")]
            word_freq[tuple(tokens)] += 1

        pair_counts = Counter()
        pair_to_words = defaultdict(set)
        for tokens, freq in tqdm(word_freq.items(), desc="init pairs", unit="type"):
            for i in range(len(tokens) - 1):
                p = (tokens[i], tokens[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(tokens)
        print(
            f"      unique={len(word_freq):,} pairs={len(pair_counts):,}，{time.time() - t2:.1f}s",
            flush=True,
        )

        merges = []
        num_merges = vocab_size - len(vocab)
        print(f"[4/4] merge × {num_merges} (目标 vocab={vocab_size}) ...", flush=True)

        pbar = tqdm(total=num_merges, desc="BPE merge", unit="merge")
        while len(vocab) < vocab_size:
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=lambda x: (pair_counts[x], x))
            best_freq = pair_counts[best_pair]
            new_token = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab[len(vocab)] = new_token

            affected = list(pair_to_words[best_pair])
            for tokens in affected:
                freq = word_freq[tokens]
                for i in range(len(tokens) - 1):
                    p = (tokens[i], tokens[i + 1])
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    pair_to_words[p].discard(tokens)

                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(tokens[i] + tokens[i + 1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_tokens = tuple(new_tokens)

                for i in range(len(new_tokens) - 1):
                    p = (new_tokens[i], new_tokens[i + 1])
                    pair_counts[p] += freq
                    pair_to_words[p].add(new_tokens)

                word_freq[new_tokens] = word_freq.get(new_tokens, 0) + freq
                del word_freq[tokens]

            pair_to_words[best_pair].clear()
            pbar.set_postfix(pair=repr(new_token)[:24], freq=best_freq, types=len(word_freq))
            pbar.update(1)
        pbar.close()
        return vocab, merges


def save(vocab, merges, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(out_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(
            {
                "vocab_size": len(vocab),
                "num_merges": len(merges),
                "special_tokens": ["<|endoftext|>"],
            },
            f,
            indent=2,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Train BPE with progress bar")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("/data1/wcz/datasets/myllm/tinystories/TinyStoriesV2-GPT4-train.txt"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data1/wcz/projects/Myllm-runs/tokenizers/tinystories"),
    )
    p.add_argument("--vocab-size", type=int, default=10_000)
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"missing: {args.input}")

    print(f"input={args.input} ({args.input.stat().st_size / 1e9:.2f} GB)")
    print(f"out={args.out_dir}  vocab_size={args.vocab_size}")

    t0 = time.time()
    bpe = BPEWithProgress(special_tokens=["<|endoftext|>"])
    vocab, merges = bpe.train(args.input, args.vocab_size)
    save(vocab, merges, args.out_dir)
    print(f"done {time.time() - t0:.1f}s | vocab={len(vocab)} merges={len(merges)}")
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
