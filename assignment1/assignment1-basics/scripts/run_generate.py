#!/usr/bin/env python3
"""
从 baseline checkpoint 自回归生成文本，支持 temperature + top-k + top-p。

用法：
  cd assignment1/assignment1-basics
  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/run_generate.py \\
    --ckpt /data1/wcz/projects/Myllm-runs/experiments/baseline_20k/ckpt_step20000.pt \\
    --prompt "Once upon a time" \\
    --max-new-tokens 256 \\
    --top-k 50 --top-p 0.95 --temperature 0.9
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch
import torch.nn.functional as F

from cs336_basics.get_tokenizer import Tokenizer
from run_train import TransformerLM

DEFAULT_VOCAB_DIR = Path("/data1/wcz/projects/Myllm-runs/tokenizers/tinystories")


def load_tokenizer(vocab_dir: Path) -> Tokenizer:
    meta = json.loads((vocab_dir / "meta.json").read_text())
    vocab = pickle.load(open(vocab_dir / "vocab.pkl", "rb"))
    merges = pickle.load(open(vocab_dir / "merges.pkl", "rb"))
    return Tokenizer(vocab, merges, meta.get("special_tokens", ["<|endoftext|>"]))


def sample_next(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    """logits: (vocab_size,)"""
    if temperature <= 0:
        return int(logits.argmax().item())

    logits = logits.float() / temperature

    if top_k > 0:
        k = min(top_k, logits.numel())
        thresh = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < thresh, float("-inf"))

    probs = F.softmax(logits, dim=-1)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # 保留 nucleus：累计概率首次超过 top_p 的最小集合
        drop = cumsum - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        if sorted_probs.sum() <= 0:
            return int(probs.argmax().item())
        probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)
        probs = probs / probs.sum()

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_id: int | None,
    device: str,
) -> list[int]:
    model.eval()
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        ctx = ids[-context_length:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]
        next_id = sample_next(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
    return ids


def main() -> None:
    p = argparse.ArgumentParser(description="Generate with baseline LM (top-k / top-p)")
    p.add_argument("--ckpt", type=Path, required=True, help="baseline run_train.py 保存的 ckpt_*.pt")
    p.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_VOCAB_DIR)
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50, help="0 表示不截断 top-k")
    p.add_argument("--top-p", type=float, default=0.95, help="1.0 表示不用 nucleus")
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    tok = load_tokenizer(args.tokenizer_dir)
    eos_id = tok.reverse_vocab.get("<|endoftext|>".encode("utf-8"))

    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"loaded {args.ckpt} (step {ckpt.get('iteration', '?')})")

    prompt_ids = tok.encode(args.prompt)
    out_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=eos_id,
        device=device,
    )
    text = tok.decode(out_ids)
    print("--- prompt ---")
    print(args.prompt)
    print("--- generated ---")
    print(text)
    print("---")
    print(f"tokens: prompt={len(prompt_ids)} total={len(out_ids)}  top_k={args.top_k} top_p={args.top_p} T={args.temperature}")


if __name__ == "__main__":
    main()
