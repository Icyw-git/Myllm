#!/usr/bin/env python3
"""Q3: side-by-side generate TinyStories ba vs OWT ba (fixed prompts / decode)."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))

import torch

from cs336_basics.get_tokenizer import Tokenizer
from run_generate import generate, load_tokenizer
from run_train import TransformerLM

PROMPTS = [
    "Once upon a time",
    "The company announced that",
    "In a small village,",
]

RUNS = {
    "tinystories_ba": {
        "ckpt": Path("/data1/wcz/projects/Myllm-runs/experiments/tinystories_ba/ckpt_step20000.pt"),
        "tokenizer_dir": Path("/data1/wcz/projects/Myllm-runs/tokenizers/tinystories"),
        "vocab_size": 10_000,
    },
    "owt_ba_20k": {
        "ckpt": Path("/data1/wcz/projects/Myllm-runs/experiments/owt_ba_20k/ckpt_step20000.pt"),
        "tokenizer_dir": Path("/data1/wcz/projects/Myllm-runs/tokenizers/openwebtext-v2"),
        "vocab_size": 32_000,
    },
}


def load_model(cfg: dict, device: str) -> TransformerLM:
    model = TransformerLM(
        cfg["vocab_size"], 256, 512, 4, 16, 1344, 10000.0
    ).to(device)
    ckpt = torch.load(cfg["ckpt"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("experiments/results/owt/q3_generate.csv"))
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    md_lines = [
        "# Q3 generate · TS ba vs OWT ba",
        "",
        f"decode: T={args.temperature} top_k={args.top_k} top_p={args.top_p} "
        f"max_new={args.max_new_tokens} seed={args.seed}",
        "",
    ]

    for run_name, cfg in RUNS.items():
        print(f"load {run_name} …")
        tok = load_tokenizer(cfg["tokenizer_dir"])
        eos_id = tok.reverse_vocab.get("<|endoftext|>".encode("utf-8"))
        model = load_model(cfg, device)
        md_lines.append(f"## {run_name}")
        md_lines.append("")
        for prompt in PROMPTS:
            torch.manual_seed(args.seed)
            ids = generate(
                model,
                tok.encode(prompt),
                max_new_tokens=args.max_new_tokens,
                context_length=256,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_id=eos_id,
                device=device,
            )
            text = tok.decode(ids)
            # strip prompt prefix display mess — keep full decode
            rows.append(
                {
                    "run": run_name,
                    "prompt": prompt,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "n_tokens": len(ids),
                    "text": text.replace("\n", "\\n"),
                }
            )
            md_lines.append(f"### prompt: `{prompt}`")
            md_lines.append("")
            md_lines.append("```")
            md_lines.append(text[:2000])
            md_lines.append("```")
            md_lines.append("")
            print(f"  [{run_name}] {prompt!r} -> {len(ids)} tok")
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = args.out.with_suffix(".md")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
