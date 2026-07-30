"""Q1 frozen eval: bytes/token across domains for a trained BPE tokenizer."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tokenize import FastEncoder  # noqa: E402
DEFAULTS = {
    "owt": Path("/data1/wcz/datasets/myllm/openwebtext/owt_valid.txt"),
    "tinystories": Path(
        "/data1/wcz/datasets/myllm/tinystories/TinyStoriesV2-GPT4-valid.txt"
    ),
    "zh": ROOT / "experiments/fixtures/zh_sample.txt",
}


def load_encoder(tokenizer_dir: Path) -> FastEncoder:
    with open(tokenizer_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(tokenizer_dir / "merges.pkl", "rb") as f:
        merges = pickle.load(f)
    meta_path = tokenizer_dir / "meta.json"
    special = ["<|endoftext|>"]
    if meta_path.is_file():
        special = json.loads(meta_path.read_text()).get("special_tokens", special)
    return FastEncoder(vocab, merges, special)


def read_prefix(path: Path, max_bytes: int) -> str:
    raw = path.read_bytes()
    if path.name == "zh_sample.txt":
        # frozen fixture: always full file (strip comment lines)
        text = path.read_text(encoding="utf-8")
        return "\n".join(
            ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
        )
    chunk = raw[:max_bytes]
    return chunk.decode("utf-8", errors="replace")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--max-bytes", type=int, default=5_000_000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    enc = load_encoder(args.tokenizer_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for domain, path in DEFAULTS.items():
        if not path.is_file():
            raise SystemExit(f"missing domain file: {domain} -> {path}")
        text = read_prefix(path, args.max_bytes)
        b = len(text.encode("utf-8"))
        chars = len(text)
        t0 = time.time()
        ids = enc.encode(text)
        runtime = time.time() - t0
        n = len(ids)
        rows.append(
            {
                "domain": domain,
                "bytes": b,
                "chars": chars,
                "tokens": n,
                "bytes_per_token": round(b / n, 4) if n else float("nan"),
                "chars_per_token": round(chars / n, 4) if n else float("nan"),
                "runtime_s": round(runtime, 3),
            }
        )
        print(
            f"{domain}: bytes/token={rows[-1]['bytes_per_token']} "
            f"chars/token={rows[-1]['chars_per_token']} tokens={n} ({runtime:.1f}s)",
            flush=True,
        )

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "domain",
                "bytes",
                "chars",
                "tokens",
                "bytes_per_token",
                "chars_per_token",
                "runtime_s",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
