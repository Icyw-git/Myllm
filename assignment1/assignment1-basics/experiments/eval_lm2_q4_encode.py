"""lm2 Q4: Tokenizer.encode vs FastEncoder throughput on a fixed byte budget."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs336_basics.get_tokenizer import Tokenizer  # noqa: E402
from scripts.run_tokenize import FastEncoder  # noqa: E402


def load(tokenizer_dir: Path):
    with open(tokenizer_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(tokenizer_dir / "merges.pkl", "rb") as f:
        merges = pickle.load(f)
    special = ["<|endoftext|>"]
    meta = tokenizer_dir / "meta.json"
    if meta.is_file():
        special = json.loads(meta.read_text()).get("special_tokens", special)
    return vocab, merges, special


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("/data1/wcz/projects/Myllm-runs/tokenizers/tinystories"),
    )
    p.add_argument(
        "--text-path",
        type=Path,
        default=Path(
            "/data1/wcz/projects/Myllm/assignment1/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt"
        ),
    )
    p.add_argument("--max-bytes", type=int, default=50_000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    raw = args.text_path.read_bytes()[: args.max_bytes]
    text = raw.decode("utf-8", errors="replace")
    vocab, merges, special = load(args.tokenizer_dir)
    ref = Tokenizer(vocab, merges, special)
    fast = FastEncoder(vocab, merges, special)

    # sanity on tiny prefix
    t0 = text[:2000]
    assert ref.encode(t0) == fast.encode(t0), "FastEncoder != Tokenizer on sanity slice"

    rows = []
    for name, enc in [("ref_Tokenizer", ref), ("fast_FastEncoder", fast)]:
        t0 = time.perf_counter()
        ids = enc.encode(text)
        dt = time.perf_counter() - t0
        n = len(ids)
        rows.append(
            {
                "encoder": name,
                "bytes": len(text.encode("utf-8")),
                "tokens": n,
                "runtime_s": round(dt, 4),
                "tokens_per_sec": round(n / dt, 1) if dt > 0 else None,
                "bytes_per_sec": round(len(text.encode("utf-8")) / dt, 1) if dt > 0 else None,
            }
        )
        print(rows[-1], flush=True)

    speedup = rows[1]["tokens_per_sec"] / rows[0]["tokens_per_sec"]
    print(f"speedup_tok_s={speedup:.1f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
