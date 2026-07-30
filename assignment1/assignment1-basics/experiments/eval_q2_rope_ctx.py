"""Q2: eval baseline vs no_rope checkpoints at multiple context lengths."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path[:0] = [str(ROOT), str(SCRIPTS)]

from ablation_model import AblationFlags, AblationTransformerLM, default_d_ff  # noqa: E402
from cs336_basics.linear import cross_entropy, get_batch  # noqa: E402
from run_train import DEFAULT_VALID  # noqa: E402


def load_model(variant: str, ctx: int, device: str, meta: dict) -> AblationTransformerLM:
    flags = AblationFlags.from_variant(variant)  # type: ignore[arg-type]
    d_model = int(meta["d_model"])
    ffn_type = flags.ffn_type
    d_ff = int(meta.get("d_ff") or default_d_ff(d_model, ffn_type))
    model = AblationTransformerLM(
        int(meta["vocab_size"]),
        ctx,
        d_model,
        int(meta["num_layers"]),
        int(meta["num_heads"]),
        d_ff,
        float(meta.get("rope_theta", 10000.0)),
        flags,
    ).to(device)
    return model


@torch.no_grad()
def eval_valid(model, data, batch_size, context_length, device, num_batches) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(data, batch_size, context_length, device)
        x, y = x.long(), y.long()
        logits = model(x)
        losses.append(cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    return float(sum(losses) / len(losses))


def read_meta(run_dir: Path) -> dict:
    mp = run_dir / "ablation_manifest.json"
    if mp.is_file():
        hp = json.loads(mp.read_text()).get("hyperparams", {})
        return hp
    return {
        "vocab_size": 10000,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "context_length": 256,
        "rope_theta": 10000.0,
        "d_ff": None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-ckpt", type=Path, required=True)
    p.add_argument("--no-rope-ckpt", type=Path, required=True)
    p.add_argument("--contexts", default="64,128,256,512")
    p.add_argument("--valid-data", type=Path, default=DEFAULT_VALID)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=40)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
    valid = np.load(args.valid_data, mmap_mode="r")

    specs = [
        ("baseline", args.baseline_ckpt),
        ("no_rope", args.no_rope_ckpt),
    ]

    rows = []
    base_by_ctx: dict[int, float] = {}

    for variant, ckpt in specs:
        run_dir = ckpt.parent
        meta = read_meta(run_dir)
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob

        for ctx in contexts:
            model = load_model(variant, ctx, device, meta)
            model.load_state_dict(state)
            vloss = eval_valid(model, valid, args.batch_size, ctx, device, args.eval_batches)
            vppl = math.exp(min(vloss, 20))
            row = {
                "variant": variant,
                "context_length": ctx,
                "valid_loss": round(vloss, 6),
                "valid_ppl": round(vppl, 4),
                "ckpt": str(ckpt),
            }
            if variant == "baseline":
                base_by_ctx[ctx] = vloss
                row["delta_vs_baseline"] = 0.0
            else:
                row["delta_vs_baseline"] = round(vloss - base_by_ctx[ctx], 6)
            rows.append(row)
            print(
                f"{variant} ctx={ctx}: valid_loss={vloss:.4f} "
                f"Δ={row['delta_vs_baseline']} ppl={vppl:.3f}",
                flush=True,
            )
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    # fill baseline deltas already 0; ensure no_rope rows computed after baseline for each ctx
    # (loop order is baseline all ctx then no_rope — wait, current loop is variant outer, ctx inner
    # so baseline fills all ctx first, then no_rope — good)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["variant", "context_length", "valid_loss", "valid_ppl", "delta_vs_baseline", "ckpt"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
