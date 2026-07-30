"""lm2 Q1–Q3: param fairness, post_norm curves, scaling table from existing runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def last_valid(log: Path):
    last = None
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "valid_loss" in r:
            last = r
    return last


def valid_series(log: Path):
    out = []
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "valid_loss" in r:
            out.append(r)
    return out


def bucket(series, name_prefix):
    if not series:
        return []
    n = len(series)
    cuts = [
        ("early", series[: max(1, n // 10)]),
        ("mid", series[n // 2 : n // 2 + max(1, n // 10)]),
        ("late", series[-max(1, n // 10) :]),
    ]
    rows = []
    for bname, chunk in cuts:
        vl = [float(r["valid_loss"]) for r in chunk]
        rows.append(
            {
                "run": name_prefix,
                "bucket": bname,
                "n": len(chunk),
                "mean_valid_loss": round(sum(vl) / len(vl), 6),
                "step_lo": chunk[0]["step"],
                "step_hi": chunk[-1]["step"],
                "first_valid_loss": round(float(chunk[0]["valid_loss"]), 6),
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ablation-root",
        type=Path,
        default=Path("/data1/wcz/projects/Myllm-runs/experiments/ablation73"),
    )
    p.add_argument(
        "--scale-root",
        type=Path,
        default=Path("/data1/wcz/projects/Myllm-runs/experiments"),
    )
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Q1 params + loss
    q1 = []
    for run in ["baseline", "silu_ffn"]:
        meta = json.loads((args.ablation_root / run / "run_meta.json").read_text())
        row = last_valid(args.ablation_root / run / "log.jsonl")
        q1.append(
            {
                "run": run,
                "params": meta["params"],
                "params_M": round(meta["params"] / 1e6, 4),
                "valid_loss": round(float(row["valid_loss"]), 6),
                "valid_ppl": round(float(row["valid_ppl"]), 4),
            }
        )
    base = q1[0]["valid_loss"]
    for r in q1:
        r["delta_vs_baseline"] = round(r["valid_loss"] - base, 6)
        r["params_delta"] = r["params"] - q1[0]["params"]
    q1_path = args.out_dir / "q1_silu_fairness.csv"
    with open(q1_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(q1[0].keys()))
        w.writeheader()
        w.writerows(q1)
    print("Q1", q1)

    # Q2 curves
    q2_rows = []
    for run in ["baseline", "post_norm", "no_rmsnorm_lr3e4"]:
        q2_rows.extend(bucket(valid_series(args.ablation_root / run / "log.jsonl"), run))
    q2_path = args.out_dir / "q2_norm_stability.csv"
    with open(q2_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["run", "bucket", "n", "mean_valid_loss", "step_lo", "step_hi", "first_valid_loss"],
        )
        w.writeheader()
        w.writerows(q2_rows)
    for r in q2_rows:
        print(f"Q2 {r['run']} {r['bucket']}: mean={r['mean_valid_loss']} first={r['first_valid_loss']}")

    # Q3 scaling
    scale_names = [
        "tinystories_ba",
        "L2_20k",
        "L6_20k",
        "d384_20k",
        "d768_20k",
        "ctx128_20k",
    ]
    q3 = []
    for name in scale_names:
        d = args.scale_root / name
        meta = json.loads((d / "run_meta.json").read_text())
        a = meta["args"]
        row = last_valid(d / "log.jsonl")
        q3.append(
            {
                "run": name,
                "num_layers": a["num_layers"],
                "d_model": a["d_model"],
                "d_ff": a["d_ff"],
                "context_length": a["context_length"],
                "steps": a["steps"],
                "params": meta["params"],
                "params_M": round(meta["params"] / 1e6, 4),
                "valid_loss": round(float(row["valid_loss"]), 6),
                "valid_ppl": round(float(row["valid_ppl"]), 4),
                "wall_s": meta.get("seconds", ""),
            }
        )
    q3.sort(key=lambda r: r["valid_loss"])
    q3_path = args.out_dir / "q3_scaling.csv"
    with open(q3_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(q3[0].keys()))
        w.writeheader()
        w.writerows(q3)
    for r in q3:
        print(
            f"Q3 {r['run']}: L={r['num_layers']} d={r['d_model']} "
            f"params_M={r['params_M']} valid={r['valid_loss']}"
        )

    print("wrote", q1_path, q2_path, q3_path)


if __name__ == "__main__":
    main()
