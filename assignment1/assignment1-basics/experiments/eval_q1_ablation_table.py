"""Q1: summarize §7.3 ablation73 logs → valid_loss table + Δ vs baseline."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def last_valid_row(log_path: Path) -> dict | None:
    last = None
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "valid_loss" in row:
            last = row
    return last


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    runs = []
    for d in sorted(args.root.iterdir()):
        log = d / "log.jsonl"
        if not log.is_file():
            continue
        row = last_valid_row(log)
        if row is None:
            continue
        manifest = {}
        mp = d / "ablation_manifest.json"
        if mp.is_file():
            manifest = json.loads(mp.read_text())
        hp = manifest.get("hyperparams", {})
        runs.append(
            {
                "run": d.name,
                "variant": row.get("variant") or manifest.get("variant") or d.name,
                "max_lr": hp.get("max_lr", ""),
                "step": row["step"],
                "valid_loss": round(float(row["valid_loss"]), 6),
                "valid_ppl": round(float(row["valid_ppl"]), 4),
                "train_loss": round(float(row.get("train_loss", float("nan"))), 6),
            }
        )

    base = next((r for r in runs if r["run"] == "baseline"), None)
    base_loss = base["valid_loss"] if base else None
    for r in runs:
        if base_loss is None:
            r["delta_vs_baseline"] = ""
        else:
            r["delta_vs_baseline"] = round(r["valid_loss"] - base_loss, 6)

    runs.sort(key=lambda r: r["valid_loss"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "variant",
                "max_lr",
                "step",
                "valid_loss",
                "valid_ppl",
                "train_loss",
                "delta_vs_baseline",
            ],
        )
        w.writeheader()
        w.writerows(runs)

    print(f"baseline valid_loss={base_loss}")
    for r in runs:
        print(
            f"{r['run']}: valid={r['valid_loss']:.4f} Δ={r['delta_vs_baseline']} ppl={r['valid_ppl']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
