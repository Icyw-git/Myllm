"""lm4 Q1–Q3: warmup / lr×batch grid / compound scaling attribution."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RUNS = Path("/data1/wcz/projects/Myllm-runs/experiments")


def load_rows(log: Path) -> list[dict]:
    if not log.is_file():
        return []
    return [json.loads(l) for l in log.read_text().splitlines() if l.strip()]


def last_valid(rows: list[dict]) -> dict | None:
    last = None
    for r in rows:
        if "valid_loss" in r:
            last = r
    return last


def meta_args(run_dir: Path) -> dict:
    p = run_dir / "run_meta.json"
    return json.loads(p.read_text()).get("args", {}) if p.is_file() else {}


def tokens(step: int, bs: int, ctx: int) -> int:
    return step * bs * ctx


def bucket_means(rows: list[dict], key: str = "valid_loss") -> dict:
    series = [r for r in rows if key in r]
    if not series:
        return {"early": None, "mid": None, "late": None}
    n = len(series)
    cuts = {
        "early": series[: max(1, n // 10)],
        "mid": series[n // 2 : n // 2 + max(1, n // 10)],
        "late": series[-max(1, n // 10) :],
    }
    return {k: round(sum(r[key] for r in v) / len(v), 6) for k, v in cuts.items()}


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def q1_warmup(out: Path) -> None:
    rows_out = []
    for name, note in [("tinystories_ba", "warmup=200"), ("warmu", "warmup=0")]:
        d = RUNS / name
        args = meta_args(d)
        rows = load_rows(d / "log.jsonl")
        lv = last_valid(rows)
        b = bucket_means(rows)
        # also early train loss spike
        trains = [r["train_loss"] for r in rows[: max(1, len(rows) // 20)]]
        rows_out.append(
            {
                "run": name,
                "note": note,
                "warmup": args.get("warmup"),
                "max_lr": args.get("max_lr"),
                "final_valid": None if lv is None else round(lv["valid_loss"], 6),
                "early_valid_mean": b["early"],
                "mid_valid_mean": b["mid"],
                "late_valid_mean": b["late"],
                "early_train_mean": round(sum(trains) / len(trains), 6) if trains else None,
                "delta_vs_ba": None
                if lv is None
                else round(lv["valid_loss"] - 1.4818434774875642, 6),
            }
        )
    # fix delta vs actual ba row
    ba_v = next(r["final_valid"] for r in rows_out if r["run"] == "tinystories_ba")
    for r in rows_out:
        r["delta_vs_ba"] = None if r["final_valid"] is None else round(r["final_valid"] - ba_v, 6)
    write_csv(
        out / "q1_warmup.csv",
        [
            "run",
            "note",
            "warmup",
            "max_lr",
            "final_valid",
            "early_valid_mean",
            "mid_valid_mean",
            "late_valid_mean",
            "early_train_mean",
            "delta_vs_ba",
        ],
        rows_out,
    )
    print("Q1 ->", out / "q1_warmup.csv")
    for r in rows_out:
        print(
            f"  {r['run']}: warm={r['warmup']} final={r['final_valid']} "
            f"early_v={r['early_valid_mean']} late_v={r['late_valid_mean']} Δ={r['delta_vs_ba']}"
        )


def q2_lr_batch(out: Path) -> None:
    grid = ["tinystories_ba", "lr6", "bs64_20k", "bs64_lr6e4_20k"]
    summary = []
    series = []
    for name in grid:
        d = RUNS / name
        args = meta_args(d)
        bs = int(args.get("batch_size", 32))
        ctx = int(args.get("context_length", 256))
        lr = args.get("max_lr")
        rows = load_rows(d / "log.jsonl")
        lv = last_valid(rows)
        summary.append(
            {
                "run": name,
                "batch_size": bs,
                "max_lr": lr,
                "tokens_per_step": bs * ctx,
                "final_step": None if lv is None else lv["step"],
                "final_tokens": None if lv is None else tokens(lv["step"], bs, ctx),
                "final_valid": None if lv is None else round(lv["valid_loss"], 6),
            }
        )
        for r in rows:
            if "valid_loss" not in r:
                continue
            series.append(
                {
                    "run": name,
                    "batch_size": bs,
                    "max_lr": lr,
                    "step": r["step"],
                    "tokens_seen": r.get("tokens_seen") or tokens(r["step"], bs, ctx),
                    "valid_loss": round(r["valid_loss"], 6),
                }
            )

    ba_tok = next(s["final_tokens"] for s in summary if s["run"] == "tinystories_ba")
    matched = []
    for name in grid:
        cand = [r for r in series if r["run"] == name]
        if not cand or ba_tok is None:
            continue
        # prefer exact or nearest <= target if overshoots a lot
        best = min(cand, key=lambda r: abs(r["tokens_seen"] - ba_tok))
        matched.append(
            {
                "run": name,
                "batch_size": best["batch_size"],
                "max_lr": best["max_lr"],
                "target_tokens": ba_tok,
                "matched_tokens": best["tokens_seen"],
                "matched_step": best["step"],
                "valid_loss": best["valid_loss"],
            }
        )

    write_csv(
        out / "q2_lr_batch_grid.csv",
        [
            "run",
            "batch_size",
            "max_lr",
            "tokens_per_step",
            "final_step",
            "final_tokens",
            "final_valid",
        ],
        summary,
    )
    write_csv(
        out / "q2_lr_batch_token_matched.csv",
        ["run", "batch_size", "max_lr", "target_tokens", "matched_tokens", "matched_step", "valid_loss"],
        matched,
    )
    print("Q2 grid:")
    for r in summary:
        print(
            f"  bs={r['batch_size']} lr={r['max_lr']}: valid={r['final_valid']} "
            f"tok={r['final_tokens']}"
        )
    print("Q2 token-matched @ ba tokens:")
    for r in matched:
        print(f"  {r['run']}: step={r['matched_step']} valid={r['valid_loss']}")


def q3_compound(out: Path) -> None:
    # stepwise references
    specs = [
        ("tinystories_ba", "baseline"),
        ("L6_20k", "depth+2"),
        ("d768_20k", "width→768 (d_ff=1344 confound)"),
        ("bs64_20k", "bs×2"),
        ("lr6", "lr×2"),
        ("bs64_lr6e4_20k", "bs×2+lr×2"),
        ("L6_d768_bs64_lr6e4_20k", "compound all"),
    ]
    ba_v = None
    rows = []
    for name, note in specs:
        d = RUNS / name
        args = meta_args(d)
        lv = last_valid(load_rows(d / "log.jsonl"))
        v = None if lv is None else lv["valid_loss"]
        if name == "tinystories_ba":
            ba_v = v
        rows.append(
            {
                "run": name,
                "note": note,
                "num_layers": args.get("num_layers"),
                "d_model": args.get("d_model"),
                "d_ff": args.get("d_ff"),
                "batch_size": args.get("batch_size"),
                "max_lr": args.get("max_lr"),
                "final_valid": None if v is None else round(v, 6),
                "delta_vs_ba": None if (v is None or ba_v is None) else round(v - ba_v, 6),
            }
        )
    # fill deltas after ba known
    ba_v = next(r["final_valid"] for r in rows if r["run"] == "tinystories_ba")
    for r in rows:
        r["delta_vs_ba"] = None if r["final_valid"] is None else round(r["final_valid"] - ba_v, 6)

    write_csv(
        out / "q3_compound_attribution.csv",
        [
            "run",
            "note",
            "num_layers",
            "d_model",
            "d_ff",
            "batch_size",
            "max_lr",
            "final_valid",
            "delta_vs_ba",
        ],
        rows,
    )
    print("Q3 attribution:")
    for r in rows:
        print(f"  {r['run']}: {r['final_valid']} Δ={r['delta_vs_ba']} ({r['note']})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "lm4",
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    q1_warmup(args.out)
    q2_lr_batch(args.out)
    q3_compound(args.out)


if __name__ == "__main__":
    main()
