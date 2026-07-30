"""lm3 Q1–Q4: LR / batch-token fairness / negative results / overfit from existing runs."""
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
    if not p.is_file():
        return {}
    return json.loads(p.read_text()).get("args", {})


def tokens_at(step: int, batch: int, ctx: int) -> int:
    return step * batch * ctx


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def q1_lr(out: Path) -> None:
    specs = [
        ("lr1", 1e-4),
        ("tinystories_ba", 3e-4),
        ("lr6", 6e-4),
        ("warmup500_20k", 3e-4),
    ]
    rows = []
    for name, lr in specs:
        d = RUNS / name
        args = meta_args(d)
        log = d / "log.jsonl"
        lv = last_valid(load_rows(log))
        rows.append(
            {
                "run": name,
                "max_lr": args.get("max_lr", lr),
                "min_lr": args.get("min_lr"),
                "warmup": args.get("warmup"),
                "steps": args.get("steps"),
                "valid_loss": None if lv is None else round(lv["valid_loss"], 6),
                "valid_step": None if lv is None else lv["step"],
            }
        )
    write_csv(
        out / "q1_lr_sweep.csv",
        ["run", "max_lr", "min_lr", "warmup", "steps", "valid_loss", "valid_step"],
        rows,
    )
    print("Q1 ->", out / "q1_lr_sweep.csv")
    for r in rows:
        print(f"  {r['run']}: max_lr={r['max_lr']} valid={r['valid_loss']}")


def q2_batch(out: Path) -> None:
    specs = ["bs16_20k", "tinystories_ba", "bs64_20k"]
    # per-step rows + token-aligned narrative at shared token budgets
    series_rows = []
    summary = []
    for name in specs:
        d = RUNS / name
        args = meta_args(d)
        bs = int(args.get("batch_size", 32))
        ctx = int(args.get("context_length", 256))
        rows = load_rows(d / "log.jsonl")
        for r in rows:
            if "valid_loss" not in r:
                continue
            tok = r.get("tokens_seen")
            if tok is None:
                tok = tokens_at(r["step"], bs, ctx)
            series_rows.append(
                {
                    "run": name,
                    "batch_size": bs,
                    "step": r["step"],
                    "tokens_seen": tok,
                    "valid_loss": round(r["valid_loss"], 6),
                    "wall_s": r.get("wall_s"),
                }
            )
        lv = last_valid(rows)
        summary.append(
            {
                "run": name,
                "batch_size": bs,
                "tokens_per_step": bs * ctx,
                "final_step": None if lv is None else lv["step"],
                "final_tokens": None
                if lv is None
                else (lv.get("tokens_seen") or tokens_at(lv["step"], bs, ctx)),
                "final_valid": None if lv is None else round(lv["valid_loss"], 6),
            }
        )

    write_csv(
        out / "q2_batch_tokens_series.csv",
        ["run", "batch_size", "step", "tokens_seen", "valid_loss", "wall_s"],
        series_rows,
    )
    write_csv(
        out / "q2_batch_tokens_summary.csv",
        ["run", "batch_size", "tokens_per_step", "final_step", "final_tokens", "final_valid"],
        summary,
    )

    # nearest-match at ba final tokens (same token budget)
    ba = next(s for s in summary if s["run"] == "tinystories_ba")
    target = ba["final_tokens"]
    fair = []
    for name in specs:
        cand = [r for r in series_rows if r["run"] == name]
        if not cand or target is None:
            continue
        best = min(cand, key=lambda r: abs(r["tokens_seen"] - target))
        fair.append(
            {
                "run": name,
                "batch_size": best["batch_size"],
                "target_tokens": target,
                "matched_tokens": best["tokens_seen"],
                "matched_step": best["step"],
                "valid_loss": best["valid_loss"],
            }
        )
    write_csv(
        out / "q2_batch_token_matched.csv",
        ["run", "batch_size", "target_tokens", "matched_tokens", "matched_step", "valid_loss"],
        fair,
    )
    print("Q2 ->", out / "q2_batch_tokens_summary.csv")
    for r in summary:
        print(
            f"  {r['run']}: bs={r['batch_size']} tok/step={r['tokens_per_step']} "
            f"final_tok={r['final_tokens']} valid={r['final_valid']}"
        )
    print("Q2 token-matched @ ba tokens:")
    for r in fair:
        print(f"  {r['run']}: step={r['matched_step']} tok={r['matched_tokens']} valid={r['valid_loss']}")


def q3_negative(out: Path) -> None:
    ba = last_valid(load_rows(RUNS / "tinystories_ba" / "log.jsonl"))
    ba_v = None if ba is None else ba["valid_loss"]
    specs = [
        ("tinystories_ba", "baseline"),
        ("wd0_20k", "weight_decay=0"),
        ("rope500k_20k", "rope_theta=5e5"),
    ]
    rows = []
    for name, note in specs:
        d = RUNS / name
        args = meta_args(d)
        lv = last_valid(load_rows(d / "log.jsonl"))
        v = None if lv is None else lv["valid_loss"]
        delta = None if (v is None or ba_v is None) else round(v - ba_v, 6)
        rows.append(
            {
                "run": name,
                "note": note,
                "weight_decay": args.get("weight_decay"),
                "rope_theta": args.get("rope_theta"),
                "valid_loss": None if v is None else round(v, 6),
                "delta_vs_ba": delta,
            }
        )
    write_csv(
        out / "q3_negative_results.csv",
        ["run", "note", "weight_decay", "rope_theta", "valid_loss", "delta_vs_ba"],
        rows,
    )
    print("Q3 ->", out / "q3_negative_results.csv")
    for r in rows:
        print(f"  {r['run']}: valid={r['valid_loss']} Δ={r['delta_vs_ba']}")


def q4_overfit(out: Path) -> None:
    rows_out = []
    summary = []
    for name in ["overfit_002", "overfit_smoke", "overfit_001"]:
        d = RUNS / name
        log = d / "log_overfit.jsonl"
        if not log.is_file():
            log = d / "log.jsonl"
        rows = load_rows(log)
        if not rows:
            summary.append(
                {
                    "run": name,
                    "n_logs": 0,
                    "first_train": None,
                    "last_train": None,
                    "min_train": None,
                    "ok_near_zero": False,
                    "note": "missing_log",
                }
            )
            continue
        trains = [r["train_loss"] for r in rows]
        for r in rows:
            rows_out.append(
                {
                    "run": name,
                    "step": r["step"],
                    "train_loss": r["train_loss"],
                    "lr": r.get("lr"),
                    "wall_s": r.get("wall_s"),
                }
            )
        last = trains[-1]
        summary.append(
            {
                "run": name,
                "n_logs": len(rows),
                "first_train": round(trains[0], 6),
                "last_train": round(last, 6),
                "min_train": round(min(trains), 6),
                "ok_near_zero": last < 0.05,
                "note": "ok" if last < 0.05 else "not_near_zero",
            }
        )
    write_csv(
        out / "q4_overfit_curve.csv",
        ["run", "step", "train_loss", "lr", "wall_s"],
        rows_out,
    )
    write_csv(
        out / "q4_overfit_summary.csv",
        ["run", "n_logs", "first_train", "last_train", "min_train", "ok_near_zero", "note"],
        summary,
    )
    print("Q4 ->", out / "q4_overfit_summary.csv")
    for r in summary:
        print(
            f"  {r['run']}: first={r['first_train']} last={r['last_train']} "
            f"ok={r['ok_near_zero']} ({r['note']})"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "lm3",
    )
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    q1_lr(args.out)
    q2_batch(args.out)
    q3_negative(args.out)
    q4_overfit(args.out)


if __name__ == "__main__":
    main()
