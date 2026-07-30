"""Q2 frozen eval: parse OWT BPE train log → s/merge vs affected buckets."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# tqdm lines look like:
# BPE merge: ... | 16/31743 [04:08<..., 12.60s/merge, affected=297332, freq=..., pair=...]
STEP_RE = re.compile(
    r"BPE merge:.*?\|\s*(\d+)/(\d+)\s*\["
    r".*?,\s*([0-9.]+)(s|merge)/s(?:/merge)?"
    r".*?affected=([0-9]+)"
    r".*?freq=([0-9.eE+-]+)"
)


def parse_rate(num: float, unit: str) -> float:
    """Return seconds per merge."""
    if unit == "s":
        return num  # N s/merge
    # N merge/s
    return 1.0 / num if num > 0 else float("nan")


def iter_rows(log_path: Path):
    # tqdm uses \r; flatten
    text = log_path.read_bytes().decode("utf-8", errors="replace").replace("\r", "\n")
    seen = {}
    for line in text.splitlines():
        if "BPE merge:" not in line or "affected=" not in line:
            continue
        # Prefer the form with explicit s/merge or merge/s near the end
        m = re.search(
            r"\|\s*(\d+)/(\d+)\s*\[[^\]]*?,\s*([0-9.]+)(s/merge|merge/s)",
            line,
        )
        am = re.search(r"affected=(\d+)", line)
        fm = re.search(r"freq=([0-9.eE+-]+)", line)
        if not (m and am and fm):
            continue
        step = int(m.group(1))
        total = int(m.group(2))
        rate = float(m.group(3))
        unit = m.group(4)
        if unit == "s/merge":
            spm = rate
        else:
            spm = 1.0 / rate if rate > 0 else float("nan")
        seen[step] = {
            "step": step,
            "total": total,
            "seconds_per_merge": spm,
            "affected": int(am.group(1)),
            "freq": float(fm.group(1)),
        }
    for step in sorted(seen):
        yield seen[step]


def bucketize(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    n = len(rows)
    cuts = [
        ("early", rows[: max(1, n // 10)]),
        ("mid", rows[n // 2 : n // 2 + max(1, n // 10)]),
        ("late", rows[-max(1, n // 10) :]),
    ]
    out = []
    for name, chunk in cuts:
        spm = [r["seconds_per_merge"] for r in chunk]
        aff = [r["affected"] for r in chunk]
        fr = [r["freq"] for r in chunk]
        out.append(
            {
                "bucket": name,
                "n_steps": len(chunk),
                "mean_s_per_merge": round(sum(spm) / len(spm), 4),
                "mean_affected": round(sum(aff) / len(aff), 1),
                "mean_freq": f"{sum(fr) / len(fr):.3e}",
                "step_lo": chunk[0]["step"],
                "step_hi": chunk[-1]["step"],
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--detail-out", type=Path, default=None)
    args = p.parse_args()

    rows = list(iter_rows(args.log))
    if len(rows) < 10:
        raise SystemExit(f"too few parsed merge rows: {len(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = bucketize(rows)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "bucket",
                "n_steps",
                "mean_s_per_merge",
                "mean_affected",
                "mean_freq",
                "step_lo",
                "step_hi",
            ],
        )
        w.writeheader()
        w.writerows(summary)

    detail = args.detail_out or args.out.with_name(args.out.stem + "_steps.csv")
    with open(detail, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["step", "total", "seconds_per_merge", "affected", "freq"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"parsed {len(rows)} steps")
    for r in summary:
        print(
            f"{r['bucket']}: mean_s/merge={r['mean_s_per_merge']} "
            f"mean_affected={r['mean_affected']} steps={r['step_lo']}-{r['step_hi']}"
        )
    print(f"wrote {args.out}")
    print(f"wrote {detail}")


if __name__ == "__main__":
    main()
