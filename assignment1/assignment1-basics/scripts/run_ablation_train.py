#!/usr/bin/env python3
"""
CS336 Section 7.3 消融训练入口（独立脚本，不改 run_train.py / linear.py baseline）。

日志：log.jsonl 含 step、wall_s、tokens_seen；SwanLab 记 train/*、valid/*、time/wall_s、data/tokens_seen。

  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/run_ablation_train.py \\
    --variant baseline --steps 8000 \\
    --out-dir /data1/wcz/projects/Myllm-runs/experiments/ablation73/baseline
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_ENV_FILE = _ROOT / ".env"


def _load_dotenv() -> None:
    if not _ENV_FILE.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_ENV_FILE, override=False)
    for key in ("SWANLAB_EXP_NAME", "SWANLAB_PROJ_NAME", "SWANLAB_WORKSPACE", "SWANLAB_DESCRIPTION"):
        if os.environ.get(key, "").strip() == "":
            os.environ.pop(key, None)


_load_dotenv()

import numpy as np
import torch
from tqdm import tqdm

from ablation_model import AblationFlags, AblationTransformerLM, default_d_ff
from cs336_basics.linear import AdamW, cross_entropy, get_batch, get_lr_cosine_schedule, gradient_clipping, save_checkpoint
from run_train import DEFAULT_TRAIN, DEFAULT_VALID, _swanlab_enabled

DEFAULT_OUT = Path("/data1/wcz/projects/Myllm-runs/experiments/ablation73")
VARIANTS = ("baseline", "no_rmsnorm", "post_norm", "no_rope", "silu_ffn")


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.no_grad()
def eval_valid(model, data, batch_size, context_length, device, num_batches) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(data, batch_size, context_length, device)
        x, y = x.long(), y.long()
        losses.append(compute_loss(model(x), y).item())
    model.train()
    return float(sum(losses) / len(losses))


def _resolve_d_ff(args: argparse.Namespace, flags: AblationFlags) -> int:
    if args.d_ff is not None:
        return args.d_ff
    if flags.ffn_type == "silu":
        return 4 * args.d_model
    return default_d_ff(args.d_model, "swiglu")


def _swanlab_init_ablation(args, n_params: int, device: str, flags: AblationFlags):
    exp_name = (os.environ.get("SWANLAB_EXP_NAME") or "").strip() or args.out_dir.name
    os.environ["SWANLAB_EXP_NAME"] = exp_name
    import swanlab

    tags = ["cs336", "ablation-7.3", flags.variant]
    if os.environ.get("SWANLAB_TAGS"):
        tags.extend(t.strip() for t in os.environ["SWANLAB_TAGS"].split(",") if t.strip())

    config = {
        "ablation": {
            "variant": flags.variant,
            "use_rmsnorm": flags.use_rmsnorm,
            "norm_style": flags.norm_style,
            "use_rope": flags.use_rope,
            "ffn_type": flags.ffn_type,
            "compare_group": args.compare_group,
        },
        "model": {
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "context_length": args.context_length,
            "params_M": round(n_params / 1e6, 4),
        },
        "optim": {"max_lr": args.max_lr, "min_lr": args.min_lr, "warmup": args.warmup},
        "train": {"batch_size": args.batch_size, "steps": args.steps},
    }
    kwargs: dict = {
        "config": config,
        "logdir": str(args.out_dir / "swanlab"),
        "experiment_name": exp_name,
        "group": args.compare_group,
        "job_type": "ablation",
        "tags": tags,
    }
    if os.environ.get("SWANLAB_DESCRIPTION"):
        kwargs["description"] = os.environ["SWANLAB_DESCRIPTION"]
    if os.environ.get("SWANLAB_PROJ_NAME"):
        kwargs["project"] = os.environ["SWANLAB_PROJ_NAME"]
    if os.environ.get("SWANLAB_WORKSPACE"):
        kwargs["workspace"] = os.environ["SWANLAB_WORKSPACE"]
    if os.environ.get("SWANLAB_MODE"):
        kwargs["mode"] = os.environ["SWANLAB_MODE"]

    run = swanlab.init(**kwargs)
    print(f"SwanLab group={args.compare_group} exp={exp_name}")
    return run


def _swanlab_log_ablation(row: dict, step: int) -> None:
    import swanlab

    payload = {
        "train/loss": row["train_loss"],
        "train/loss_avg": row["train_loss_avg"],
        "train/lr": row["lr"],
        "train/ppl": math.exp(min(row["train_loss"], 20)),
        "time/wall_s": row["wall_s"],
        "data/tokens_seen": row["tokens_seen"],
    }
    if "valid_loss" in row:
        payload["valid/loss"] = row["valid_loss"]
        payload["valid/ppl"] = row["valid_ppl"]
    swanlab.log(payload, step=step)


def main() -> None:
    p = argparse.ArgumentParser(description="CS336 §7.3 ablation training")
    p.add_argument("--variant", choices=VARIANTS, default="baseline")
    p.add_argument("--compare-group", default="ablation-7.3")
    p.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--valid-data", type=Path, default=DEFAULT_VALID)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--max-lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--cosine-steps", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--swanlab", action="store_true")
    p.add_argument("--no-swanlab", action="store_true")
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT / args.variant
    if args.cosine_steps is None:
        args.cosine_steps = args.steps

    flags = AblationFlags.from_variant(args.variant)
    args.d_ff = _resolve_d_ff(args, flags)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "section": "7.3",
        "variant": args.variant,
        "flags": {
            "use_rmsnorm": flags.use_rmsnorm,
            "norm_style": flags.norm_style,
            "use_rope": flags.use_rope,
            "ffn_type": flags.ffn_type,
        },
        "hyperparams": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    with open(args.out_dir / "ablation_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r")
    tokens_per_step = args.batch_size * args.context_length

    print(f"variant={args.variant}  d_ff={args.d_ff}  lr={args.max_lr}  steps={args.steps}")

    model = AblationTransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        flags,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params/1e6:.2f}M")

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    use_swan = _swanlab_enabled(args)
    swan_run = None
    if use_swan:
        try:
            swan_run = _swanlab_init_ablation(args, n_params, device, flags)
        except Exception as e:
            print(f"SwanLab 失败，继续训练: {e}")
            use_swan = False

    log_path = args.out_dir / "log.jsonl"
    t0 = time.time()
    model.train()
    pbar = tqdm(range(args.steps), desc=args.variant, unit="step")

    for step in pbar:
        lr = get_lr_cosine_schedule(step, args.max_lr, args.min_lr, args.warmup, args.cosine_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        x, y = x.long(), y.long()

        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model(x), y)
        loss.backward()
        loss_v = loss.item()
        if not math.isfinite(loss_v):
            pbar.write(f"step {step+1}: non-finite loss — 停止")
            if args.grad_clip > 0:
                gradient_clipping(model.parameters(), args.grad_clip)
            optimizer.step()
            break
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        pbar.set_postfix(loss=f"{loss_v:.4f}", lr=f"{lr:.2e}")

        if (step + 1) % args.log_every == 0 or step == 0:
            row = {
                "step": step + 1,
                "train_loss": loss_v,
                "train_loss_avg": loss_v,
                "lr": lr,
                "wall_s": round(time.time() - t0, 3),
                "tokens_seen": (step + 1) * tokens_per_step,
                "variant": args.variant,
            }
            if (step + 1) % args.eval_every == 0:
                vloss = eval_valid(
                    model, valid_data, args.batch_size, args.context_length, device, args.eval_batches
                )
                row["valid_loss"] = vloss
                row["valid_ppl"] = math.exp(min(vloss, 20))
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if use_swan:
                _swanlab_log_ablation(row, step=step + 1)

        if (step + 1) % args.ckpt_every == 0:
            save_checkpoint(model, optimizer, step + 1, args.out_dir / f"ckpt_step{step+1}.pt")

    with open(args.out_dir / "run_meta.json", "w") as f:
        json.dump({"variant": args.variant, "params": n_params, "seconds": time.time() - t0}, f, indent=2)

    if use_swan and swan_run is not None:
        import swanlab

        swanlab.finish()
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
