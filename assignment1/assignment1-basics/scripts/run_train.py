"""
简单训练脚手架：串起 linear.py 里已有组件 + tqdm 进度。
不改 cs336_basics/linear.py。

用法：
  # 单 batch 过拟合 sanity（推荐先跑这个）
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_train.py --overfit --steps 500

  # 正常短训（.env 已配置时默认开 SwanLab）
  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/run_train.py --steps 2000 \\
    --out-dir /data1/wcz/projects/Myllm-runs/experiments/run001
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def _load_dotenv() -> None:
    if not _ENV_FILE.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        print(f"提示: 安装 python-dotenv 以自动加载 {_ENV_FILE}  →  uv pip install python-dotenv")
        return
    load_dotenv(_ENV_FILE, override=False)
    _sanitize_swanlab_env()


def _sanitize_swanlab_env() -> None:
    """空字符串会让 SwanLab 在 import 阶段 pydantic 报错，等同未设置。"""
    for key in (
        "SWANLAB_EXP_NAME",
        "SWANLAB_PROJ_NAME",
        "SWANLAB_WORKSPACE",
        "SWANLAB_DESCRIPTION",
    ):
        if os.environ.get(key, "").strip() == "":
            os.environ.pop(key, None)


_load_dotenv()

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from cs336_basics.linear import (
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
    transformer_lm,
)

DEFAULT_TRAIN = Path("/data1/wcz/datasets/myllm/tokenized/tinystories_train.npy")
DEFAULT_VALID = Path("/data1/wcz/datasets/myllm/tokenized/tinystories_valid.npy")
DEFAULT_OUT = Path("/data1/wcz/projects/Myllm-runs/experiments/tinystories_smoke")


class TransformerLM(nn.Module):
    """把函数式 transformer_lm 包成可优化的 Module（参数名用 _ 代替 .）。"""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        # 逻辑键（给 transformer_lm） -> register_parameter 名（不能含 '.'）
        self._key_to_name: dict[str, str] = {}

        def add(key: str, shape: tuple[int, ...]) -> None:
            name = key.replace(".", "__")
            p = nn.Parameter(torch.empty(*shape))
            nn.init.trunc_normal_(p, mean=0.0, std=0.02, a=-0.04, b=0.04)
            self.register_parameter(name, p)
            self._key_to_name[key] = name

        add("token_embeddings.weight", (vocab_size, d_model))
        for i in range(num_layers):
            p = f"layers.{i}."
            add(p + "attn.q_proj.weight", (d_model, d_model))
            add(p + "attn.k_proj.weight", (d_model, d_model))
            add(p + "attn.v_proj.weight", (d_model, d_model))
            add(p + "attn.output_proj.weight", (d_model, d_model))
            add(p + "ln1.weight", (d_model,))
            nn.init.ones_(getattr(self, self._key_to_name[p + "ln1.weight"]))
            add(p + "ln2.weight", (d_model,))
            nn.init.ones_(getattr(self, self._key_to_name[p + "ln2.weight"]))
            add(p + "ffn.w1.weight", (d_ff, d_model))
            add(p + "ffn.w2.weight", (d_model, d_ff))
            add(p + "ffn.w3.weight", (d_ff, d_model))
        add("ln_final.weight", (d_model,))
        nn.init.ones_(getattr(self, self._key_to_name["ln_final.weight"]))
        add("lm_head.weight", (vocab_size, d_model))

    def weight_dict(self) -> dict[str, torch.Tensor]:
        return {k: getattr(self, n) for k, n in self._key_to_name.items()}

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        return transformer_lm(
            self.vocab_size,
            self.context_length,
            self.d_model,
            self.num_layers,
            self.num_heads,
            self.d_ff,
            self.rope_theta,
            self.weight_dict(),
            in_indices,
        )


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # 你的 cross_entropy 是 2D：(N, V) / (N,)
    return cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.no_grad()
def eval_valid(
    model: TransformerLM,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(data, batch_size, context_length, device)
        x, y = x.long(), y.long()
        loss = compute_loss(model(x), y)
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def _swanlab_enabled(args: argparse.Namespace) -> bool:
    if getattr(args, "no_swanlab", False):
        return False
    if getattr(args, "swanlab", False):
        return True
    return bool(os.environ.get("SWANLAB_API_KEY"))


def _swanlab_run_config(args: argparse.Namespace, n_params: int, device: str) -> dict:
    """SwanLab 面板里的超参结构（分组键名）。"""
    return {
        "data": {
            "train": str(args.train_data),
            "valid": str(args.valid_data),
        },
        "model": {
            "vocab_size": args.vocab_size,
            "context_length": args.context_length,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "rope_theta": args.rope_theta,
            "params_M": round(n_params / 1e6, 4),
        },
        "optim": {
            "max_lr": args.max_lr,
            "min_lr": args.min_lr,
            "warmup": args.warmup,
            "cosine_steps": args.cosine_steps,
            "weight_decay": args.weight_decay,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "grad_clip": args.grad_clip,
        },
        "train": {
            "batch_size": args.batch_size,
            "steps": args.steps,
            "eval_every": args.eval_every,
            "eval_batches": args.eval_batches,
            "log_every": args.log_every,
            "ckpt_every": args.ckpt_every,
            "seed": args.seed,
            "overfit": args.overfit,
            "device": device,
        },
        "run": {
            "out_dir": str(args.out_dir),
        },
    }


def _swanlab_experiment_name(args: argparse.Namespace) -> str:
    name = (os.environ.get("SWANLAB_EXP_NAME") or "").strip()
    return name or args.out_dir.name


def _swanlab_init(args: argparse.Namespace, n_params: int, device: str):
    exp_name = _swanlab_experiment_name(args)
    os.environ["SWANLAB_EXP_NAME"] = exp_name
    import swanlab

    tags = ["cs336", "overfit" if args.overfit else "tinystories"]
    if os.environ.get("SWANLAB_TAGS"):
        tags.extend(t.strip() for t in os.environ["SWANLAB_TAGS"].split(",") if t.strip())

    kwargs: dict = {
        "config": _swanlab_run_config(args, n_params, device),
        "logdir": str(args.out_dir / "swanlab"),
        "experiment_name": exp_name,
        "group": "overfit" if args.overfit else "train",
        "job_type": "train",
        "tags": tags,
    }
    if os.environ.get("SWANLAB_DESCRIPTION"):
        kwargs["description"] = os.environ["SWANLAB_DESCRIPTION"]
    # project / workspace / mode 优先读 .env：SWANLAB_PROJ_NAME, SWANLAB_WORKSPACE, SWANLAB_MODE
    if os.environ.get("SWANLAB_PROJ_NAME"):
        kwargs["project"] = os.environ["SWANLAB_PROJ_NAME"]
    if os.environ.get("SWANLAB_WORKSPACE"):
        kwargs["workspace"] = os.environ["SWANLAB_WORKSPACE"]
    mode = os.environ.get("SWANLAB_MODE")
    if mode:
        kwargs["mode"] = mode

    run = swanlab.init(**kwargs)
    print(
        f"SwanLab: project={kwargs.get('project', '(env/default)')} "
        f"exp={kwargs['experiment_name']} logdir={kwargs['logdir']}"
    )
    return run


def _swanlab_log_row(row: dict, step: int) -> None:
    import swanlab

    payload = {
        "train/loss": row["train_loss"],
        "train/loss_avg": row["train_loss_avg"],
        "train/lr": row["lr"],
        "train/ppl": math.exp(min(row["train_loss"], 20)),
        "time/wall_s": row["wall_s"],
    }
    if "valid_loss" in row:
        payload["valid/loss"] = row["valid_loss"]
        payload["valid/ppl"] = row["valid_ppl"]
    swanlab.log(payload, step=step)


def main() -> None:
    p = argparse.ArgumentParser(description="TinyStories training scaffold")
    p.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--valid-data", type=Path, default=DEFAULT_VALID)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--max-lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--cosine-steps", type=int, default=None, help="default = steps")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--overfit",
        action="store_true",
        help="固定同一个 batch 反复训，loss 应很快降到接近 0",
    )
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--swanlab", action="store_true", help="开启 SwanLab（默认：.env 有 API Key 则开）")
    p.add_argument("--no-swanlab", action="store_true", help="关闭 SwanLab")
    args = p.parse_args()
    if args.cosine_steps is None:
        args.cosine_steps = args.steps

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，改用 cpu")
        device = "cpu"

    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r")
    print(f"train tokens={train_data.shape[0]:,}  valid={valid_data.shape[0]:,}")
    print(
        f"model: L={args.num_layers} d={args.d_model} h={args.num_heads} "
        f"ff={args.d_ff} ctx={args.context_length} vocab={args.vocab_size}"
    )
    print(f"device={device}  overfit={args.overfit}  steps={args.steps}")

    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
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

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"resumed from {args.resume} @ step={start_step}")

    # overfit：抽一次 batch，后面一直用它
    fixed_batch = None
    if args.overfit:
        x0, y0 = get_batch(train_data, args.batch_size, args.context_length, device)
        fixed_batch = (x0.long(), y0.long())
        print(f"overfit batch locked: x={tuple(fixed_batch[0].shape)}")

    use_swan = _swanlab_enabled(args)
    swan_run = None
    if use_swan:
        try:
            swan_run = _swanlab_init(args, n_params, device)
        except ImportError:
            print("SwanLab 未安装: uv pip install swanlab  （见 --index-url 清华源）")
            use_swan = False
        except Exception as e:
            print(f"SwanLab 初始化失败，继续训练（无云端日志）: {e}")
            use_swan = False

    log_path = args.out_dir / ("log_overfit.jsonl" if args.overfit else "log.jsonl")
    metrics = []
    t0 = time.time()
    model.train()
    pbar = tqdm(range(start_step, args.steps), desc="train", unit="step")
    running = 0.0
    running_n = 0

    for step in pbar:
        lr = get_lr_cosine_schedule(
            step, args.max_lr, args.min_lr, args.warmup, args.cosine_steps
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        if fixed_batch is not None:
            x, y = fixed_batch
        else:
            x, y = get_batch(train_data, args.batch_size, args.context_length, device)
            x, y = x.long(), y.long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = compute_loss(logits, y)
        loss.backward()
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        loss_v = loss.item()
        running += loss_v
        running_n += 1
        pbar.set_postfix(loss=f"{loss_v:.4f}", lr=f"{lr:.2e}", ppl=f"{math.exp(min(loss_v, 20)):.2f}")

        if (step + 1) % args.log_every == 0 or step == start_step:
            avg = running / max(running_n, 1)
            row = {
                "step": step + 1,
                "train_loss": loss_v,
                "train_loss_avg": avg,
                "lr": lr,
                "wall_s": round(time.time() - t0, 2),
                "overfit": args.overfit,
            }
            running = 0.0
            running_n = 0
            if (not args.overfit) and (step + 1) % args.eval_every == 0:
                vloss = eval_valid(
                    model,
                    valid_data,
                    args.batch_size,
                    args.context_length,
                    device,
                    args.eval_batches,
                )
                row["valid_loss"] = vloss
                row["valid_ppl"] = math.exp(min(vloss, 20))
                pbar.write(f"step {step+1}: train={loss_v:.4f} valid={vloss:.4f} lr={lr:.2e}")
            metrics.append(row)
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if use_swan:
                _swanlab_log_row(row, step=step + 1)

        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.steps:
            ckpt = args.out_dir / f"ckpt_step{step+1}.pt"
            save_checkpoint(model, optimizer, step + 1, ckpt)
            pbar.write(f"saved {ckpt}")

    # overfit 终检
    if args.overfit:
        final = metrics[-1]["train_loss"] if metrics else loss_v
        print(f"\noverfit final loss={final:.4f} (期望明显下降，理想接近 0)")
        if final > 1.0:
            print("警告：loss 仍偏高，检查 mask/RoPE/lr 或加长 --steps")

    meta = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "params": n_params,
        "seconds": round(time.time() - t0, 2),
        "final_train_loss": metrics[-1]["train_loss"] if metrics else None,
    }
    with open(args.out_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if use_swan and swan_run is not None:
        import swanlab

        if metrics:
            last = metrics[-1]
            swanlab.log(
                {
                    "final/train_loss": last["train_loss"],
                    "final/train_ppl": math.exp(min(last["train_loss"], 20)),
                    **(
                        {
                            "final/valid_loss": last["valid_loss"],
                            "final/valid_ppl": last["valid_ppl"],
                        }
                        if "valid_loss" in last
                        else {}
                    ),
                },
                step=metrics[-1]["step"],
            )
        swanlab.finish()

    print(f"done -> {args.out_dir}  log={log_path}")


if __name__ == "__main__":
    main()
