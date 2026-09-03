"""FlashAttention backward 三方对比: naive vs flash-存P(不重计算) vs flash-重计算.

跑法（你的终端）:
    cd /root/Myllm/assignment2_system/assignment2-systems
    .venv/bin/python /root/Myllm/exercises/bench_flash_bwd.py [--causal] [--sizes 1024,4096,8192]

"重计算 (recompute)" 是什么:
- 存P版: forward 把 softmax 概率矩阵 P (N×N) 物化保存, backward 直接用它。
  backward 省一次 QK^T 重算 (少 2·N²·D FLOPs), 但 P 活在显存里 → O(N²)。
- 重计算版 (我们实现的 flash): forward 只存 lse (N 个数), backward 用
  P = exp(S - lse) 逐块重建 P。代价 = 每块多算一次 QK^T,
  总 FLOPs 14·N²·D vs 存P版 12·N²·D (+16.7%); 收益 = 显存 O(N)。
- naive: einsum+softmax 全自动求导, S 和 P 都物化 (2×N²), backward 12·N²·D。

FLOPs 记账 (每个 matmul 2·N²·D):
    forward: S=QK^T (2) + O=PV (2)            = 4·N²·D   (三种相同)
    backward(存P/naive): dV + dP + dQ + dK     = 8·N²·D
    backward(重计算):     上式 + 重算 QK^T (2) = 10·N²·D
"""
import argparse

import torch

from cs336_systems.attention import (
    FlashAttentionPytorchAutogradFunction,
    FlashAttentionTritonAutogradFunction,
)


def naive_attention(q, k, v, is_causal):
    scale = q.shape[-1] ** -0.5
    s = torch.einsum("...qd,...kd->...qk", q, k) * scale
    if is_causal:
        n_q, n_k = q.shape[-2], k.shape[-2]
        mask = torch.arange(n_q, device=s.device)[None, :, None] >= torch.arange(
            n_k, device=s.device)[None, None, :]
        s = torch.where(mask, s, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("...qk,...kd->...qd", p, v)


class StoredPAttention(torch.autograd.Function):
    """不重计算的 flash 变体: forward 物化 P 并保存 (O(N²) 显存), backward 直接用。

    与重计算版唯一的算法差异: backward 不再算 S=QK^T。
    注意 backward 里 p/dp/ds 三个 N² 矩阵同时存活 → 峰值 ≈ 3×N²,
    这正是"不重计算省 FLOPs 但爆显存"的实证。
    """

    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        scale = q.shape[-1] ** -0.5
        s = torch.matmul(q, k.transpose(-2, -1)) * scale
        if is_causal:
            n_q, n_k = q.shape[-2], k.shape[-2]
            mask = torch.arange(n_q, device=s.device)[None, :, None] >= torch.arange(
                n_k, device=s.device)[None, None, :]
            s = torch.where(mask, s, float("-inf"))
        p = torch.softmax(s, dim=-1)
        o = torch.matmul(p, v)
        ctx.save_for_backward(q, k, v, p, o)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, p, o = ctx.saved_tensors
        scale = q.shape[-1] ** -0.5
        delta = (do * o).sum(-1, keepdim=True)
        dp = torch.matmul(do, v.transpose(-2, -1))
        ds = p * (dp - delta)
        dq = torch.matmul(ds, k) * scale
        dk = torch.matmul(ds.transpose(-2, -1), q) * scale
        dv = torch.matmul(p.transpose(-2, -1), do)
        return dq, dk, dv, None


# (名字, 前向调用, FLOPs 系数 a: 总 FLOPs = a·N²·D)
IMPLS = [
    ("naive", naive_attention, 12),
    ("stored-P", StoredPAttention.apply, 12),
    ("flash-pytorch(recompute)", FlashAttentionPytorchAutogradFunction.apply, 14),
    ("flash-triton(recompute)", FlashAttentionTritonAutogradFunction.apply, 14),
]


def _grads(impl, B, N, D, is_causal):
    torch.manual_seed(0)
    q = torch.randn(B, N, D, device="cuda", requires_grad=True)
    k = torch.randn(B, N, D, device="cuda", requires_grad=True)
    v = torch.randn(B, N, D, device="cuda", requires_grad=True)
    out = impl(q, k, v, is_causal)
    do = torch.randn_like(out)
    return torch.autograd.grad(out, (q, k, v), do)


def check_correctness(is_causal):
    """每种实现与 naive 比梯度 (只比最小的 N, 快速排错用)。"""
    B, N, D = 1, 256, 64
    ref = _grads(naive_attention, B, N, D, is_causal)
    print(f"correctness (B={B},N={N},D={D},causal={is_causal}), max|grad diff| vs naive:")
    for name, impl, _ in IMPLS[1:]:
        g = _grads(impl, B, N, D, is_causal)
        err = max((a - b).abs().max().item() for a, b in zip(ref, g))
        print(f"  {name:28s} {err:.2e}")


def bench(impl, flops_coeff, B, N, D, is_causal, iters):
    torch.manual_seed(0)
    q = torch.randn(B, N, D, device="cuda", requires_grad=True)
    k = torch.randn(B, N, D, device="cuda", requires_grad=True)
    v = torch.randn(B, N, D, device="cuda", requires_grad=True)

    def step():
        out = impl(q, k, v, is_causal)
        do = torch.randn_like(out)
        torch.autograd.grad(out, (q, k, v), do)

    for _ in range(2):  # warmup (含 triton JIT)
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        step()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    tflops = flops_coeff * N * N * D / (ms * 1e-3) / 1e12
    return ms, peak_mb, tflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1024,4096,8192")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--causal", action="store_true")
    args = ap.parse_args()

    check_correctness(args.causal)
    print()

    sizes = [int(s) for s in args.sizes.split(",")]
    print(f"fwd+bwd, B={args.batch}, D={args.d}, causal={args.causal}, iters={args.iters}")
    header = f"{'impl':28s} {'N':>6s} {'ms/step':>9s} {'peak MB':>9s} {'TFLOP/s':>8s} {'mem/naive':>9s}"
    print(header)
    for N in sizes:
        base = None
        for name, impl, coeff in IMPLS:
            try:
                ms, peak, tf = bench(impl, coeff, args.batch, N, args.d, args.causal, args.iters)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{name:28s} {N:>6d} {'OOM':>9s}")
                continue
            if name == "naive":
                base = peak
            ratio = f"{peak / base:8.1%}" if base else "     n/a"
            print(f"{name:28s} {N:>6d} {ms:>9.1f} {peak:>9.1f} {tf:>8.2f} {ratio:>9s}")
        print()


if __name__ == "__main__":
    main()
