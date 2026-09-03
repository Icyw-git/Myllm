"""官方网格 attention benchmark: 覆盖 handout 4.1.1 / 4.2(a) / 4.2.2.

跑法（你的终端, 先小后大）:
    cd /root/Myllm/assignment2_system/assignment2-systems
    # 4.1.1: naive vs flash 官方网格 (B=8, fp32, 100 次计时 + OOM 记账)
    .venv/bin/python /root/Myllm/exercises/sweep_attention.py
    # 4.2(a): compile 对照 (同样网格, 加 compile 变体)
    .venv/bin/python /root/Myllm/exercises/sweep_attention.py --impls naive,naive-compile,flash-pytorch,flash-pytorch-compile
    # 4.2.2: flash 官方大表 (B=1, causal, bf16, 长序列)
    .venv/bin/python /root/Myllm/exercises/sweep_attention.py --b 1 --causal --dtype bf16 --iters 20 --n-list 128,256,512,1024,2048,4096,8192,16384,32768,65536 --impls naive,flash-pytorch,flash-triton
"""
import argparse

import torch

import cs336_basics.model as basics_model
from cs336_systems.attention import (
    FlashAttentionPytorchAutogradFunction as FlashPy,
    FlashAttentionTritonAutogradFunction as FlashTri,
)


def _flat3(t):
    *lead, s, d = t.shape
    return t.reshape(-1, s, d).contiguous(), lead, s, d


def make_impl(name):
    base = name.replace('-compile', '')
    if base == 'naive':
        def fn(q, k, v, is_causal):
            mask = None
            if is_causal:
                mask = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device).tril()
            return basics_model.scaled_dot_product_attention(q, k, v, mask)
    elif base in ('flash-pytorch', 'flash-triton'):
        cls = FlashPy if base == 'flash-pytorch' else FlashTri

        def fn(q, k, v, is_causal):
            qf, lead, s, d = _flat3(q)
            kf, _, _, _ = _flat3(k)
            vf, _, _, _ = _flat3(v)
            return cls.apply(qf, kf, vf, is_causal).reshape(*lead, s, d)
    else:
        raise ValueError(name)
    return torch.compile(fn) if name.endswith('-compile') else fn


def bench_point(fn, B, N, d, dtype, is_causal, iters, warmup, device):
    """返回 (fwd_ms, fwd峰值MB[op净增], bwd_ms)。"""
    torch.manual_seed(0)
    q = torch.randn(B, N, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, N, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, N, d, device=device, dtype=dtype, requires_grad=True)
    g = torch.ones_like(q)

    def fwd():
        return fn(q, k, v, is_causal)

    for _ in range(warmup):
        out = fwd()
        del out
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        out = fwd()
        del out
    end.record()
    torch.cuda.synchronize()
    fwd_ms = start.elapsed_time(end) / iters

    # fwd 显存: 单次带梯度 forward 后 allocator 峰值净增 = op 的激活占用
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    out = fwd()
    torch.cuda.synchronize()
    mem_mb = (torch.cuda.max_memory_allocated() - base_mem) / 1024**2
    del out

    # bwd 计时 (每次 fresh 图 + backward)
    for _ in range(warmup):
        out = fwd()
        out.backward(g)
    start.record()
    for _ in range(iters):
        out = fwd()
        out.backward(g)
    end.record()
    torch.cuda.synchronize()
    bwd_ms = start.elapsed_time(end) / iters

    del q, k, v, g
    return fwd_ms, mem_mb, bwd_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--impls', default='naive,flash-pytorch')
    ap.add_argument('--b', type=int, default=8)
    ap.add_argument('--d-list', default='16,32,64,128')
    ap.add_argument('--n-list', default='256,512,1024,2048,4096,8192,16384')
    ap.add_argument('--dtype', choices=['fp32', 'bf16'], default='fp32')
    ap.add_argument('--causal', action='store_true')
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=3)
    args = ap.parse_args()

    device = 'cuda'
    assert torch.cuda.is_available()
    dtype = torch.float32 if args.dtype == 'fp32' else torch.bfloat16
    n_list = [int(x) for x in args.n_list.split(',')]
    d_list = [int(x) for x in args.d_list.split(',')]
    print(f'B={args.b} dtype={args.dtype} causal={args.causal} iters={args.iters} (+{args.warmup} warmup)')
    print('impl,d,N,fwd_ms,mem_MB,bwd_ms')

    for name in args.impls.split(','):
        fn = make_impl(name)
        for d in d_list:
            for N in n_list:
                try:
                    fwd_ms, mem_mb, bwd_ms = bench_point(
                        fn, args.b, N, d, dtype, args.causal, args.iters, args.warmup, device)
                    print(f'{name},{d},{N},{fwd_ms:.3f},{mem_mb:.1f},{bwd_ms:.3f}')
                except RuntimeError as e:
                    torch.cuda.empty_cache()
                    if 'out of memory' in str(e).lower():
                        print(f'{name},{d},{N},OOM,,')
                    else:
                        raise
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
