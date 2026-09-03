"""gradient_checkpointing (b): 单层分段 checkpointing 的最优段大小实验.

跑法（你的终端）:
    cd /root/Myllm/assignment2_system/assignment2-systems
    .venv/bin/python /root/Myllm/exercises/bench_checkpointing.py
    .venv/bin/python /root/Myllm/exercises/bench_checkpointing.py --mode fwd   # xl forward-only

理论预测 (单层分段, 每 block residuals ≈ c 单位, 1 单位 = 一个残差流张量):
    峰值激活(k) = N/k × C + k × c × C     (C = 一个残差流张量, N = 层数)
    → k* = sqrt(N/c), 峰值 = 2·sqrt(c·N)·C

3090 适配声明 (writeup 要写):
    xl full training step 固定开销 = 16B/参数 × 3.41B ≈ 54.6GB,
    large 也有 16B × 1.34B ≈ 21.5GB (加最少激活即超 24GB, 实测全 k OOM)。
    checkpointing 只省激活、救不了参数/梯度/优化器状态 →
    full step 用 medium (0.42B, 固定 6.8GB, fp32 全曲线可跑),
    xl 用 forward-only 验证同样的规律。
"""
import argparse

import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

SIZES = {
    'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
    'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
    'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
    'xl': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32},
}
VOCAB = 10000


def forward_with_ckpt(model, token_ids, blocks_per_seg: int | None):
    """按 blocks_per_seg 分段 checkpoint 的 forward; None = 不 checkpoint."""
    x = model.token_embeddings(token_ids)
    layers = model.layers
    if blocks_per_seg is None:
        for layer in layers:
            x = layer(x)
    else:
        for i in range(0, len(layers), blocks_per_seg):
            seg = layers[i:i + blocks_per_seg]

            def seg_run(h, seg=seg):
                for layer in seg:
                    h = layer(h)
                return h

            # use_reentrant=False: 支持闭包/关键字参数, RNG 状态自动保存
            x = checkpoint(seg_run, x, use_reentrant=False)
    x = model.ln_final(x)
    return model.lm_head(x)


def apply_attention_patch(attention: str) -> None:
    """把模型里的 naive attention 换成 flash (复用 benchmark.py 的做法)."""
    import cs336_basics.model as basics_model
    from cs336_systems.attention import FlashAttentionPytorchAutogradFunction

    def _sdpa(Q, K, V, mask=None):
        is_causal = mask is not None
        *lead, seq, d_k = Q.shape
        Qf = Q.reshape(-1, seq, d_k).contiguous()
        Kf = K.reshape(-1, seq, d_k).contiguous()
        Vf = V.reshape(-1, seq, d_k).contiguous()
        out = FlashAttentionPytorchAutogradFunction.apply(Qf, Kf, Vf, is_causal)
        return out.reshape(*lead, seq, d_k)

    basics_model.scaled_dot_product_attention = _sdpa


def run_case(cfg, blocks_per_seg, mode):
    torch.manual_seed(0)
    device = 'cuda'
    model = BasicsTransformerLM(vocab_size=VOCAB, context_length=cfg['ctx'],
                                rope_theta=10000.0, **cfg['model']).to(device)
    opt = AdamW(model.parameters(), lr=1e-4)
    token_ids = torch.randint(0, VOCAB, (cfg['batch'], cfg['ctx']), device=device)

    def step():
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['mixed']):
            out = forward_with_ckpt(model, token_ids, blocks_per_seg)
            loss = cross_entropy(out[..., :-1, :], token_ids[..., 1:])
        if mode == 'full':
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        return loss

    step()  # warmup (含 cudnn/cublas 初始化)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    import time
    t0 = time.perf_counter()
    step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    peak = torch.cuda.max_memory_allocated() / 1024**2
    del model, opt
    torch.cuda.empty_cache()
    return peak, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-size', default='medium', choices=list(SIZES))
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--ctx', type=int, default=2048)
    ap.add_argument('--mode', default='full', choices=['full', 'fwd'])
    ap.add_argument('--mixed', action='store_true')
    ap.add_argument('--attention', default='naive', choices=['naive', 'flash-pytorch'],
                    help='flash 会让每 block 的 c 从 ~62 掉回 ~8, 最优 k 随之右移')
    ap.add_argument('--segments', default='0,1,2,4,8,16,36',
                    help='逗号分隔的段大小 k; 0 = 不 checkpoint 的基线')
    args = ap.parse_args()
    if args.attention != 'naive':
        apply_attention_patch(args.attention)
    cfg = dict(model=SIZES[args.model_size], batch=args.batch, ctx=args.ctx, mixed=args.mixed)
    N = cfg['model']['num_layers']
    # 一个残差流张量的字节数 (fp32; mixed 下 backward 时仍是 fp32 存量为主, 近似即可)
    C = args.batch * args.ctx * cfg['model']['d_model'] * 4 / 1024**2

    print(f"model={args.model_size} (N={N} layers), b{args.batch} s{args.ctx}, "
          f"mode={args.mode}{' bf16' if args.mixed else ' fp32'}")
    print(f"1 residual stream tensor = {C:.1f} MiB")
    print(f"{'k(块/段)':>8s} {'段数':>6s} {'ms/step':>9s} {'峰值MB':>9s} {'激活MB(估)':>11s}")
    for k_str in args.segments.split(','):
        k = int(k_str)
        try:
            peak, ms = run_case(cfg, None if k == 0 else k, args.mode)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{k:>8d} {'OOM':>15s}")
            continue
        n_seg = '-' if k == 0 else str((N + k - 1) // k)
        est = '基线' if k == 0 else f"≈{N / k * C + k * 8 * C:.0f}"
        print(f"{k:>8d} {n_seg:>6s} {ms:>9.1f} {peak:>9.1f} {est:>11s}")


if __name__ == '__main__':
    main()
