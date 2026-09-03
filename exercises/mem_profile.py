"""memory_profiling (2.1.6): 内存时间线 pickle + 峰值表 + 最大分配块.

3090 适配: xl full step 需 16B/参数×3.41B=54.6GB 固定开销 > 24GB, 不可行;
用 small (0.13B) 跑 ctx 128 / 2048 的 full step, writeup 声明局限。

跑法:
    cd /root/Myllm/assignment2_system/assignment2-systems
    .venv/bin/python /root/Myllm/exercises/mem_profile.py
输出:
    1) 峰值表: {ctx}×{fwd,full}×{fp32,bf16}
    2) 两个 pickle: /tmp/mem_ctx128.pkl, /tmp/mem_ctx2048.pkl
       (上传 https://pytorch.org/memory_viz 生成 Active memory timeline 截图)
    3) 最大分配块 top10 (来自 pickle 解析)
"""
import pickle

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

CFG = dict(d_model=512, d_ff=2048, num_layers=12, num_heads=8,
           vocab_size=10000, context_length=2048)


def run_steps(ctx: int, mode: str, mixed: bool, steps: int = 10):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    cfg = dict(CFG, context_length=ctx)
    model = BasicsTransformerLM(**cfg).to('cuda')
    opt = AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, 10000, (4, ctx), device='cuda')
    y = torch.randint(0, 10000, (4, ctx), device='cuda')

    def step():
        with torch.autocast('cuda', torch.bfloat16, enabled=mixed):
            if mode == 'forward':
                with torch.no_grad():
                    model(x)
                return
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()  # 参数(+优化器状态)常驻
    for _ in range(steps):
        step()
    peak = torch.cuda.max_memory_allocated()
    del model, opt, x, y
    torch.cuda.empty_cache()
    return peak / 1024**2, base / 1024**2


def snapshot_and_dump(ctx: int, path: str, steps: int = 10):
    """full step fp32, 记录内存时间线并 dump pickle."""
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    cfg = dict(CFG, context_length=ctx)
    model = BasicsTransformerLM(**cfg).to('cuda')
    opt = AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, 10000, (4, ctx), device='cuda')
    y = torch.randint(0, 10000, (4, ctx), device='cuda')

    def step():
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    torch.cuda.memory._dump_snapshot(path)
    torch.cuda.memory._record_memory_history(None)
    print(f'snapshot -> {path}')
    del model, opt, x, y
    torch.cuda.empty_cache()


def top_blocks(path: str, k: int = 10):
    with open(path, 'rb') as f:
        snap = pickle.load(f)
    blocks = []
    for seg in snap.get('segments', []):
        for b in seg.get('blocks', []):
            if b.get('state') in ('active_allocated', 'active'):
                blocks.append(b['size'])
    blocks.sort(reverse=True)
    print(f'\ntop{k} 最大活跃块 ({path}):')
    for s in blocks[:k]:
        print(f'  {s / 1024**2:8.1f} MB')


def main():
    print('=== 峰值表 (peak MB | 常驻 MB) ===')
    print('ctx,mode,dtype,peak_MB,resident_MB')
    for ctx in (128, 2048):
        for mode in ('forward', 'full'):
            for mixed in (False, True):
                dt = 'bf16' if mixed else 'fp32'
                peak, base = run_steps(ctx, mode, mixed)
                print(f'{ctx},{mode},{dt},{peak:.0f},{base:.0f}')

    snapshot_and_dump(128, '/tmp/mem_ctx128.pkl')
    snapshot_and_dump(2048, '/tmp/mem_ctx2048.pkl')
    top_blocks('/tmp/mem_ctx128.pkl')
    top_blocks('/tmp/mem_ctx2048.pkl')


if __name__ == '__main__':
    main()
