"""单卡双进程 (gloo) 的 DDP/FSDP 简化 profiling。

跑法（你的终端）:
    cd /root/Myllm/assignment2_system/assignment2-systems
    .venv/bin/python /root/Myllm/exercises/ddp_fsdp_profile.py

注意/局限（写 writeup 时要声明）:
- 两个进程共享同一张 3090: 显存数字是"每进程"的, 两进程之和才是 GPU 总占用;
  优化器分片的 1/N 节省体现为每进程峰值下降, 而不是"装下装不下的模型"
- gloo 走主机内存转发, 不代表真实 NVLink/PCIe P2P 带宽, 计时是悲观上界
- 对比项: baseline(无并行) / ddp(全量优化器) / ddp-sharded(ZeRO-1) / fsdp(ZeRO-3)
"""
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.distributed import DDP, FSDP
from cs336_systems.optimizer import ShardedOptimizer

MODEL = dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12,
             vocab_size=10000, context_length=512)
VARIANTS = ["baseline", "ddp", "ddp-sharded", "fsdp"]


def build(variant, device):
    torch.manual_seed(1234)  # 各变体同 seed, 权重一致 (broadcast 变成纯通信演示)
    model = BasicsTransformerLM(**MODEL).to(device)
    if variant == "ddp":
        model = DDP(model)
        return model, torch.optim.AdamW(model.parameters(), lr=1e-4)
    if variant == "ddp-sharded":
        model = DDP(model)
        return model, ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=1e-4)
    if variant == "fsdp":
        model = FSDP(model)
        return model, ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=1e-4)
    return model, torch.optim.AdamW(model.parameters(), lr=1e-4)


def worker(rank: int, world_size: int, variant: str):
    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT="29501")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    device = "cuda"
    model, opt = build(variant, device)

    x = torch.randint(0, MODEL["vocab_size"], (4, MODEL["context_length"]), device=device)
    targets = torch.randint(0, MODEL["vocab_size"], (4, MODEL["context_length"]), device=device)

    def step():
        opt.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, targets)
        loss.backward()
        if variant != "baseline":
            model.finish_gradient_synchronization()
        opt.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()  # 只统计训练期峰值 (排除建模型的开销差异干扰较小)
    times = []
    for i in range(7):  # 2 warmup + 5 timed
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        if i >= 2:
            times.append(time.perf_counter() - t0)

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{variant:>11}] rank{rank}: 每步 {sum(times) / len(times) * 1000:7.1f}ms | "
          f"每进程峰值显存 {peak:.2f}GB", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), "\n")
    for variant in VARIANTS:
        mp.spawn(worker, args=(2, variant), nprocs=2, join=True)
    print("\n提示: 显存对比看 ddp vs ddp-sharded (优化器状态减半) 和 fsdp (参数/梯度也减半);")
    print("      计时受 gloo 主机内存转发影响, 只能看相对关系, 不能当绝对通信性能。")
