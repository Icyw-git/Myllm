"""验证 ShardedOptimizer 的隐含前提（ZeRO-1 试金石实验）。

world_size=2，每个 rank 喂不同的数据（这正是官方测试没覆盖的场景）：

- Case A: 不同数据 + 不同步梯度，直接 step
          → 预期"炸"：各卡用各自梯度的不同片段更新，拼出弗兰肯斯坦参数
- Case B: 不同数据 + 先把梯度跨卡平均（模拟 DDP all-reduce），再 step
          → 预期与"全量 batch + 普通 SGD"的参考严格一致
- 同时检查 Case B 的参数跨 rank 一致性

参考(ground truth)：全量 batch(两卡数据拼接) + 普通 SGD 的一步更新。
"""
import copy
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

sys.path.insert(0, "/root/Myllm/assignment2_system/assignment2-systems")
from cs336_systems.optimizer import ShardedOptimizer


def worker(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 两份数据所有 rank 都能确定性重建，保证和参考用的是同一批
    torch.manual_seed(100)
    x0 = torch.randn(8, 4)
    torch.manual_seed(101)
    x1 = torch.randn(8, 4)
    x_local = [x0, x1][rank]          # 本 rank 只见自己的 8 条
    x_full = torch.cat([x0, x1])      # 参考的全量 16 条

    torch.manual_seed(7)              # 所有 rank 初始化相同
    base = nn.Linear(4, 2)
    loss_fn = lambda m, x: ((m(x) - 1) ** 2).mean()

    def fresh():
        return copy.deepcopy(base)

    # ---- 参考：全量 batch + 普通 SGD 一步 ----
    ref = fresh()
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.1)
    loss_fn(ref, x_full).backward()
    ref_opt.step()
    ref_w = ref.weight.data

    # ---- Case A: 不同数据 + 不同步梯度 ----
    a = fresh()
    opt_a = ShardedOptimizer(a.parameters(), torch.optim.SGD, lr=0.1)
    loss_fn(a, x_local).backward()
    opt_a.step()
    diff_a = (a.weight.data - ref_w).abs().max().item()

    # ---- Case B: 不同数据 + 先跨卡平均梯度（模拟 DDP all-reduce）----
    b = fresh()
    opt_b = ShardedOptimizer(b.parameters(), torch.optim.SGD, lr=0.1)
    loss_fn(b, x_local).backward()
    for p in b.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)  # DDP 在 backward 里做的事
        p.grad /= world_size
    opt_b.step()
    diff_b = (b.weight.data - ref_w).abs().max().item()

    # ---- Case B 跨 rank 一致性 ----
    w_gather = [torch.zeros_like(b.weight.data) for _ in range(world_size)]
    dist.all_gather(w_gather, b.weight.data)
    cross_rank_diff = (w_gather[0] - w_gather[1]).abs().max().item()

    print(f"rank{rank} | Case A 与参考误差: {diff_a:.6f} | "
          f"Case B 与参考误差: {diff_b:.2e} | B 跨 rank 误差: {cross_rank_diff:.2e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
    print("\n预期: Case A 有明显误差(弗兰肯斯坦), Case B ≈ 0 且跨 rank 一致")
