"""all-reduce 通信 benchmark（gloo 单机多进程模拟, handout all_reduce 题简化版）.

3090 适配声明: 单卡机器无 NVLink/NCCL多卡, 用 gloo + 多进程模拟。
gloo 的 CUDA all_reduce 走 GPU->CPU->ring(共享内存)->CPU->GPU,
测得的是 PCIe+主机内存带宽, 不是真 NCCL ring 带宽 —— writeup 需声明,
但 **总线流量公式 2(n-1)/n × S 与延迟结构不变**, 结论可迁移。

跑法:
    cd /root/Myllm/assignment2_system/assignment2-systems
    .venv/bin/python /root/Myllm/exercises/bench_allreduce.py
"""
import os
import statistics as st
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES_MB = [1, 10, 100, 1000]
WORLD_SIZES = [2, 4, 6]
ITERS, WARMUP = 20, 5


def worker(rank: int, world: int, size_mb: int, results):
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29511'
        dist.init_process_group('gloo', rank=rank, world_size=world)
        n = size_mb * 1024 * 1024 // 4
        x = torch.randn(n).cuda() if torch.cuda.is_available() else torch.randn(n)
        for _ in range(WARMUP):
            dist.all_reduce(x)
        if rank == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.barrier()
        ts = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            dist.all_reduce(x)
            if rank == 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
            dist.barrier()
        if rank == 0:
            results[f'{world},{size_mb}'] = (st.mean(ts), st.stdev(ts))
        dist.destroy_process_group()
    except Exception as e:
        print(f'[rank{rank} world{world} {size_mb}MB] FAILED: {e!r}', flush=True)
        raise


def main():
    mgr = mp.Manager()
    results = mgr.dict()
    ctx = mp.get_context('spawn')
    for world in WORLD_SIZES:
        for mb in SIZES_MB:
            procs = []
            for r in range(world):
                p = ctx.Process(target=worker, args=(r, world, mb, results))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
    print('world,MB,ms,std,alg_GB/s,bus_GB/s', flush=True)
    for key in sorted(results, key=lambda k: (int(k.split(',')[0]), int(k.split(',')[1]))):
        world, mb = map(int, key.split(','))
        t, s = results[key]
        alg = mb / 1024 / t
        bus = 2 * (world - 1) / world * alg
        print(f'{world},{mb},{t*1000:.2f},{s*1000:.2f},{alg:.2f},{bus:.2f}', flush=True)


if __name__ == '__main__':
    main()
