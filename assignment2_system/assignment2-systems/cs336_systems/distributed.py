"""Distributed data-parallel training: DDP and FSDP.

Both wrappers follow the same lifecycle used by the tests:

- Construction: the wrapper rewrites the underlying module's parameters
  (broadcast / shard) and installs autograd hooks for gradient communication.
- ``forward``: keeps the model functional during forward/backward.
- ``finish_gradient_synchronization``: called by the test harness after
  ``loss.backward()`` and before ``optimizer.step()``. It must wait for any
  in-flight communication so that every parameter has a correct, final
  gradient in ``p.grad``.

Tests (gloo backend, CPU) expect the wrapper to expose the original module as
``self.module``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


# --------------------------------------------------------------------------- #
# DDP: Distributed Data Parallel
# --------------------------------------------------------------------------- #
class DDP(nn.Module):
    """Distributed Data Parallel container.

    Responsibilities:
    1. On construction, broadcast the parameters of rank 0 to all other ranks
       so every replica starts from the same weights.
    2. During the backward pass, all-reduce each parameter's gradient as soon
       as it is ready, *overlapping* communication with the rest of the
       backward computation. Use ``post_accumulate_grad_hook`` (or
       ``register_hook`` on the ``.grad`` tensor) plus asynchronous
       ``all_reduce(..., async_op=True)`` per parameter, then divide the
       reduced gradient by ``world_size`` (averaging).
    3. Expose ``finish_gradient_synchronization()`` that waits on all pending
       all-reduce handles so ``optimizer.step()`` sees the final gradients.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        # 记录"在途"的异步 all-reduce：(参数, 通信句柄) 对
        self._pending: list[tuple[nn.Parameter, dist.Work]] = []

        # 1. 构造时：把 rank0 的参数广播给所有卡，保证各副本起点相同
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # 2. 给每个需要梯度的叶子参数注册钩子：梯度一累加完就异步 all-reduce
        #    用 data_ptr() 去重，避免 tied weights（同一张量）被 reduce 两次
        seen: set[int] = set()
        for param in self.module.parameters():
            if not param.requires_grad or param.data_ptr() in seen:
                continue
            seen.add(param.data_ptr())
            param.register_post_accumulate_grad_hook(self._make_grad_hook(param))

    def _make_grad_hook(self, param: nn.Parameter):
        """返回一个钩子函数：在 param.grad 就绪时被自动调用，发起异步 all-reduce。

        注意必须在闭包里先持有 param（钩子签名只收参数本身），
        async_op=True 让通信在后台进行，backward 继续算其他层——这就是 overlap。
        """

        def hook(_: nn.Parameter) -> None:
            if param.grad is None:
                return
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._pending.append((param, handle))

        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """等待所有在途 all-reduce 完成，再把梯度除以 world_size（求平均）。

        在 loss.backward() 之后、optimizer.step() 之前被调用。
        """
        for _, handle in self._pending:
            handle.wait()
        # wait 之后 all-reduce 才真正写完 param.grad，此时才能安全地做平均
        for param, _ in self._pending:
            param.grad /= self.world_size
        self._pending.clear()


# --------------------------------------------------------------------------- #
# FSDP: Fully-Sharded Data Parallel
# --------------------------------------------------------------------------- #
class FSDP(nn.Module):
    """Fully-Sharded Data Parallel container.

    Parameter partitioning policy (matching the test's expectations):
    - Parameters of ``Linear`` and ``Embedding`` modules (from
      ``cs336_basics.model``) are *sharded* across ranks along dim 0: rank r
      keeps rows ``[r * shard_rows : (r + 1) * shard_rows]`` of each weight.
    - Every other parameter (e.g. ``RMSNorm`` weights) is *replicated* and
      kept in full on every rank.

    Lifecycle (静止态 = fp32 分片):
      forward:  all-gather 把分片拼回完整 fp32 权重(可选转 compute_dtype)，
                param.data 换成 full。不在 forward 末尾 reshard——backward
                还需要完整权重值(如 Linear 的 grad_input)。
      backward: 梯度以 full 形状累加(不做任何通信)。
      finish:   按“所有 rank 一致的规范顺序”逐参数阻塞通信:
                - 分片参数: 梯度转 fp32 → reduce-scatter 成本地片 → 平均，
                  同时把 param.data 换回 fp32 分片(reshard)
                - 复制参数: all-reduce → 平均
                此时 grad.shape == data.shape(都是分片), dtype == fp32。

    通信顺序说明: 集合通信按发出顺序跨 rank 配对。曾尝试在 backward 的
    post-accum 钩子里发异步通信以重叠计算, 但钩子触发顺序受 autograd 引擎
    调度影响, 各 rank 难以保证严格一致, gloo 会静默错配产生垃圾梯度。
    因此这里统一放到 finish 中阻塞执行(TODO: 用固定 bucket 顺序恢复重叠)。

    Mixed precision (``compute_dtype`` 非 None): forward 时把 all-gather 来的
    fp32 权重转成 compute_dtype 参与计算；master 分片始终 fp32；梯度转回 fp32。

    ``gather_full_params()`` 把分片 all-gather 成完整张量返回,
    复制参数原样返回 —— 测试用它和非并行模型逐参数对比。
    """

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # 注册表(key 用 Parameter 对象本身, 别用 data_ptr——
        # forward 会把 param.data 换成 full, 指针会变)
        self._sharded: list[nn.Parameter] = []                  # 分片参数(规范顺序)
        self._replicated: list[nn.Parameter] = []               # 复制参数(规范顺序)
        self._rest_shards: dict[nn.Parameter, torch.Tensor] = {}     # param -> fp32 master 分片
        self._full_shapes: dict[nn.Parameter, tuple[int, ...]] = {}  # param -> 完整形状

        # 1. 分类 + 分片: Linear/Embedding 的 weight 沿 dim0 切, 其余复制
        from cs336_basics.model import Embedding, Linear

        seen: set[nn.Parameter] = set()
        for mod in self.module.modules():
            if not isinstance(mod, (Linear, Embedding)):
                continue
            param = mod.weight
            if not param.requires_grad or param in seen:
                continue
            seen.add(param)
            full_shape = tuple(param.data.shape)
            rows = full_shape[0]
            # 测试保证首维偶数(CHANGELOG 26.1.4); 不整除需 padding, 这里直接断言
            assert rows % self.world_size == 0, (
                f"dim0={rows} 无法被 world_size={self.world_size} 整除: {full_shape}"
            )
            n = rows // self.world_size
            lo, hi = self.rank * n, (self.rank + 1) * n
            param.data = param.data[lo:hi].clone()  # fp32 master 分片(静止态)
            self._sharded.append(param)
            self._rest_shards[param] = param.data
            self._full_shapes[param] = full_shape

        # 2. 其余参数 → 复制态
        for param in self.module.parameters():
            if not param.requires_grad or param in seen:
                continue
            seen.add(param)
            self._replicated.append(param)

    # --- public API ------------------------------------------------------- #

    def forward(self, *args, **kwargs):
        # 1. all-gather: 每个分片参数拼回完整 fp32 权重, 必要时转 compute_dtype
        for param in self._sharded:
            gather = [torch.empty_like(param.data) for _ in range(self.world_size)]
            dist.all_gather(gather, param.data)  # 阻塞: 计算前权重必须就绪
            full = torch.cat(gather, dim=0)
            if self.compute_dtype is not None:
                full = full.to(self.compute_dtype)
            param.data = full
        # 2. 正常前向。不在这里 reshard——backward 还需要完整权重
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """阻塞式梯度同步: 分片参数 reduce-scatter, 复制参数 all-reduce。
        所有 rank 按 _sharded/_replicated 的规范顺序发起集合通信, 保证配对。
        完成后: grad.shape == data.shape(分片), dtype == fp32。"""
        ws = self.world_size
        for param in self._sharded:
            if param.grad is None:
                continue
            g = param.grad.detach()
            if g.dtype != torch.float32:
                g = g.float()  # 混合精度: fp16 梯度转回 fp32(master 语义)
            rest = self._rest_shards[param]
            numel = g.numel()
            chunk = rest.numel()  # 分片大小(测试保证整除, 无需 padding)
            padded = torch.zeros(chunk * ws, dtype=torch.float32, device=g.device)
            padded[:numel] = g.reshape(-1)
            out = torch.empty(chunk, dtype=torch.float32, device=g.device)
            dist.reduce_scatter_tensor(out, padded)  # 阻塞
            param.data = rest  # 先 reshard(此时 data 仍是 full, 必须先换回分片)
            param.grad = (out / ws).view(rest.shape)  # 再赋分片梯度
        for param in self._replicated:
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)  # 阻塞
            param.grad /= ws

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """All-gather every sharded parameter to reconstruct the full tensor.

        Returns a dict mapping parameter names (as in ``module.named_parameters()``)
        to full-size tensors. Replicated parameters are returned as-is.
        """
        result: dict[str, torch.Tensor] = {}
        for name, param in self.module.named_parameters():
            if param in self._full_shapes:
                shard = param.data  # 静止态 fp32 master 分片
                gather = [torch.empty_like(shard) for _ in range(self.world_size)]
                dist.all_gather(gather, shard)
                result[name] = torch.cat(gather, dim=0).view(self._full_shapes[param])
            else:
                result[name] = param.data
        return result
