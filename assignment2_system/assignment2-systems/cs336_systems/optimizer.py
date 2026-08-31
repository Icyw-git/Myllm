"""Optimizer state sharding (ZeRO-style).

``ShardedOptimizer`` wraps any ``torch.optim.Optimizer`` subclass (e.g.
AdamW/Adam/SGD) so that each rank only holds the *optimizer state* for its
slice of each parameter, while the parameters themselves stay full-size on
every rank.

Contract (from the tests):

- ``get_sharded_optimizer(params, optimizer_cls, **kwargs)`` returns an
  instance of ``ShardedOptimizer``.
- After an equal number of ``zero_grad / backward / step`` cycles, the model
  weights must be bit-identical to training with the unsharded optimizer,
  and identical across ranks.

Implementation sketch:

- On construction, for each parameter ``p``, flatten it and split it into
  ``world_size`` contiguous chunks. Keep a *persistent* local shard tensor
  ``shard_p`` (one per parameter) for this rank.
- Build the wrapped optimizer over the shard tensors:
  ``self._wrapped = optimizer_cls(shard_tensors, **kwargs)``. Because the
  shard tensors are stable objects, the wrapped optimizer's ``state`` dict
  stays valid across steps.
- ``step()``:
  1. Copy each parameter's local gradient slice into ``shard_p.grad``
     (``p.grad`` is full-size since the model is not sharded).
  2. Call ``self._wrapped.step()``; this updates the shard in-place using
     only the local optimizer state (exp_avg / exp_avg_sq / ...).
  3. All-gather all shards and write the reconstructed full tensor back into
     ``p.data`` so every rank has the identical updated parameter.
- ``zero_grad`` forwards to the wrapped optimizer (and zeroes the full
  parameter gradients, which are owned by the model).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    """Optimizer that shards optimizer state across ranks."""

    def __init__(
        self,
        params: Iterable[torch.Tensor | dict],
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs,
    ):
        defaults = dict(kwargs)
        super().__init__(params, defaults)  # keeps param groups in `self.param_groups`

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank() #每个 rank 的索引

        # Registry of (full_param, local_shard) pairs:
        #   self._shards: dict[torch.Tensor, torch.Tensor]   # param -> shard
        self._shards: dict[torch.Tensor, torch.Tensor] = {} # 每个参数的本地 shard 张量，大小为 ``chunk_size``
        self._chunk_sizes: dict[torch.Tensor, int] = {}  # param -> 每片的元素个数（含补齐）

        # 1. For each unique parameter (dedupe shared/tied weights), flatten
        #    and split into world_size chunks; keep this rank's chunk as a
        #    persistent tensor.
        seen: set[int] = set() #参数去重，避免重复处理共享权重
        for group in self.param_groups:
            for param in group["params"]:
                if param.data_ptr() in seen:
                    continue
                seen.add(param.data_ptr())

                numel = param.numel() # 参数个数
                # 每片大小向上取整；不整除时最后一片补零（all-gather 要求各卡等长）
                chunk = (numel + self.world_size - 1) // self.world_size # 每片的元素个数
                flat = param.data.view(-1) # 展平参数张量，方便切片
                lo, hi = self.rank * chunk, min((self.rank + 1) * chunk, numel) #
                # 持久化 shard：优化器状态挂在它身上，跨 step 存活
                self._shards[param] = flat[lo:hi].clone() # 每个参数的本地 shard 张量，大小为 ``chunk_size``
                self._chunk_sizes[param] = chunk # 每片的元素个数（含补齐）

        # 2. Construct the wrapped optimizer over the local shards:
        #    self._wrapped = optimizer_cls(self._shards.values(), **kwargs)
        #    NOTE: pass the *shard tensors themselves* (not nn.Parameters of
        #    the model) so the wrapped optimizer's state keys are stable.
        self._wrapped = optimizer_cls(list(self._shards.values()), **kwargs) # 包装优化器，只包含当前 rank 的 shard 张量

    @torch.no_grad()
    def step(self, closure=None):
        """Update parameters in place using only locally-held optimizer state.

        Steps:
        1. Copy the local slice of each full gradient into the shard's ``.grad``.
        2. ``self._wrapped.step()``.
        3. All-gather the updated shards and copy the reconstructed full
           parameter back into ``param.data``.
        """
        # 1. 把完整梯度的本地切片交给 shard（clone 出独立张量，别用视图）
        # 前提：p.grad 必须是"已全局归约(平均)后的最终梯度"——
        #   - standalone: 各 rank 数据相同，天然成立
        #   - 配 DDP:     all-reduce 已在 backward 中完成
        #   - 配 FSDP:    reduce-scatter 已把本地片写进 p.grad（形状==shard）
        for param, shard in self._shards.items():
            g = param.grad
            if g is None:
                shard.grad = None
            elif g.numel() == param.numel():
                # 完整梯度：切出自己负责的区间
                chunk = self._chunk_sizes[param]
                lo, hi = self.rank * chunk, (self.rank + 1) * chunk
                shard.grad = g.view(-1)[lo:hi].clone()
            else:
                # 已经是 reduce-scatter 后的本地片（FSDP 形态）：直接用，勿再归约
                shard.grad = g.view(-1)[: shard.numel()].clone() # 每个参数的本地 shard 张量的梯度，大小为 ``chunk_size``

        # 2. 本地 Adam 更新：只用本地 m/v（ZeRO-1 等价性的来源）
        self._wrapped.step()

        # 3. all-gather 各卡更新后的 shard，拼回完整参数写回 p.data
        for param, shard in self._shards.items():
            chunk = self._chunk_sizes[param] # 每片的元素个数（含补齐）
            numel = param.numel() # 参数个数
            # 补零到等长，满足 all-gather 的等尺寸要求
            padded = torch.zeros(chunk, dtype=shard.dtype, device=shard.device)
            padded[: shard.numel()] = shard
            gather_list = [torch.zeros_like(padded) for _ in range(self.world_size)] # 为all-gather通信准备的列表，每个元素为 ``chunk_size`` 的张量，示例：[torch.zeros(10), torch.zeros(10), ...]
            dist.all_gather(gather_list, padded) #all-gather之后的结果是在 ``gather_list`` 中，每个元素为 ``chunk_size`` 的张量，示例：[torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), ...]
            param.data.view(-1).copy_(torch.cat(gather_list)[:numel])

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero out all full-size parameter gradients and the wrapped
        optimizer's shard gradients."""
        super().zero_grad(set_to_none=set_to_none)
        for shard in self._shards.values():
            shard.grad = None
