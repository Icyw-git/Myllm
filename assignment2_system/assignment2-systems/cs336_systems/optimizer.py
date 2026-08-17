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
        self.rank = dist.get_rank()

        # Registry of (full_param, local_shard) pairs:
        #   self._shards: dict[torch.Tensor, torch.Tensor]   # param -> shard

        # 1. For each unique parameter (dedupe shared/tied weights), flatten
        #    and split into world_size chunks; keep this rank's chunk as a
        #    persistent tensor.
        # 2. Construct the wrapped optimizer over the local shards:
        #    self._wrapped = optimizer_cls(self._shards.values(), **kwargs)
        #    NOTE: pass the *shard tensors themselves* (not nn.Parameters of
        #    the model) so the wrapped optimizer's state keys are stable.

        raise NotImplementedError  # TODO(you): implement

    @torch.no_grad()
    def step(self, closure=None):
        """Update parameters in place using only locally-held optimizer state.

        Steps:
        1. Copy the local slice of each full gradient into the shard's ``.grad``.
        2. ``self._wrapped.step()``.
        3. All-gather the updated shards and copy the reconstructed full
           parameter back into ``param.data``.
        """
        raise NotImplementedError  # TODO(you): implement

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero out all full-size parameter gradients and the wrapped
        optimizer's shard gradients."""
        super().zero_grad(set_to_none=set_to_none)
        raise NotImplementedError  # TODO(you): implement
