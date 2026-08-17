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

        # TODO(you): keep a registry of in-flight all-reduce handles, e.g.
        #   self._pending_handles: list[dist.Work] = []
        #   self._bucket_to_handle: dict[nn.Parameter, dist.Work] = {}

        # 1. Broadcast all parameters (and buffers) from rank 0 so every
        #    replica starts with the same weights.
        #    e.g. for param in module.parameters(): dist.broadcast(param.data, src=0)

        # 2. For every leaf parameter with requires_grad=True, register a
        #    post-accumulate-grad hook that starts an async all-reduce of
        #    ``param.grad`` and records the returned handle.
        #    NOTE: handle tied/shared parameters (same tensor used twice) to
        #    avoid double-reducing; tie by ``param.data_ptr()``.

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """Wait for all pending gradient all-reduces, then divide by world_size.

        Called by the training loop right after ``loss.backward()``.
        """
        raise NotImplementedError  # TODO(you): implement


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

    Responsibilities:
    1. Shard the parameters, then all-gather them back to full tensors before
       each forward / backward so compute sees unsharded weights.
    2. During backward: sharded params reduce-scatter their (full) gradients
       back to the owning rank; replicated params all-reduce their gradients.
    3. Mixed precision (``compute_dtype`` is not None): cast the *full*
       gathered weights to ``compute_dtype`` before forward/backward compute,
       keep master (sharded) weights in fp32, and cast gradients back to fp32.
    4. ``gather_full_params()`` all-gathers every sharded param into a full
       tensor and returns ``{param_name: full_tensor}``; replicated params are
       returned as-is. This is how tests compare against a non-parallel model.
    """

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # Registry of sharded parameters, e.g.:
        #   self._sharded_params: list[nn.Parameter]     # the local shards
        #   self._shard_to_full_shape: dict[nn.Parameter, tuple[int, ...]]
        #   self._all_gather_handles: list[dist.Work]
        #   self._reduce_scatter_handles: list[dist.Work]

        # 1. Classify parameters: for each (name, param) in module.named_parameters():
        #    - if the containing module is a cs336_basics Linear/Embedding:
        #      shard along dim 0. Replace param.data with the local shard
        #      (a new nn.Parameter). Store the original full shape.
        #    - else: leave replicated, but register an all-reduce grad hook.
        # 2. For sharded params, register a grad hook that reduce-scatters the
        #    full gradient down to the local shard.
        # 3. Implement all-gather logic used before forward (see ``_gather`` /
        #    ``_reshard`` helpers).

    # --- helpers (structure to fill in) ---------------------------------- #

    def _all_gather_params(self) -> None:
        """All-gather all shards into full tensors, cache them on the module,
        and (if compute_dtype is set) cast them to compute_dtype for compute.
        The master fp32 shards must remain untouched."""
        raise NotImplementedError  # TODO(you): implement

    def _reduce_scatter_grads(self) -> None:
        """Reduce-scatter the full gradients of sharded params so each rank
        holds the gradient of its own shard; cast to fp32 if mixed precision."""
        raise NotImplementedError  # TODO(you): implement

    def _reshard(self) -> None:
        """Restore the module's parameters back to their local shards after
        forward/backward so the optimizer only ever sees sharded params."""
        raise NotImplementedError  # TODO(you): implement

    # --- public API ------------------------------------------------------- #

    def forward(self, *args, **kwargs):
        # 1. self._all_gather_params()
        # 2. out = self.module(*args, **kwargs)
        # 3. self._reshard()
        # 4. return out
        raise NotImplementedError  # TODO(you): implement

    def finish_gradient_synchronization(self):
        """Wait for all pending reduce-scatter / all-reduce handles so every
        local parameter holds its final gradient (in fp32, shape == data)."""
        raise NotImplementedError  # TODO(you): implement

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """All-gather every sharded parameter to reconstruct the full tensor.

        Returns a dict mapping parameter names (as in ``module.named_parameters()``)
        to full-size tensors. Replicated parameters are returned as-is.
        """
        raise NotImplementedError  # TODO(you): implement
