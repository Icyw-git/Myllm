"""FlashAttention (FlashAttention-2 style) implemented as torch.autograd.Function.

This module provides two interchangeable implementations of the same
flash-attention algorithm:

1. ``FlashAttentionPytorchAutogradFunction``: a pure-PyTorch reference
   implementation (no Triton). Used for correctness checking.
2. ``FlashAttentionTritonAutogradFunction``: a Triton-kernel implementation
   used for actual speedups on GPU.

Both must satisfy the same contract:

- ``forward(q, k, v, is_causal)`` returns ``o`` of shape ``(B, N, D)``.
- The forward pass must save the log-sum-exp of the attention scores, of
  shape ``(B, N)``, via ``ctx.save_for_backward`` so that tests can inspect it
  (the tests search for exactly one saved tensor of shape ``(B, N)``).
- ``backward(ctx, do)`` returns gradients ``(dq, dk, dv, None)``.

Algorithm (FlashAttention-2, block-wise online softmax):

Forward: iterate over KV blocks ``j``, maintaining running max ``m``,
running sum ``l`` (row sums of exp(S - m)) and output accumulator ``o``.

    S_j    = q @ k_j^T * scale
    m_new  = max(m, rowmax(S_j))
    alpha  = exp(m - m_new)
    p_j    = exp(S_j - m_new)
    l      = alpha * l + rowsum(p_j)
    o      = alpha * o + p_j @ v_j

Finalize with ``lse = m + log(l)`` and scale ``o = o / l``.

Backward: recompute ``p = exp(S - lse)`` for each KV block and accumulate
``dv = p^T @ do``, ``dp = do @ v^T``, ``dq += dp @ k``, ``dk += dp^T @ q``.

Reference for the exact math: https://arxiv.org/abs/2205.14135
"""

import torch
from torch import Tensor


def _attention_and_lse(q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> tuple[Tensor, Tensor]:
    """Reference (non-flash) attention used only for cross-checking.

    Computes ``o`` and the log-sum-exp ``lse`` of the scaled attention scores.
    """
    n_queries = q.shape[-2]
    n_keys = k.shape[-2]
    scale = 1 / (q.shape[-1] ** 0.5)
    s = torch.einsum("...qd,...kd->...qk", q, k) * scale
    if is_causal:
        # Keep only positions where query_index >= key_index.
        mask = torch.arange(n_queries, device=s.device)[None, :, None] >= torch.arange(n_keys, device=s.device)[None, None, :]
        s = torch.where(mask, s, -1e6)
    p = torch.softmax(s, dim=-1)
    o = torch.einsum("...qk,...kd->...qd", p, v)
    lse = torch.logsumexp(s, dim=-1)
    return o, lse


class FlashAttentionPytorchAutogradFunction(torch.autograd.Function):
    """FlashAttention-2 forward/backward written in pure PyTorch (no Triton).

    NOTE: the block size is a free hyper-parameter; pick something that
    balances register pressure and the number of kernel launches.
    """

    BLOCK_SIZE = 128  # TODO(you): tune this (commonly 64/128/256).

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
        """Run the block-wise online-softmax forward pass.

        Args:
            q: (B, N_Q, D) query tensor.
            k: (B, N_K, D) key tensor.
            v: (B, N_K, D) value tensor.
            is_causal: if True, apply a causal mask.
        Returns:
            o: (B, N_Q, D) attention output.
        """


        # 1. Initialize m = full(-inf, (B, N_Q)), l = zeros((B, N_Q)),
        #    o = zeros((B, N_Q, D)).
        # 2. Loop over KV blocks j (block size BLOCK_SIZE):
        #    - compute S_j = q @ k_j^T * scale
        #    - if is_causal, mask out future positions with -inf
        #    - online-softmax update (see module docstring)
        # 3. lse = m + log(l); o = o / l.
        # 4. ctx.save_for_backward(q, k, v, lse); ctx.is_causal = is_causal.
        batch,n_q,d=q.shape
        n_k=k.shape[-2]
        device,dtype=q.device,q.dtype
        block=FlashAttentionPytorchAutogradFunction.BLOCK_SIZE
        scale=d ** -0.5
        m=torch.full((batch,n_q),float('-inf'),device=device,dtype=dtype) #每块的最大值，逐行更新
        l=torch.zeros_like(m) #每块的运行和，逐行更新
        o=torch.zeros_like(q) #累计输出，每次循环更新
        for j0 in range(0,n_k,block):
            j1=min(j0+block,n_k)  # 最后一块可能不满，取实际长度
            k_j=k[:,j0:j1]
            v_j=v[:,j0:j1]
            s=torch.matmul(q,k_j.transpose(-2,-1))*scale #计算当前块的注意力分数，形状为(B,N_Q,N_K)
            if is_causal:
                q_idx=torch.arange(n_q,device=device).reshape(n_q,1) #查询索引，形状为(N_Q,1)，变成一列
                k_idx=torch.arange(j0,j1,device=device).reshape(1,j1-j0) #key索引，形状为(1,N_K)，变成一行
                s=torch.where(q_idx>=k_idx,s,float('-inf')) #对未来位置的分数进行掩码，值为-inf,这两个形状为(N_Q,N_K)
            m_new=torch.max(m,s.max(dim=-1).values) #当前块的最大值，形状为(B,N_Q)
            p=torch.exp(s-m_new.unsqueeze(-1)) #当前块的注意力权重，形状为(B,N_Q,N_K)，稳定计算
            alpha=torch.exp(m-m_new) #当前的系数，形状为(N_Q)
            l=alpha*l+p.sum(dim=-1)  #更新当前块的运行和，形状为(B,N_Q)
            o=alpha.unsqueeze(-1)*o+torch.matmul(p,v_j) #更新当前块的输出，形状为(B,N_Q,D)
            m=m_new
        lse=m+torch.log(l) #计算当前块的注意力分数，形状为(B,N_Q)，这是取对数后的结果
        o=o/l.unsqueeze(-1) #归一化当前块的输出，形状为(B,N_Q,D)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.is_causal=is_causal
        return o

    @staticmethod
    def backward(ctx, do: Tensor) -> tuple[Tensor | None, ...]:
        """Return (dq, dk, dv, None) matching the forward's gradient.

        Recompute ``p = exp(S - lse)`` block-wise from the saved q/k/v/lse
        and accumulate dv, dp, dq, dk as described in the module docstring.
        """
        q,k,v,o,lse=ctx.saved_tensors
        batch,n_q,dim=q.shape
        n_k=k.shape[-2]
        device,dtype=q.device,q.dtype
        block=FlashAttentionPytorchAutogradFunction.BLOCK_SIZE
        scale=dim ** -0.5
        dq=torch.zeros_like(q)
        dk=torch.zeros_like(k)
        dv=torch.zeros_like(v)
        d=(do*o).sum(dim=-1)  # rowsum(do ⊙ o)，每个 query 行的标量
        for j0 in range(0,n_k,block):
            j1=min(j0+block,n_k)  # 最后一块可能不满
            k_j=k[:,j0:j1]
            v_j=v[:,j0:j1]
            s=torch.matmul(q,k_j.transpose(-2,-1))*scale
            if ctx.is_causal:
                q_idx=torch.arange(n_q,device=device).reshape(n_q,1)
                k_idx=torch.arange(j0,j1,device=device).reshape(1,j1-j0)
                s=torch.where(q_idx>=k_idx,s,float('-inf'))
            p=torch.exp(s-lse.unsqueeze(-1))  # 重算 P，不存它
            dp=torch.matmul(do,v_j.transpose(-2,-1))  # dO V_j^T
            ds=p*(dp-d.unsqueeze(-1))  # dS = P ⊙ (dP - d)
            dq=dq+torch.matmul(ds,k_j)*scale
            dk[:,j0:j1]=torch.matmul(ds.transpose(-2,-1),q)*scale
            dv[:,j0:j1]=torch.matmul(p.transpose(-2,-1),do)
        return dq,dk,dv,None


class FlashAttentionTritonAutogradFunction(torch.autograd.Function):
    """FlashAttention-2 forward/backward implemented with Triton kernels.

    The forward and backward should each launch a single fused Triton kernel
    (or a small set of kernels), one program per ``(batch, query-block)``.

    Suggested kernel layout:
    - Forward kernel: grid = (ceil(N_Q / BLOCK_M), B), each program loops over
      KV blocks of size BLOCK_N, keeps running (m, l, o) in registers, and
      finally writes ``o`` and ``lse`` to global memory.
    - Backward kernel: recompute softmax from saved lse, accumulate
      dq/dk/dv. Either a single kernel that loops over KV blocks and writes
      dq (with atomic adds if needed), or split kernels per gradient.

    NOTE: requires a CUDA GPU; tests are skipped automatically otherwise.
    """

    BLOCK_M = 128  # TODO(you): tune
    BLOCK_N = 128  # TODO(you): tune

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
        raise NotImplementedError  # TODO(you): implement

    @staticmethod
    def backward(ctx, do: Tensor) -> tuple[Tensor | None, ...]:
        raise NotImplementedError  # TODO(you): implement
