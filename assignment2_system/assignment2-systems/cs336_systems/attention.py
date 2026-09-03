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
import triton
import triton.language as tl
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

    BLOCK_M = 64  # 每个 program 负责的 query 行数
    BLOCK_N = 64  # 每次内层循环处理的 key 数

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
        """Run the flash-attention forward pass with a single fused Triton kernel.

        One program per (query-block, batch): loads its Q block once into
        registers/SMEM, then streams over K/V blocks applying the online
        softmax update, finally writing o and lse to global memory.
        """
        B, N_Q, D = q.shape
        N_K = k.shape[-2]
        assert q.is_cuda, "expect CUDA inputs"
        assert q.dtype in (torch.float32, torch.bfloat16, torch.float16), (
            f"unsupported dtype {q.dtype}: fp32 走 CUDA core(ieee), "
            "bf16/fp16 走 tensor core")
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

        o = torch.empty_like(q)
        lse = torch.empty(B, N_Q, device=q.device, dtype=torch.float32)

        BLOCK_M = FlashAttentionTritonAutogradFunction.BLOCK_M
        BLOCK_N = FlashAttentionTritonAutogradFunction.BLOCK_N
        BLOCK_D = triton.next_power_of_2(D)
        grid = (triton.cdiv(N_Q, BLOCK_M), B)

        _flash_attn_fwd_kernel[grid](
            q, k, v, o, lse,
            N_Q, N_K, D, D ** -0.5,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            lse.stride(0), lse.stride(1),
            IS_CAUSAL=is_causal,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4,
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do: Tensor) -> tuple[Tensor | None, ...]:
        """Triton backward: 拆成 dQ 与 dK/dV 两个 kernel（各自内部零写冲突）。

        算法与 PyTorch 版完全相同（见 FlashAttentionPytorchAutogradFunction.backward）:
        1. 预处理 delta = rowsum(dO ⊙ O)，(B, N_Q)，每行一个标量
        2. dQ kernel: 每个 program 负责一个 Q 块, 沿 KV 循环, 用 lse 重算
           P = exp(S - lse)，累积 dq —— dq 只被本 program 写, 直接 store
        3. dK/dV kernel: 每个 program 负责一个 K 块, 沿 Q 循环, 累积 dk/dv
           —— 若仍按 Q 块分 program, 多个 program 会写同一片 dk/dv（冲突）,
           所以按 K 块分 grid, 转置一下循环方向即可零冲突
        不存 N² 的 P, 用重算 S=QK^T（+2N²D FLOPs）换 O(N) 显存。
        """
        q, k, v, o, lse = ctx.saved_tensors
        do = do.contiguous()
        B, N_Q, D = q.shape
        N_K = k.shape[-2]
        assert do.shape == o.shape

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        # 预处理用 PyTorch 一行即可（N×D 逐行内积, 不是瓶颈）; fp32 保证精度
        delta = (do.float() * o.float()).sum(-1)  # (B, N_Q)

        BLOCK_M = FlashAttentionTritonAutogradFunction.BLOCK_M
        # backward 循环内比 forward 多驻留 q/do 常驻 tile + trans 缓冲,
        # KV 块用 32 才能塞进 3090 的 99KB smem（forward 用 64）
        BLOCK_N = 32
        BLOCK_D = triton.next_power_of_2(D)

        _flash_attn_dq_kernel[(triton.cdiv(N_Q, BLOCK_M), B)](
            q, k, v, do, lse, delta, dq,
            N_Q, N_K, D, D ** -0.5,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            lse.stride(0), lse.stride(1),
            delta.stride(0), delta.stride(1),
            IS_CAUSAL=ctx.is_causal,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,  # 循环内驻留 k/v 两块 tile, 3 级流水会超 3090 的 99KB smem
        )
        _flash_attn_dkdv_kernel[(triton.cdiv(N_K, BLOCK_N), B)](
            q, k, v, do, lse, delta, dk, dv,
            N_Q, N_K, D, D ** -0.5,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            lse.stride(0), lse.stride(1),
            delta.stride(0), delta.stride(1),
            IS_CAUSAL=ctx.is_causal,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )
        return dq, dk, dv, None


@triton.jit
def _flash_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    N_Q, N_K, D, scale,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_lseb, stride_lsen,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """FlashAttention-2 forward, Algorithm 1 (https://arxiv.org/abs/2205.14135).

    grid = (ceil(N_Q / BLOCK_M), B)。每个 program：
      1. 载入自己的 Q 块 (BLOCK_M, D)
      2. 沿 K 维循环 KV 块，维护运行态 (m, l, acc)
      3. 收尾：acc /= l，写回 o 和 lse = m + log(l)
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # 本 program 负责的 query 行
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < N_Q
    d_mask = offs_d < D  # D 不是 2 的幂时保护（测试里 D=64=BLOCK_D，恒真）

    base = pid_b * stride_qb
    q = tl.load(
        q_ptr + base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
        mask=q_mask[:, None] & d_mask[None, :], other=0.0,
    )  # (BLOCK_M, BLOCK_D)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)  # 运行 max
    l_i = tl.zeros([BLOCK_M], tl.float32)                # 运行 softmax 分母
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)       # 运行输出（未归一化）

    # causal 时，query 行 i 只需要 key 0..i，所以本 program 只需处理
    # key < (pid_m+1)*BLOCK_M；后面的块整块都在"未来"，直接跳过
    if IS_CAUSAL:
        hi = tl.minimum(N_K, (pid_m + 1) * BLOCK_M)
    else:
        hi = N_K

    for n0 in range(0, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < N_K

        k = tl.load(
            k_ptr + pid_b * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=kv_mask[:, None] & d_mask[None, :], other=0.0,
        )  # (BLOCK_N, BLOCK_D)
        # input_precision 只对 fp32 输入有意义: ieee=CUDA core, 默认 tf32=tensor core。
        # bf16/fp16 输入时 tl.dot 恒走 tensor core、fp32 累加, 此参数被忽略。
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale  # (BLOCK_M, BLOCK_N)

        # 掩码必须在 exp 之前做：padding 的 key (kv_mask=False) 其 k=0,
        # 若不掩码会以 s=0 混进 softmax；causal 未来位置置 -inf, exp 后为 0
        s = tl.where(kv_mask[None, :], s, float("-inf"))
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))

        # online softmax 更新（见模块 docstring 的公式）
        m_new = tl.maximum(m_i, tl.max(s, axis=1))       # (BLOCK_M,)
        alpha = tl.exp(m_i - m_new)                       # 旧累计值的缩放系数
        p = tl.exp(s - m_new[:, None])                    # 本块未归一化概率
        l_i = alpha * l_i + tl.sum(p, axis=1)
        v = tl.load(
            v_ptr + pid_b * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=kv_mask[:, None] & d_mask[None, :], other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        m_i = m_new

    acc = acc / l_i[:, None]                              # 最终归一化
    lse = m_i + tl.log(l_i)                               # (BLOCK_M,) 真 logsumexp

    tl.store(
        o_ptr + pid_b * stride_ob + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od,
        acc, mask=q_mask[:, None] & d_mask[None, :],
    )
    tl.store(lse_ptr + pid_b * stride_lseb + offs_m * stride_lsen, lse, mask=q_mask)


@triton.jit
def _flash_attn_dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dq_ptr,
    N_Q, N_K, D, scale,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_dob, stride_don, stride_dod,
    stride_dqb, stride_dqn, stride_dqd,
    stride_lseb, stride_lsen,
    stride_deltab, stride_deltan,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """dQ kernel：grid = (ceil(N_Q/BLOCK_M), B)，与 forward kernel 同构。

    每个 program 拥有一个 Q 块（独占 → dq 可直接 store, 无冲突）：
      载入 Q 块与 dO 块 → 沿 KV 块循环：
        S_j = Q·K_j^T * scale → 掩码 → P_j = exp(S_j - lse)（重算, 不用存 N²）
        dP_j = dO·V_j^T, dS_j = P_j ⊙ (dP_j - delta)
        dq += dS_j·K_j * scale
    causal 裁剪与 forward 相同：query 行 i 只需要 key 0..i。
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    q_mask = offs_m < N_Q
    d_mask = offs_d < D

    base = pid_b * stride_qb
    q = tl.load(q_ptr + base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    do_ = tl.load(do_ptr + pid_b * stride_dob + offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod,
                  mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    lse = tl.load(lse_ptr + pid_b * stride_lseb + offs_m * stride_lsen, mask=q_mask, other=0.0)
    delta = tl.load(delta_ptr + pid_b * stride_deltab + offs_m * stride_deltan, mask=q_mask, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    if IS_CAUSAL:
        hi = tl.minimum(N_K, (pid_m + 1) * BLOCK_M)
    else:
        hi = N_K

    for n0 in range(0, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < N_K
        k = tl.load(k_ptr + pid_b * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=kv_mask[:, None] & d_mask[None, :], other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        # 掩码必须在 exp 之前: padding key 与 causal 未来位置置 -inf → P=0
        s = tl.where(kv_mask[None, :], s, float("-inf"))
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))
        p = tl.exp(s - lse[:, None])                       # 重算 P_j, 只活在寄存器里
        v = tl.load(v_ptr + pid_b * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=kv_mask[:, None] & d_mask[None, :], other=0.0)
        dp = tl.dot(do_, tl.trans(v), input_precision="ieee")
        ds = p * (dp - delta[:, None])                     # softmax 反传的 delta 技巧
        acc += tl.dot(ds.to(q.dtype), k, input_precision="ieee")

    acc *= scale
    tl.store(dq_ptr + pid_b * stride_dqb + offs_m[:, None] * stride_dqn + offs_d[None, :] * stride_dqd,
             acc, mask=q_mask[:, None] & d_mask[None, :])


@triton.jit
def _flash_attn_dkdv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, lse_ptr, delta_ptr, dk_ptr, dv_ptr,
    N_Q, N_K, D, scale,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_dob, stride_don, stride_dod,
    stride_dkb, stride_dkn, stride_dkd,
    stride_dvb, stride_dvn, stride_dvd,
    stride_lseb, stride_lsen,
    stride_deltab, stride_deltan,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """dK/dV kernel：grid = (ceil(N_K/BLOCK_N), B)，循环方向转置。

    每个 program 拥有一个 K 块（独占 → dk/dv 可直接 store, 无冲突），
    沿 Q 块循环把"所有 query 行对该 key 块的贡献"累积完：
        S = Q_i·K^T * scale → P = exp(S - lse) → dS = P ⊙ (dP - delta)
        dk += dS^T·Q_i * scale,  dv += P^T·dO_i
    causal 时 key j 只被 query i>=j 触碰 → Q 循环从 key 块所在 Q 块开始,
    对角块内的越界由掩码兜底。越界的 padding query 行 s=-inf → P=0, 不污染累加。
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    kv_mask = offs_n < N_K
    d_mask = offs_d < D

    base = pid_b * stride_kb
    k = tl.load(k_ptr + base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=kv_mask[:, None] & d_mask[None, :], other=0.0)
    v = tl.load(v_ptr + pid_b * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=kv_mask[:, None] & d_mask[None, :], other=0.0)

    dk_acc = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)

    if IS_CAUSAL:
        lo = (pid_n * BLOCK_N) // BLOCK_M * BLOCK_M   # 对齐到 Q 块边界
    else:
        lo = 0

    for m0 in range(lo, N_Q, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        q_mask = offs_m < N_Q
        q = tl.load(q_ptr + pid_b * stride_qb + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                    mask=q_mask[:, None] & d_mask[None, :], other=0.0)
        do_ = tl.load(do_ptr + pid_b * stride_dob + offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod,
                      mask=q_mask[:, None] & d_mask[None, :], other=0.0)
        lse = tl.load(lse_ptr + pid_b * stride_lseb + offs_m * stride_lsen, mask=q_mask, other=0.0)
        delta = tl.load(delta_ptr + pid_b * stride_deltab + offs_m * stride_deltan, mask=q_mask, other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        # padding query 行必须置 -inf：其 q=0 → s=0, 否则 exp(0-lse)≠0 混进 dk/dv
        s = tl.where(q_mask[:, None], s, float("-inf"))
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))
        p = tl.exp(s - lse[:, None])
        dp = tl.dot(do_, tl.trans(v), input_precision="ieee")
        ds = p * (dp - delta[:, None])
        dk_acc += tl.dot(tl.trans(ds).to(k.dtype), q, input_precision="ieee")
        dv_acc += tl.dot(tl.trans(p).to(v.dtype), do_, input_precision="ieee")

    dk_acc *= scale
    tl.store(dk_ptr + pid_b * stride_dkb + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd,
             dk_acc, mask=kv_mask[:, None] & d_mask[None, :])
    tl.store(dv_ptr + pid_b * stride_dvb + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd,
             dv_acc, mask=kv_mask[:, None] & d_mask[None, :])
