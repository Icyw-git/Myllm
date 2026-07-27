from collections.abc import Iterable
from typing import IO, Any, BinaryIO,Tuple

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import numpy as np
import os



class Linear(nn.Module):
    def __init__(self,in_dim:int,out_dim:int,weights:Float[Tensor, " d_out d_in"]):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.weights=weights
    def forward(self,in_features:Float[Tensor, " ... d_in"]):
        return einops.einsum(in_features,self.weights,"... d_in,d_out d_in->... d_out")

def embedding(vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],)-> Float[Tensor, " ... d_model"]:

    embeddings=torch.stack([weights[token_id] for token_id in token_ids],dim=0)
    return embeddings

def silu(in_features: Float[Tensor, " ... d_model"]):
    return in_features*torch.sigmoid(in_features) #逐元素相乘，注意sigmoid的使用

def swiglu(d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],)-> Float[Tensor, " ... d_model"]:
    # 错过：搞混 w1/w2/w3 角色。讲义：SiLU(xW1) * (xW3)，再乘 W2 投回 d_model。
    # w1/w3: (d_ff, d_model)；w2: (d_model, d_ff)。
    Linear1=Linear(d_ff,d_model,w1_weight)
    Linear2=Linear(d_model,d_ff,w2_weight)
    Linear3=Linear(d_ff,d_model,w3_weight)
    return Linear2(silu(Linear1(in_features))*Linear3(in_features))

def softmax(in_features: Float[Tensor, " ... d_model"],dim: int): #这是softmax的变体，使用log-sum-exp技巧避免数值溢出
    # 错过：sum/max 不加 keepdim=True 时最后一维被挤掉，和原 tensor 广播会错。
    x_max=torch.max(in_features,dim=dim,keepdim=True)[0]
    x=in_features-x_max
    log_sum=torch.log(torch.sum(torch.exp(x),dim=dim,keepdim=True)) #写的时候注意dim和keepdim的使用,作用是保持维度不变，避免广播错误
    logp=x-log_sum
    return torch.exp(logp)


def scaled_dot_product_attention(Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,)-> Float[Tensor, " ... queries d_v"]:
    # 错过：函数体里留半截代码（如 scores=enin）会让整个文件语法错误，
    # import linear 失败，连 softmax 测试都会挂。没写完时先 raise。

    d_k=Q.shape[-1]
    # 注意mask的位置：必须在softmax之前把非法位置设成 -inf
    scores=einops.einsum(Q,K," ... queries d_k, ... keys d_k->... queries keys")/math.sqrt(d_k)
    if mask is not None:
        # 错过：写成 masked_fill(mask, ...) 时，True=可见会把该看的位置盖掉；
        # 讲义/测试里 mask True 表示允许，所以要用 ~mask 盖掉禁止位置。
        scores=scores.masked_fill(~mask,float("-inf"))
    scores=softmax(scores,dim=-1)
    return einops.einsum(scores,V,"... queries keys, ... keys d_v->... queries d_v")

def rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    temp=torch.sqrt(torch.mean(in_features**2,dim=-1,keepdim=True)+eps) #注意keepdim和dim的使用,目的是保持维度不变，避免广播错误，eps是为了避免分母为0
    return in_features/temp*weights

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    # 错过：max_start=len-context_length 且 randint(0, max_start+1) 会抽到 start=len-cl。
    # 此时 x 长 cl、y=dataset[start+1:start+cl+1] 只有 cl-1 → torch.tensor 报
    # expected sequence of length 6 at dim 1 (got 7)。
    # 正确：窗口要 cl+1 个 token；合法 start 是 0..len-cl-1，
    # 即 max_start=len-cl，randint 上界用 max_start（不含）。
    max_start=len(dataset)-context_length
    starts=np.random.randint(0,max_start,size=(batch_size,))
    inputs=[]
    outputs=[]
    for i,start in enumerate(starts):
        inputs.append(dataset[start:start+context_length])
        outputs.append(dataset[start+1:start+context_length+1])
    # 错过：torch.tensor(list_of_ndarrays) 又慢又在长度不一致时难读。
    # 正确：先 np.stack(..., axis=0) 得到 (B, cl)，再 from_numpy。
    inputs=np.stack(inputs,axis=0)
    outputs=np.stack(outputs,axis=0) # stack 在最前新建一维：(B, context_length)
    return torch.from_numpy(inputs).to(device),torch.from_numpy(outputs).to(device)
    
def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):
    # 错过：先 softmax 再 log —— logits 很大时 softmax 变 inf，log 后全坏。
    # 正确：用 log-sum-exp（先减 max）直接算 log_softmax。
    x_max=torch.max(inputs,dim=-1,keepdim=True)[0] #注意keepdim和dim的使用
    x=inputs-x_max
    log_sum=torch.log(torch.sum(torch.exp(x),dim=-1,keepdim=True)) #注意keepdim和dim的使用
    logp=x-log_sum
    prob=logp[torch.arange(targets.shape[0]),targets] #注意targets的使用
    return -torch.mean(prob)

def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    *prefix,seq_len,d_model=in_features.shape

    seq_len=in_features.shape[-2]
    in_features=in_features.view(-1,seq_len,d_model)
    batch_size=in_features.shape[0]

    # 权重 (d_out, d_in)，和 Linear 一样用 x @ W.T
    Q=in_features @ q_proj_weight.T
    K=in_features @ k_proj_weight.T
    V=in_features @v_proj_weight.T

    # 错过：/ 得到 float，view 要 int，必须用 //
    d_k=d_model // num_heads
    # view 成 (batch, seq, heads, d_k) 再 transpose 成 (batch, heads, seq, d_k)
    Q=Q.view(batch_size,seq_len,num_heads,d_k).transpose(1,2)
    V=V.view(batch_size,seq_len,num_heads,d_k).transpose(1,2)
    K=K.view(batch_size,seq_len,num_heads,d_k).transpose(1,2)

    scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k)
    mask=torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device),
        diagonal=1,
    )  # 生成上三角矩阵，用于遮蔽未来的token
    scores=scores.masked_fill(mask,float("-inf")) #1的位置设为-inf，避免softmax时出现无穷大
    scores=softmax(scores,dim=-1)

    results=torch.matmul(scores,V)
    # 错过：transpose 后直接 view 可能因内存不连续报错，要先 contiguous()
    results=results.transpose(1,2).contiguous().view(batch_size,seq_len,d_model)

    outputs=torch.matmul(results,o_proj_weight.T)
    outputs=outputs.view(*prefix,seq_len,d_model)
    return outputs


def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    assert d_k %2==0
    *prefix,seq_len,d_k=in_query_or_key.shape
    # 错过：写成 arange(0,d_k,2) 后直接 1/theta**freq_seq，少了 /d_k，
    # 变成 θ^{-0},θ^{-2},θ^{-4}...；pos0 和第一对碰巧还能对上，后面全炸。
    # 正确：讲义 θ^{-2i/d}，即 (0,2,4,...)/d_k 再放进指数。
    freq_seq=torch.arange(0,d_k,2,dtype=torch.float32)/d_k
    rope_theta=1/theta**freq_seq # (d_k//2,) 每对维度一个频率
    # angles 广播（PyTorch 从右边对齐）：
    #   token_positions: (..., S)
    #   unsqueeze(-1)  → (..., S, 1)     # 给频率维留位置
    #   rope_theta     → (d_k//2,)
    #   (..., S, 1) * (d_k//2,) → (..., S, d_k//2)
    # 含义：angles[..., s, i] = position[..., s] * ω[i]
    angles=token_positions.unsqueeze(-1)*rope_theta
    cos=angles.cos()
    sin=angles.sin()
    # 成对 (x0,x1),(x2,x3),... 做二维旋转；x 与 cos 都是 (..., S, d_k//2) 可直接乘
    x=in_query_or_key.view(*prefix,seq_len,d_k//2,2)
    rope_x=torch.stack([
        x[...,0]*cos-x[...,1]*sin,
        x[...,0]*sin+x[...,1]*cos],
        dim=-1
    )

    rope_x=rope_x.view(*prefix,seq_len,d_k)

    return rope_x

def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    *prefix,seq_len,d_model=in_features.shape #保存前导维度，后面再 view 回去
    in_features=in_features.view(-1,seq_len,d_model)
    batch_size=in_features.shape[0]

    # 错过：曾对整段 in_features 先 RoPE 再 Q/K/V；RoPE 应加在拆 head 后的 Q、K 上，V 不加。
    Q=in_features@q_proj_weight.T
    K=in_features@k_proj_weight.T
    V=in_features@v_proj_weight.T #注意转置；权重约定是 (d_out, d_in)，和 Linear 一样用 @ W.T

    # 错过：d_k=d_model/num_heads 得到 float（如 16.0），assert 16.0==16 能过，但 view 报 TypeError。
    # 正确：必须用 // 得到 int。
    d_k=d_model // num_heads
    Q=Q.view(batch_size,seq_len,num_heads,d_k).transpose(1,2)
    K=K.view(batch_size,seq_len,num_heads,d_k).transpose(1,2)
    V=V.view(batch_size,seq_len,num_heads,d_k).transpose(1,2)

    # 对每个 head 的 d_k 维做 RoPE（频率公式同 rope()，别忘了 /d_k）
    #
    # --- angles 怎么对齐（同 rope）---
    #   positions (B,S) → unsqueeze(-1) → (B,S,1)
    #   rope_theta (d_k/2,) 从右边广播
    #   (B,S,1)*(d_k/2,) → (B,S,d_k/2) = angles
    #
    # --- 乘到 Q/K 时的第二步广播（这里容易错）---
    #   拆 head 后 Q[...,0] 是 (B, H, S, d_k/2)
    #   若直接用 cos=(B,S,d_k/2)，从右边对齐会变成：
    #        Q:  B, H, S, d_k/2
    #       cos:    B, S, d_k/2
    #   → dim1 拿 H 去对 B。测试里常 B==H==4 碰巧过；训练 B=8,H=16 就炸：
    #     size a (16) must match b (8) at non-singleton dimension 1
    #   正确：cos/sin 再 unsqueeze(1) → (B, 1, S, d_k/2)，在 head 维广播。
    freq_seq=torch.arange(0,d_k,2,dtype=in_features.dtype,device=in_features.device)/d_k
    rope_theta=1/theta**freq_seq
    angles=token_positions.to(in_features.device).unsqueeze(-1)*rope_theta  # (B,S,d_k/2)
    cos=angles.cos().unsqueeze(1)  # (B,1,S,d_k/2)
    sin=angles.sin().unsqueeze(1)
    Q=Q.view(batch_size,num_heads,seq_len,d_k//2,2)
    rope_Q=torch.stack([
        Q[...,0]*cos-Q[...,1]*sin,
        Q[...,0]*sin+Q[...,1]*cos
    ],dim=-1)
    rope_Q=rope_Q.view(batch_size,num_heads,seq_len,d_k)

    K=K.view(batch_size,num_heads,seq_len,d_k//2,2)
    rope_K=torch.stack([
        K[...,0]*cos-K[...,1]*sin,
        K[...,0]*sin+K[...,1]*cos
    ],dim=-1)
    rope_K=rope_K.view(batch_size,num_heads,seq_len,d_k)

    scores=torch.matmul(rope_Q,rope_K.transpose(-2,-1))/math.sqrt(d_k)
    # diagonal=1：严格上三角为 True，盖掉未来 token
    mask=torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device),
        diagonal=1,
    )
    scores=scores.masked_fill(mask,float("-inf"))
    attn=softmax(scores,dim=-1)
    results=torch.matmul(attn,V)
    # transpose 后内存可能不连续，view 前要 contiguous()
    results=results.transpose(1,2).contiguous().view(batch_size,seq_len,d_model)
    outputs=results@o_proj_weight.T
    outputs=outputs.view(*prefix,seq_len,d_model)
    return outputs


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """

    batch_size,seq_len,d_model=in_features.shape
    # adapter 没给 positions；完整序列用 0..seq_len-1，扩成 (batch, seq)
    token_positions=torch.arange(seq_len,dtype=torch.long,device=in_features.device).unsqueeze(0).expand(batch_size,-1)
    # pre-norm：先 RMSNorm，再子层；残差加在子层外面，不是塞进 attn/ffn 内部
    #rmsnorm
    x=rmsnorm(d_model,1e-5,weights["ln1.weight"],in_features)
    #多头
    x=multihead_self_attention_with_rope(d_model,num_heads,max_seq_len,theta,weights["attn.q_proj.weight"],weights["attn.k_proj.weight"],weights["attn.v_proj.weight"],weights["attn.output_proj.weight"],x,token_positions)
    out1=x+in_features #残差：attn 输出 + 进 ln1 之前的原 x
    in2=out1

    #rmsnorm
    x=rmsnorm(d_model,1e-5,weights["ln2.weight"],in2)
    #swiglu
    x=swiglu(d_model,d_ff,weights["ffn.w1.weight"],weights["ffn.w2.weight"],weights["ffn.w3.weight"],x)
    out2=in2+x #残差：ffn 输出 + 进 ln2 之前的 in2
    return out2
    

def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """

    # in_indices: (batch, seq) → embedding 后 (batch, seq, d_model)
    x=weights["token_embeddings.weight"][in_indices]
    for i in range(num_layers):
        # 错过1：写成 weights["layers.{i}.attn..."] —— 普通字符串不会插值，
        # KeyError: 'layers.{i}.attn.q_proj.weight'。要用 f"layers.{i}...."。
        # 错过2：把 q/k/v/... 十几个张量当位置参数塞进 transformer_block，
        # 但签名是 (..., weights: dict, in_features)，会 TypeError（7 vs 15）。
        # 正确：剥掉 layers.{i}. 前缀，拼成和单 block 一样的短键 dict 再传入。
        prefix=f"layers.{i}."
        block_weights={k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
        x=transformer_block(d_model,num_heads,d_ff,context_length,rope_theta,block_weights,x)
    x=rmsnorm(d_model,1e-5,weights["ln_final.weight"],x)
    # 输出是未归一化 logits，不要再 softmax
    x=x@weights["lm_head.weight"].T
    return x

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads=[] # 存储所有参数的梯度
    for p in parameters:
        if p.grad is not None:
            grads.append(p.grad.view(-1)) # 将梯度展平成一维

    if grads is None: # 没有梯度，直接返回
        l2=0.0


    grads=torch.cat(grads)
    eps=1e-5
    l2=torch.sqrt(torch.sum(grads**2)).item() #计算l2范数，注意返回值是标量，不是张量
    if l2>max_l2_norm:
        scale=max_l2_norm/(l2+eps) #计算缩放因子，避免除以0
        for p in parameters: # 对每个参数，按比例缩放梯度
            if p.grad is not None:
                p.grad.mul_(scale) # 缩放梯度




class AdamW(torch.optim.Optimizer):
    def __init__(self,parameters: Iterable[torch.nn.Parameter],lr: float,betas: Tuple[float, float],eps: float,weight_decay: float):
        defaults={
            "lr":lr,
            "betas":betas,
            "eps":eps,
            "weight_decay":weight_decay,
            "step":0
        }
        super().__init__(parameters,defaults)

    def step(self,closure=None):
        with torch.no_grad():
            loss=None #损失值，这里不用计算损失函数，因为AdamW的step方法会自动计算损失函数
            if closure is not None: #closure作用是计算损失函数，并返回损失值
                loss=closure()
            for group in self.param_groups: #遍历所有参数组，每个参数组包含一个学习率和两个动量参数，为什么要「组」？
                #同一优化器里，不同参数可以有不同学习率 / weight decay，所以需要「组」来管理这些参数
                lr=group["lr"] #学习率
                beta1,beta2=group["betas"] #两个动量参数
                eps=group["eps"] #防止分母为0的常数
                weight_decay=group["weight_decay"] #权重衰减系数
                step=group["step"] #步数
                for p in group["params"]: #遍历所有参数，这里的参数和param_group的params是同一个参数
                    if p.grad is not None:
                        grad=p.grad.data #data属性是张量的原始数据，grad是梯度，现在一般不建议使用操作data,直接使用no_grad()上下文管理器防止梯度图错误
                        state=self.state[p] #state是字典，存储了参数的动量和平方动量
                        if len(state)==0:
                            state['step']=0
                            state["exp_avg"]=torch.zeros_like(p.data)
                            state["exp_avg_sq"]=torch.zeros_like(p.data)
                        state['step']+=1
                        t=state['step']
                        exp_avg,exp_avg_sq=state["exp_avg"],state["exp_avg_sq"]
                        exp_avg=exp_avg*beta1+(1-beta1)*grad
                        exp_avg_sq=exp_avg_sq*beta2+(1-beta2)*(grad**2)
                        bias_correction1=1-beta1**t
                        bias_correction2=1-beta2**t
                        denom=torch.sqrt(exp_avg_sq/bias_correction2)+eps
                        step_size=lr*exp_avg/bias_correction1/denom
                        if weight_decay!=0:
                            p.data=p.data-step_size-lr*weight_decay*p.data
                        else:
                            p.data=p.data-step_size
                        state["exp_avg"]=exp_avg
                        state["exp_avg_sq"]=exp_avg_sq
            return loss



def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW
    
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        lr=max_learning_rate*(it/warmup_iters)
    elif warmup_iters<=it <=cosine_cycle_iters:
        progress=(it-warmup_iters)/(cosine_cycle_iters-warmup_iters)

        lr=min_learning_rate+0.5*(max_learning_rate-min_learning_rate)*(1+math.cos(math.pi*progress))
    else:
        lr=min_learning_rate
    return lr
    
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    dict={
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'iteration':iteration,
    }
    torch.save(dict,out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    dict=torch.load(src)
    model.load_state_dict(dict['model_state_dict'])
    optimizer.load_state_dict(dict['optimizer_state_dict'])

    return dict['iteration']



    



    







