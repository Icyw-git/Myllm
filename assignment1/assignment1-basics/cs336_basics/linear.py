from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
import numpy as np


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
    max_start=len(dataset)-context_length
    starts=np.random.randint(0,max_start+1,size=(batch_size,))
    inputs=[]
    outputs=[]
    for i,start in enumerate(starts):
        inputs.append(dataset[start:start+context_length])
        outputs.append(dataset[start+1:start+context_length+1])
    return torch.tensor(inputs,dtype=torch.long,device=device),torch.tensor(outputs,dtype=torch.long,device=device)
    
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
    mask=torch.triu(torch.ones(seq_len,seq_len),diagonal=1) #生成上三角矩阵，用于遮蔽未来的token
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
    rope_theta=1/theta**freq_seq #形状 (d_k//2,)
    # unsqueeze(-1)：(..., seq_len) -> (..., seq_len, 1)，和 rope_theta 广播成 (..., seq_len, d_k//2)
    angles=token_positions.unsqueeze(-1)*rope_theta
    cos=angles.cos()
    sin=angles.sin()
    # 成对 (x0,x1),(x2,x3),... 做二维旋转
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
    freq_seq=torch.arange(0,d_k,2,dtype=torch.float32)/d_k
    rope_theta=1/theta**freq_seq
    angles=token_positions.unsqueeze(-1)*rope_theta
    cos=angles.cos()
    sin=angles.sin()
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
    mask=torch.triu(torch.ones(seq_len,seq_len,dtype=torch.bool),diagonal=1)
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



    







