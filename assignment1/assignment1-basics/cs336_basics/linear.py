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
    Linear1=Linear(d_ff,d_model,w1_weight)
    Linear2=Linear(d_model,d_ff,w2_weight)
    Linear3=Linear(d_ff,d_model,w3_weight)
    return Linear2(silu(Linear1(in_features))*Linear3(in_features))

def softmax(in_features: Float[Tensor, " ... d_model"],dim: int): #这是softmax的变体，使用log-sum-exp技巧避免数值溢出
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
    scores=einops.einsum(Q,K," ... queries d_k, ... keys d_k->... queries keys")/math.sqrt(d_k) #注意mask的位置
    if mask is not None:
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

    Q=in_features @ q_proj_weight.T
    K=in_features @ k_proj_weight.T
    V=in_features @v_proj_weight.T



    d_k=d_model // num_heads
    Q=Q.view(batch_size,seq_len,num_heads,d_k).transpose(1,2)
    V=V.view(batch_size,seq_len,num_heads,d_k).transpose(1,2)
    K=K.view(batch_size,seq_len,num_heads,d_k).transpose(1,2)

    scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k)
    mask=torch.triu(torch.ones(seq_len,seq_len),diagonal=1) #生成上三角矩阵，用于遮蔽未来的token
    scores=scores.masked_fill(mask,float("-inf")) #1的位置设为-inf，避免softmax时出现无穷大
    scores=softmax(scores,dim=-1)

    results=torch.matmul(scores,V)
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
    freq_seq=torch.arange(0,d_k,2,dtype=torch.float32)/d_k #这里是为了生成频率序列，步长为2，范围从0到d_k，除以d_k是为了归一化
    rope_theta=1/theta**freq_seq #生成角度序列，1/theta**freq_seq是为了生成不同频率的角度，形状为(d_k//2,)
    angles=token_positions.unsqueeze(-1)*rope_theta #使用unsqueeze(-1)将token_positions的形状从(..., seq_len)变为(..., seq_len, 1)，然后与rope_theta相乘，得到每个token位置对应的角度，形状为(..., seq_len, d_k//2)，便于广播
    cos=angles.cos()
    sin=angles.sin()
    x=in_query_or_key.view(*prefix,seq_len,d_k//2,2)
    rope_x=torch.stack([
        x[...,0]*cos-x[...,1]*sin,
        x[...,0]*sin+x[...,1]*cos],
        dim=-1
    ) #这里是将旋转后的结果堆叠起来，dim=-1表示在最后一个维度上堆叠，得到形状为(..., seq_len, d_k//2, 2)

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
    *prefix,seq_len,d_model=in_features.shape #保存前导维度，seq_len是序列长度，d_model是特征维度
    in_features=in_features.view(-1,seq_len,d_model)
    batch_size=in_features.shape[0]
    
    Q=in_features@q_proj_weight.T
    K=in_features@k_proj_weight.T
    V=in_features@v_proj_weight.T #注意转置

    d_k=d_model // num_heads #必须使用// 不然会返回浮点数，后续的view会报错
    assert d_k==d_model // num_heads
    Q=Q.view(batch_size,seq_len,num_heads,d_k).transpose(1,2)
    K=K.view(batch_size,seq_len,num_heads,d_k).transpose(1,2)
    V=V.view(batch_size,seq_len,num_heads,d_k).transpose(1,2)


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
    rope_Q=rope_Q.view(batch_size,num_heads,seq_len,d_k) #注意这里的view是为了将最后一个维度从2恢复到d_k

    freq_seq=torch.arange(0,d_k,2,dtype=torch.float32)/d_k
    rope_theta=1/theta**freq_seq
    angles=token_positions.unsqueeze(-1)*rope_theta
    cos=angles.cos()
    sin=angles.sin()
    K=K.view(batch_size,num_heads,seq_len,d_k//2,2)

    rope_K=torch.stack([
        K[...,0]*cos-K[...,1]*sin,
        K[...,0]*sin+K[...,1]*cos
    ],dim=-1)
    rope_K=rope_K.view(batch_size,num_heads,seq_len,d_k)



    scores=torch.matmul(rope_Q,rope_K.transpose(-2,-1))/math.sqrt(d_k)
    mask=torch.triu(torch.ones(seq_len,seq_len,dtype=torch.bool),diagonal=1) #diagonal=1表示上三角矩阵的对角线以上部分为True，其他部分为False
    scores=scores.masked_fill(mask,float("-inf"))
    attn=softmax(scores,dim=-1)
    results=torch.matmul(attn,V)
    results=results.transpose(1,2).contiguous().view(batch_size,seq_len,d_model) #注意这里的contiguous()是为了保证内存连续性，避免view报错
    outputs=results@o_proj_weight.T
    outputs=outputs.view(*prefix,seq_len,d_model)
    return outputs



    


    







