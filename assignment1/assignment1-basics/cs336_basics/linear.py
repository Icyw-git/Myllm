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

def softmax(in_features: Float[Tensor, " ... d_model"],dim: int):
    x_max=torch.max(in_features,dim=dim,keepdim=True)[0]
    x=in_features-x_max
    exp_sum=torch.sum(torch.exp(x),dim=dim,keepdim=True)
    return torch.exp(x)/exp_sum


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
    temp=torch.sqrt(torch.mean(in_features**2,dim=-1,keepdim=True)+eps) #注意keepdim和dim的使用
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
    scores=torch.log(softmax(inputs,dim=-1))
    batch_idx=torch.arange(inputs.shape[0])
    probs=scores[batch_idx,targets]


    return -torch.mean(probs)




