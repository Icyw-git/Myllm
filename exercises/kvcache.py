import torch
import torch.nn as nn
import math
from torch import Tensor # 张量类型
from einops import einsum, rearrange,repeat,reduce
import einops
torch.manual_seed(0)

# 组件：RMSNorm

# 输入：
# x: [B, T, D]，浮点张量
# weight: [D]，可训练参数
# eps: 标量

# 输出：
# y: [B, T, D]

# 计算：
# 沿最后一维计算均方根
# 输出形状与输入相同

# 必须满足：
# 支持 FP32/BF16
# 不修改输入
# 可以反向传播
# weight 梯度形状为 [D]

# 暂不支持：
# 非最后一维归一化
# 分布式张量
# 自定义 CUDA





class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-6):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(d_model)) # 参数注册在模块里,可训练可保存
        self.eps=eps

    def forward(self,input: Tensor) -> Tensor:
        dtype=input.dtype # 记录输入类型,先提升至float32进行计算 之后再转换回输入类型
        x=input.float()
        mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        output=self.weight * x / rms
        return output.to(dtype)

def softmax(input:Tensor):
    x_max=torch.max(input,dim=-1,keepdim=True)[0] #(B,1)
    input_norm=input-x_max #(B,V)
    log_sum=torch.log(torch.sum(torch.exp(input_norm),dim=-1,keepdim=True)) #(B,1)
    log_prob=input_norm-log_sum
    return torch.exp(log_prob)

def cross_entropy(input:Tensor,target:Tensor): #target:one-hot
    prob=softmax(input)
    loss=-torch.sum(torch.log(prob)*target,dim=-1).mean() #batch维进行平均
    return loss

class multihead_attn_with_rope(nn.Module):
    def __init__(self,d_model:int,num_heads:int):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        assert d_model % num_heads==0
        self.head_model=d_model//num_heads
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.w_o=nn.Linear(d_model,d_model)

    def rope(self,x:Tensor,theta:float=10000.0,)->Tensor:
        *lead,seq_len,d_model=x.shape
        
        assert d_model %2==0
        d_k=d_model //2
        xr=rearrange(x,'... t (dk two) -> ... t dk two',two=2)
        range=torch.arange(d_k,device=x.device,dtype=x.dtype)
        ids=torch.arange(seq_len,device=x.device,dtype=x.dtype)
        theta_list=einsum(ids,theta**(-2*range/d_model),'i, j -> i j') # 分母是 head_dim，不是 d_k
        
        x0=xr[...,0]*torch.cos(theta_list)-xr[...,1]*torch.sin(theta_list) #[...,0]是取最后一维的第0个元素，[:,1]是取第二维的第1个
        x1=xr[...,1]*torch.cos(theta_list)+xr[...,0]*torch.sin(theta_list)
        


        return rearrange([x0,x1],'two ... t dk -> ... t (dk two)')







    def forward(self,input:Tensor,causal:bool=True):
        batch_size,seq_len,d_model=input.shape
        q=self.w_q(input)
        k=self.w_k(input)
        v=self.w_v(input)

  


        # 先 view 成 (B,T,H,d)（从最后一维 D 切出 head），再 transpose 到 (B,H,T,d)
        q=rearrange(q,'b t (h d) -> b h t d',h=self.num_heads)
        k=rearrange(k,'b t (h d) -> b h t d',h=self.num_heads)
        v=rearrange(v,'b t (h d) -> b h t d',h=self.num_heads)

        q=self.rope(q)
        k=self.rope(k)



        
        attn=einsum(q,k,'b h t d, b h s d -> b h t s')/math.sqrt(self.head_model)
        if causal:
            mask=torch.triu(torch.ones(seq_len,seq_len,device=input.device,dtype=torch.bool),diagonal=1)
            attn=attn.masked_fill(mask,float('-inf'))
        score=softmax(attn)

        output=einsum(score,v,'b h t s, b h s d -> b h t d')
        output=rearrange(output,'b h t d -> b t (h d)')
        return self.w_o(output)


class SwiGLU(nn.Module):
    def __init__(self,d_in:int,d_ff:int):
        super().__init__()
        self.w1=nn.Linear(d_in,d_ff)
        self.w2=nn.Linear(d_in,d_ff)
        self.w3=nn.Linear(d_ff,d_in)

    def forward(self,input:Tensor)->Tensor:
        x1=self.w1(input)
        x2=self.w2(input)
        swish=x1/(1+torch.exp(-x1))
        return self.w3(swish*x2)


        
        
class TransformerBlock(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int):
        super().__init__()
        self.norm1=RMSNorm(d_model)                        # 两组独立参数,不再共用
        self.attn=multihead_attn_with_rope(d_model,num_heads)
        self.norm2=RMSNorm(d_model)
        self.ffn=SwiGLU(d_model,d_ff)
        

    def forward(self,input:Tensor,causal:bool=True)->Tensor:
        x=input+self.attn(self.norm1(input),causal)  # 残差绕开 norm
        x=x+self.ffn(self.norm2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self,d_model:int,num_heads:int,num_blocks:int,d_ff:int,vocab_size:int):
        super().__init__()
        self.blocks=nn.ModuleList([TransformerBlock(d_model,num_heads,d_ff) for _ in range(num_blocks)])
        self.out=nn.Linear(d_model,vocab_size)
        
    def forward(self,input:Tensor,causal:bool=True)->Tensor:
        for blk in self.blocks:
            input=blk(input,causal)
        return self.out(input)












if __name__ =="__main__":
    input=torch.rand((1,3,6),dtype=torch.bfloat16)
    norm=RMSNorm(6).to(torch.bfloat16)
    print(norm(input))

    x=torch.tensor([[0.2,0.4,0.4]],dtype=torch.float32)
    print(softmax(x))

    y=torch.tensor([[0,1,0]],dtype=torch.float32)
    print(cross_entropy(x,y))

    input=torch.rand((1,3,8),dtype=torch.bfloat16)
    attn=multihead_attn_with_rope(8,2).to(torch.bfloat16)   # 模型必须与输入同 dtype    
    print(attn(input))
    print(attn.rope(input).shape)

    input=torch.rand((4,10,256),dtype=torch.bfloat16)
    out=TransformerLM(256,8,2,1024,1000).to(input.device,input.dtype) # device 和 dtype 都要一致
    out=out(input)

    print(out)


