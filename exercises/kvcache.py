from accelerate.utils import id_tensor_storage
import torch
import torch.nn as nn
import math
from torch import Tensor # 张量类型
from einops import einsum, rearrange,repeat,reduce
import einops
from transformers.utils.dummy_pt_objects import Idefics2Model
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

    def rope(self,x:Tensor,theta:float=10000.0,past_kv:dict[Tensor,Tensor]=None)->Tensor:
        *lead,seq_len,d_model=x.shape
        
        assert d_model %2==0
        d_k=d_model //2
        xr=rearrange(x,'... t (dk two) -> ... t dk two',two=2)
        range=torch.arange(d_k,device=x.device,dtype=x.dtype)
        if past_kv:
            cache_len=past_kv['k'].shape[2] if 'k' in past_kv else 0
            ids=torch.arange(cache_len,cache_len+seq_len,device=x.device,dtype=x.dtype)
        else:

            ids=torch.arange(seq_len,device=x.device,dtype=x.dtype)
        theta_list=einsum(ids,theta**(-2*range/d_model),'i, j -> i j') # 分母是 head_dim，不是 d_k
        
        x0=xr[...,0]*torch.cos(theta_list)-xr[...,1]*torch.sin(theta_list) #[...,0]是取最后一维的第0个元素，[:,1]是取第二维的第1个
        x1=xr[...,1]*torch.cos(theta_list)+xr[...,0]*torch.sin(theta_list)
        


        return rearrange([x0,x1],'two ... t dk -> ... t (dk two)')




    def forward(self,input:Tensor,causal:bool=True,past_kv:dict[Tensor,Tensor]=None):
        batch_size,seq_len,d_model=input.shape
        q=self.w_q(input)
        k=self.w_k(input)
        v=self.w_v(input)

  


        # 先 view 成 (B,T,H,d)（从最后一维 D 切出 head），再 transpose 到 (B,H,T,d)
        q=rearrange(q,'b t (h d) -> b h t d',h=self.num_heads)
        k=rearrange(k,'b t (h d) -> b h t d',h=self.num_heads)
        v=rearrange(v,'b t (h d) -> b h t d',h=self.num_heads)

        q=self.rope(q,past_kv=past_kv)
        k=self.rope(k,past_kv=past_kv)

        if past_kv:
            k=torch.cat([past_kv['k'],k],dim=2)
            v=torch.cat([past_kv['v'],v],dim=2)





        
        attn=einsum(q,k,'b h t d, b h s d -> b h t s')/math.sqrt(self.head_model)
        if causal:
            k_len=k.shape[2]                                  # = cache_len + seq_len
            q_pos=torch.arange(k_len-seq_len,k_len,device=input.device,dtype=torch.float32)
            k_pos=torch.arange(k_len,device=input.device,dtype=torch.float32)
            mask=q_pos[:,None]<k_pos[None,:]
            attn=attn.masked_fill(mask,float('-inf'))


        score=softmax(attn)

        output=einsum(score,v,'b h t s, b h s d -> b h t d')
        output=rearrange(output,'b h t d -> b t (h d)')
        new_kv={'k':k,'v':v}
        return self.w_o(output),new_kv



class GQA(nn.Module):
    def __init__(self,d_model:int,num_heads:int,num_kv_heads:int):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.num_kv_heads=num_kv_heads  
        assert num_heads % num_kv_heads==0
        self.head_dim=d_model//num_heads
        self.group_size=num_heads//num_kv_heads
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,num_kv_heads*self.head_dim)
        self.w_v=nn.Linear(d_model,num_kv_heads*self.head_dim)
        self.w_o=nn.Linear(d_model,d_model)

    def repeat_kv(self,input:Tensor,n_repeat:int)->Tensor:
        batch,num_kv_heads,seq_len,head_dim=input.shape
        repeat_input=repeat(input,'b h t d -> b (h n_repeat) t d',n_repeat=n_repeat)
        return repeat_input

    def forward(self,input:Tensor,causal:bool=True,past_kv:dict[Tensor,Tensor]=None):
        batch_size,seq_len,d_model=input.shape
        q=self.w_q(input)
        k=self.w_k(input)
        v=self.w_v(input)

        

        q=rearrange(q,'b t (h d) -> b h t d',h=self.num_heads)
        k=rearrange(k,'b t (h d) -> b h t d',h=self.num_kv_heads)
        v=rearrange(v,'b t (h d) -> b h t d',h=self.num_kv_heads)

        if past_kv:
            k=torch.cat([past_kv['k'],k],dim=2)
            v=torch.cat([past_kv['v'],v],dim=2)
        new_kv={'k':k,'v':v}                        # ★ cache 存 H_kv 个(复制前)

        k=self.repeat_kv(k,self.group_size)         # ★ 复制只为算 attention
        v=self.repeat_kv(v,self.group_size)

        attn=einsum(q,k,'b h t d, b h s d -> b h t s')/math.sqrt(self.head_dim)
        if causal:
            k_len=k.shape[2]                                  # = cache_len + seq_len
            q_pos=torch.arange(k_len-seq_len,k_len,device=input.device,dtype=torch.float32)
            k_pos=torch.arange(k_len,device=input.device,dtype=torch.float32)
            mask=q_pos[:,None]<k_pos[None,:]
            attn=attn.masked_fill(mask,float('-inf'))


        score=softmax(attn)

        output=einsum(score,v,'b h t s, b h s d -> b h t d')
        output=rearrange(output,'b h t d -> b t (h d)')
        return self.w_o(output),new_kv





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
        

    def forward(self,input:Tensor,causal:bool=True,past_kv:dict[str,Tensor]=None)->Tensor:
        output,new_kv=self.attn(self.norm1(input),causal,past_kv)
        x=input+output  # 残差绕开 norm
        x=x+self.ffn(self.norm2(x))
        return x,new_kv

class TransformerLM(nn.Module):
    def __init__(self,d_model:int,num_heads:int,num_blocks:int,d_ff:int,vocab_size:int):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,d_model)

        self.blocks=nn.ModuleList([TransformerBlock(d_model,num_heads,d_ff) for _ in range(num_blocks)])
        self.out=nn.Linear(d_model,vocab_size)
        
    def forward(self,ids,causal:bool=True,past_kvs:list[dict[str,Tensor]]=None)->Tensor:
        input=self.embedding(ids)

        new_kvs=[]
        for i,blk in enumerate(self.blocks):
            past=past_kvs[i] if past_kvs else None

            input,new_kv=blk(input,causal,past_kv=past)
            new_kvs.append(new_kv)

        return self.out(input),new_kvs

@torch.no_grad()
def greedy_generate(model,prompt_ids,max_new_tokens,use_cache:bool=True):
    ids=prompt_ids
    if use_cache:
        logits,cache=model(ids,causal=True)
        for _ in range(max_new_tokens):
            next=logits[:,-1].argmax(-1,keepdim=True)
            ids=torch.cat([ids,next],dim=1)
            logits,cache=model(next,causal=True,past_kvs=cache) # 只喂新 token, 历史在 cache 里
    else:
        logits,cache=model(ids,causal=True)
        for _ in range(max_new_tokens):
            next=logits[:,-1].argmax(-1,keepdim=True)
            ids=torch.cat([ids,next],dim=1)
            logits,cache=model(ids,causal=True)

    return ids


def sample(logits:Tensor,temperature:float=1.0,top_p:float=None,top_k:int=None)->Tensor:
    logits=logits/temperature
    if top_k is not None:
        k=min(top_k,logits.size(-1))
        kth=logits.topk(k,dim=-1).values[...,-1,None]
        mask=(logits<kth)
        logits=logits.masked_fill(mask,float('-inf'))
        

    if top_p is not None:
        sl,si=logits.sort(dim=-1,descending=True)          # 排序的是 logits
        pr=softmax(sl)
        cum=pr.cumsum(dim=-1)
        sl=sl.masked_fill((cum-pr)>top_p,float('-inf'))    # 减 pr, 保证至少保留 1 个
        logits=sl.scatter(-1,index=si,src=sl)                        # 还原的是 logits, 不是概率,重排回原来的顺序

    next=torch.multinomial(softmax(logits),num_samples=1)
    return next


        





        






@torch.no_grad()
def generate(model,prompt_ids,max_new_tokens,use_cache:bool=True,temperature:float=1.0,top_p:float=None,top_k:int=None):
    ids=prompt_ids
    if use_cache:
        logits,cache=model(ids,causal=True)                         # prefill 整段
        for _ in range(max_new_tokens):
            next=sample(logits[:,-1],temperature,top_p,top_k)       # ① 取最后位置 → (B,V)
            ids=torch.cat([ids,next],dim=1)                         # ② 沿序列维拼接
            logits,cache=model(next,causal=True,past_kvs=cache)     # ③ 只喂新 token
    else:
        for _ in range(max_new_tokens):
            logits,_=model(ids,causal=True)                         # ④ 不传 cache
            next=sample(logits[:,-1],temperature,top_p,top_k)
            ids=torch.cat([ids,next],dim=1)
    return ids



    









        















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

    ids=torch.randint(0,1000,(4,10))          # (B,T) long ← token ids, 不是 float
    lm=TransformerLM(256,8,2,1024,1000)
    logits,new_kvs=lm(ids)

    print(logits.shape)
    print(new_kvs[0]['k'].shape)

    model=TransformerLM(256,8,4,1024,1000)
    prompt_ids=torch.tensor([[1,2,3,4,5]],dtype=torch.long)
    generated_ids=greedy_generate(model,prompt_ids,10,use_cache=False)
    print(generated_ids)

    ids=generate(model,prompt_ids,10,top_p=0.7)
    print(ids)

    


