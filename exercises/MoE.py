import torch
import torch.nn as nn
import torch.nn.functional as F
from thinc.layers import ragged2list


class Expert_ReLu(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return self.ffn(x)

class Expert_SwiGLU(nn.Module): #一个专家的前向传播，类似ffn，使用SwiGLU激活函数
    def __init__(self,dim):
        super().__init__()
        self.W1 = nn.Linear(dim, dim * 4)
        self.W2 = nn.Linear(dim, dim * 4)
        self.W3 = nn.Linear(dim * 4, dim)
        self.silu=nn.SiLU()
    def forward(self,x):
        return self.W3(self.silu(self.W1(x))*self.W2(x))

class TC_MoE(nn.Module): #选取top_k个专家的输出加权平均
    def __init__(self, dim, num_experts, k):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([Expert_SwiGLU(dim) for _ in range(num_experts)])

    def forward(self, x, tokens=None, verbose=False):  # x:(batch,d_model)
        batch, d_model = x.shape

        gate_scores = F.softmax(self.router(x), dim=-1)  # scores:(batch,num_experts)

        topk_scores, topk_idx = gate_scores.topk(self.k, dim=-1)  # topk_scores:(batch,k),topk_idx:(batch,k)

        out = torch.zeros_like(x)  # out:(batch,d_model)

        for i in range(self.k):
            expert_ids = topk_idx[:, i]  # expert_ids:(batch,)
            expert_weight = topk_scores[:, i]  # expert_weight:(batch,)
            expert_output = torch.zeros_like(x)
            for e_id, expert in enumerate(self.experts):

                mask = (expert_ids == e_id).unsqueeze(1)  # mask:(batch,1) 只保留当前专家的token

                if mask.sum() == 0:
                    continue

                expert_output += expert(x * mask)  # expert_output:(batch,d_model) 加上当前专家的输出

            out += expert_output * expert_weight.unsqueeze(1)  # out:(batch,d_model) 加上一轮输出乘以对应的专家权重

        return out


class EC_MoE(nn.Module): #选取top_k个token的输出加权平均
    def __init__(self,dim,num_experts,k):
        super().__init__()
        self.num_experts=num_experts

        self.k=k
        self.router=nn.Linear(dim,num_experts)
        self.experts=nn.ModuleList([Expert_SwiGLU(dim) for _ in range(num_experts)])

    def forward(self,x,tokens=None,verbose=False):  # x:(batch,d_model)
        batch,d_model=x.shape
        scores=F.softmax(self.router(x).T,dim=-1)  # scores:(num_experts,batch)
        topk_scores,topk_idx=scores.topk(min(self.k,batch),dim=-1) #选取top_k个token的分数和索引 topk_scores:(num_experts,k),topk_idx:(num_experts,k)
        out=torch.zeros_like(x)  # out:(batch,d_model)
        for e in range(self.num_experts):
            token_scores=topk_scores[e]  # token_scores:(k,) 第e个专家对应的top_k个token的分数
            token_idx=topk_idx[e]
            selected_tokens=x[token_idx]  # selected_tokens:(k,d_model) 取出对应的token的输入
            expert_output=self.experts[e](selected_tokens)  # expert_output:(k,d_model)
            weighted_output=expert_output*token_scores.unsqueeze(1) # weighted_output:(k,d_model)

            out[token_idx]+=weighted_output  # out:(batch,d_model) [k,d_model]+=[k,d_model] 对应token的输出加权平均

        return out





if __name__ == "__main__":
    x = torch.randn(32, 64)
    model = TC_MoE(64, 16, 4)
    model1=EC_MoE(64,16,4)
    out = model(x)
    print(out.shape)
    print(out)
    out1=model1(x)
    print(out1.shape)
    print(out1)
