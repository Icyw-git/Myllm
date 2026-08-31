from __future__ import annotations
import torch
import torch.nn as nn
import torch.distributed as dist


class DDP(nn.Module):
    def __init__(self,module:nn.Module):
        super().__init__()

        self.module=module #这是要并行化的模型，比如模型
        self.world_size=dist.get_world_size() #这是并行化的设备数量

        

        self._pending:list[tuple[nn.Parameter,dist.Work]]=[] #待处理的梯度计算，每个元素是一个元组，包含参数和梯度计算的句柄

        for param in self.module.parameters():
            dist.broadcast(param.data,src=0) #广播参数到所有设备，从设备0开始广播

        seen:set[int]=set()

        for param in self.module.parameters(): # 遍历参数，注册梯度计算钩子，只对需要梯度的参数注册
            if not param.requires_grad or param.data_ptr() in seen:
                continue
            seen.add(param.data_ptr())
            param.register_post_accumulate_grad_hook(self._make_grad_hook(param)) # 注册梯度计算钩子，用于在参数梯度计算完成后进行梯度聚合

    def _make_grad_hook(self,param:nn.Parameter):
        def hook(_:nn.Parameter): 
            if param.grad is None:
                return 
            handle=dist.all_reduce(param.grad,op=dist.ReduceOp.SUM,async_op=True) #支持异步梯度聚合，避免阻塞主进程的执行
            self._pending.append((param,handle)) # 将参数和梯度计算句柄添加到待处理列表中

        return hook
    

    def forward(self,*args,**kwargs): #作用是将输入传递给模型，返回模型的输出，同时在前向传播过程中，会触发梯度计算钩子，将梯度聚合到参数中
        return self.module(*args,**kwargs)


    def finish_gradient_accumulation(self):
        for _,handle in self._pending: #等待所有梯度计算完成，确保所有设备的梯度都已聚合
            handle.wait()



        for param,_ in self._pending: # 对每个参数的梯度进行归一化，确保每个设备的梯度都相同
            param.grad/=self.world_size
        self._pending.clear()


# ===================== 使用示例 =====================
# 运行: .venv/bin/python /root/Myllm/exercises/ddp.py
# 注意: worker 必须定义在模块顶层——mp.spawn 的子进程会重新 import 本文件,
#       藏在 if __name__ == "__main__" 里的函数子进程找不到。
import os
import torch.multiprocessing as mp


def worker(rank: int, world_size: int):
    # --- 1. 环境设置: 每个 rank 一个进程 ---
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # --- 2. 建模型 + DDP 包裹 ---
    # 各 rank 故意用不同 seed 初始化，验证构造时 broadcast 确实同步了权重
    torch.manual_seed(rank)
    model = nn.Linear(4, 2)

    ddp = DDP(model)  # 构造完成后，所有 rank 的权重 == rank0 的权重

    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.1)  # 参数透传，和普通用法一样

    # --- 3. 训练一步: 每个 rank 喂不同的数据 ---
    torch.manual_seed(100 + rank)  # 数据故意不同
    x = torch.randn(8, 4)
    loss = ((ddp(x) - 1) ** 2).mean()
    loss.backward()  # 钩子在这里被自动触发，异步 all-reduce 已在后台发出

    # --- 4. step 前同步梯度 (等待 + 平均) ---
    ddp.finish_gradient_accumulation()
    optimizer.step()

    print(f"rank{rank} 权重 w: {ddp.module.weight.data.flatten()[:2].tolist()} "
          f"梯度均值: {ddp.module.weight.grad.mean().item():.6f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
    # 验证标准: 两个 rank 打印的梯度均值和更新后的权重完全一致
    # (尽管它们各自看到的数据不同、初始权重不同)

    

        
    



