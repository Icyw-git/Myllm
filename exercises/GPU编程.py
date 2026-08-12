import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
import time
import math


class MLP(nn.Module):
    def __init__(self,dim:int,num_layers:int):
        super().__init__()
        self.layers=nn.ModuleList([nn.Linear(dim,dim) for _ in range(num_layers)])
    def forward(self,x:torch.Tensor):
        for layer in self.layers:
            x=layer(x)
            x=F.relu(x)
        return x

def run_mlp(dim:int,num_layers:int,batch_size:int,num_steps:int)->Callable:
    model=MLP(dim,num_layers)
    x=torch.randn(batch_size,dim)
    def run():
        for step in range(num_steps):
            y=model(x).mean()
            y.backward()
    return run


def benchmark(description:str,run:Callable,num_warmups:int=1,num_trials:int=3):
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times:list[float]=[]
    for trial in range(num_trials):
        start_time=time.time()
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time=time.time()
        times.append((end_time-start_time)*1000)
    mean_time=sum(times)/len(times)
    return mean_time

def run_operation2(dim:int,operation:Callable):
    x=torch.randn(dim,dim,device='cuda' if torch.cuda.is_available() else 'cpu')
    y=torch.randn(dim,dim,device='cuda' if torch.cuda.is_available() else 'cpu')
    def run():
        z=operation(x,y)

    return run


def benchmarking():
    # 测试我们的MLP

    dim = 256  # @inspect dim
    num_layers = 4  # @inspect num_layers
    batch_size = 256  # @inspect batch_size
    num_steps = 2  # @inspect num_steps
    mlp_base = benchmark("run_mlp", run_mlp(dim=dim, num_layers=num_layers, batch_size=batch_size,
                                            num_steps=num_steps))  # @inspect mlp_base

    # 以下是基础扩展测试

    # 对步数进行缩放
    step_results = []

    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x num_steps)",
                           run_mlp(dim=dim, num_layers=num_layers,
                                   batch_size=batch_size,
                                   num_steps=scale * num_steps))  # @inspect result, @inspect scale, @inspect num_steps
        step_results.append((scale, result))  # @inspect step_results

    # 增加层数
    layer_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x num_layers)",
                           run_mlp(dim=dim, num_layers=scale * num_layers,
                                   batch_size=batch_size,
                                   num_steps=num_steps))  # @inspect result, @inspect scale, @inspect num_layers, @inspect num_steps
        layer_results.append((scale, result))  # @inspect layer_results

    # 增加批次大小
    batch_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x batch_size)",
                           run_mlp(dim=dim, num_layers=num_layers,
                                   batch_size=scale * batch_size,
                                   num_steps=num_steps))  # @inspect result, @inspect scale, @inspect num_layers, @inspect num_steps
        batch_results.append((scale, result))  # @inspect batch_results

    # 对维度进行缩放
    dim_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x dim)",
                           run_mlp(dim=scale * dim, num_layers=num_layers,
                                   batch_size=batch_size,
                                   num_steps=num_steps))  # @inspect result, @inspect scale, @inspect num_layers, @inspect num_steps
        dim_results.append((scale, result))  # @inspect dim_results


if __name__ == "__main__":
    print(benchmark('sleep',lambda :time.sleep(50/1000)))
    if torch.cuda.is_available():
        dims=[1024,2048,4096,8192]
    else:
        dims=[128,256,512,1024]

    matmul_results=[]
    for dim in dims:
        result=benchmark(f'matmul {dim}x{dim}',run_operation2(dim,torch.matmul))
        matmul_results.append((dim,result))
    print(matmul_results)