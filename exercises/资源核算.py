from jaxtyping import Float
import torch
import torch.nn as nn
import numpy as np
import psutil,os
def estimate_training_time(num_samples,num_parameters,num_gpus,tflops_per_gpu,mfu:float=0.5):
    flops=6*num_samples*num_parameters
    flops_per_second=num_gpus*tflops_per_gpu*mfu*1e12

    time=flops / flops_per_second /86400
    return time

class Linear(nn.Module):
    def __init__(self,input_dim:int,output_dim:int):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(input_dim,output_dim)/np.sqrt(input_dim))

    def forward(self,x:torch.Tensor):
        return torch.matmul(x,self.weight)

class Cruncher(nn.Module):
    def __init__(self,num_layers:int,dim:int):
        super().__init__()
        self.layers=nn.ModuleList([Linear(dim,dim) for i  in range(num_layers)])
        self.out=Linear(dim,1)

    def forward(self,x:torch.Tensor):
        for layer in self.layers:
            x=layer(x)

        return self.out(x)

def estimate_memory(model:nn.Module,dtype_byte:int=4):
    num_param=sum(p.numel() for p in model.parameters())
    memory=4*dtype_byte*num_param
    return {
        'num_param':num_param,
        'memory_byte':f'{memory}byte',
        'memory_mb':f'{memory/1024**2}MB'

    }



if __name__ =='__main__':
    days=estimate_training_time(15e12,70e9,1024,989.5)
    print(days)

    model=Cruncher(3,5)
    param_sizes=[]
    for name,param in model.state_dict().items():
        param_sizes.append((name,param.numel()))
        print((name,param))

    print(param_sizes)

    x=torch.randn(4,5)
    y=model(x)
    print(y.size())
    result=estimate_memory(model)
    print(result)

    process=psutil.Process(os.getpid())
    print(f'内存占用：{process.memory_info().rss /1024**2:.2f}MB')

