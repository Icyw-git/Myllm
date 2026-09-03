"""End-to-end benchmark of the basics Transformer model.

指标说明:
- CPU 时间: timeit 墙钟时间(含 kernel launch 等开销)
- GPU 时间: torch.cuda.Event 计时(纯 GPU 执行时间, 更接近理论值)
- 计算强度: 每步 FLOPs 估算 / 每步显存流量估算 (FLOP/Byte), 对照 roofline
- 显存占用: PyTorch allocator 的峰值 allocated / reserved
"""
import argparse
import statistics
import timeit
from contextlib import nullcontext

import torch

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

# Table 1 from the assignment handout; vocab/context are the defaults
# (vocab=10000, batch=4, context=512) unless overridden on the CLI.
MODEL_SIZES = {
    'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12, 'vocab_size': 10000, 'context_length': 512},
    'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16, 'vocab_size': 10000, 'context_length': 512},
    'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20, 'vocab_size': 10000, 'context_length': 512},
    'xl': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32, 'vocab_size': 10000, 'context_length': 512},
    '10b': {'d_model': 4608, 'd_ff': 12288, 'num_layers': 50, 'num_heads': 36, 'vocab_size': 10000, 'context_length': 512},
}


def build_model(args) -> BasicsTransformerLM:
    config = dict(MODEL_SIZES[args.model_size])
    if args.context_length is not None:
        config['context_length'] = args.context_length
    return BasicsTransformerLM(**config)


def count_parameters(model: torch.nn.Module) -> int:
    """总参数量(含 embedding)。"""
    return sum(p.numel() for p in model.parameters())


def estimate_flops_per_step(
    n_params: int, num_layers: int, d_model: int,
    batch_size: int, seq_len: int, mode: str,
) -> float:
    """估算单步 FLOPs (近似值, 对照 handout 3.1 节)。

    - 矩阵乘主导项: 2 * N * D (D = batch*seq); 前向 1 份, 反向再 +2 份
      (反向 ≈ 2x 前向: dW 和 dX 各约一份)
    - attention 二次项: 每层 QK^T 和 AV 各 2*B*L^2*d, 前向共 4*B*L^2*d/层
    - 'forward' 模式 1x, 'forward-backward'/'full' 模式 3x
    """
    tokens = batch_size * seq_len
    forward_flops = 2 * n_params * tokens + 4 * num_layers * batch_size * seq_len * seq_len * d_model
    multiplier = 1 if mode == 'forward' else 3
    return forward_flops * multiplier


def estimate_bytes_per_step(
    n_params: int, num_layers: int, batch_size: int, seq_len: int,
    d_model: int, mixed_precision: bool, mode: str, attention: str = 'naive',
) -> float:
    """估算单步显存流量 (含激活值, 对照 handout 3.1 节)。

    权重侧 (与之前一致):
    - fp32 参数 4B / bf16 2B (注意: full 模式下 optimizer 状态仍 fp32, 此处略低估)
    - forward 读 1 遍; backward 读权重+写梯度共 2 遍; full 加 optimizer 读写 ≈ 4 遍
    激活侧 (新增; 保守估计, 实际张量接触次数更多, 故 AI 仍偏高):
    - 每层隐状态张量约 12 次 B*L*d 的读/写 (qkv/rope/ffn 中间量)
    - attention 中间矩阵: 朴素实现 S、P 各"写+读"一次 = 16*B*L^2 字节;
      flash 不物化 N^2 矩阵, 该项为 0 —— 这正是两种实现的差异点
    - backward 需重读 saved activations, 记 2 倍
    """
    bytes_per_param = 2 if mixed_precision else 4
    passes = {'forward': 1, 'forward-backward': 2, 'full': 4}[mode]
    bytes_moved = n_params * bytes_per_param * passes

    act_bpe = 2 if mixed_precision else 4
    hidden = 12 * batch_size * seq_len * d_model * act_bpe
    attn = 0 if attention.startswith('flash') else 16 * batch_size * seq_len * seq_len * act_bpe
    mode_mult = 1 if mode == 'forward' else 2
    bytes_moved += num_layers * (hidden + attn) * mode_mult
    return bytes_moved


def apply_attention_patch(attention: str) -> None:
    """把 A1 模型的朴素 attention 替换为我们的 FlashAttention 实现。

    A1 的 CausalMultiHeadSelfAttention.forward 以
    ``scaled_dot_product_attention(Q=..., K=..., V=..., mask=causal_mask)``
    调用模块级函数; flash 的接口是 ``apply(Q, K, V, is_causal)`` 且内部
    自带 1/sqrt(d_k) 缩放, 故两者可以直接互换 (mask 恒为 causal)。
    """
    if attention == 'naive':
        return

    def _sdpa(Q, K, V, mask=None):
        is_causal = mask is not None
        # flash 实现只支持 3 维 (B, N, D): 把前导维 (batch*heads) 展平成 B,
        # 输出再还原。self-attention 中 Q/K/V 的 seq 长度相同。
        # .contiguous(): rearrange/RoPE 产生的非连续张量无法直接喂给 triton kernel。
        *lead, seq, d_k = Q.shape
        Qf = Q.reshape(-1, seq, d_k).contiguous()
        Kf = K.reshape(-1, seq, d_k).contiguous()
        Vf = V.reshape(-1, seq, d_k).contiguous()
        if attention == 'flash-triton':
            if not torch.cuda.is_available():
                raise RuntimeError('flash-triton 需要 CUDA')
            from cs336_systems.attention import FlashAttentionTritonAutogradFunction
            out = FlashAttentionTritonAutogradFunction.apply(Qf, Kf, Vf, is_causal)
        else:
            from cs336_systems.attention import FlashAttentionPytorchAutogradFunction
            out = FlashAttentionPytorchAutogradFunction.apply(Qf, Kf, Vf, is_causal)
        return out.reshape(*lead, seq, d_k)

    basics_model.scaled_dot_product_attention = _sdpa


def time_steps(
    model, optimizer, x, targets, mode: str, warmup: int, n: int,
    mixed_precision: bool = False,
) -> tuple[list[float], list[float]]:
    """跑 warmup + n 个计步 step。

    返回 (cpu_times, gpu_times), 单位秒。GPU 计时用 cuda.Event;
    若无 CUDA, gpu_times 为空列表。
    """
    use_cuda = torch.cuda.is_available()
    ctx = (torch.autocast('cuda', dtype=torch.bfloat16)
           if mixed_precision and use_cuda else nullcontext())

    def step():
        with ctx:
            if mode == 'forward':
                with torch.no_grad():
                    model(x)
            elif mode == 'forward-backward':
                logits = model(x)
                loss = cross_entropy(logits, targets)
                loss.backward()
            elif mode == 'full':
                optimizer.zero_grad()
                logits = model(x)
                loss = cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
            else:
                raise ValueError(f"Unknown mode: {mode}")
        # 约定: 每步后同步, 保证计时不含上一步的排队 kernel
        if use_cuda:
            torch.cuda.synchronize()

    gpu_events = use_cuda  # 是否启用 Event 计时

    for _ in range(warmup):
        step()

    cpu_times, gpu_times = [], []
    for _ in range(n):
        if gpu_events:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()

        t0 = timeit.default_timer()
        step()
        cpu_times.append(timeit.default_timer() - t0)

        if gpu_events:
            end_ev.record()
            torch.cuda.synchronize()  # 确保 end_ev 已完成再取时间
            gpu_times.append(start_ev.elapsed_time(end_ev) / 1000.0)  # ms -> s

    return cpu_times, gpu_times


def report(
    times_cpu: list[float], times_gpu: list[float], args, device: str,
    n_params: int, num_layers: int, d_model: int,
) -> None:
    """打印完整指标报告。"""
    context_length = args.context_length or MODEL_SIZES[args.model_size]['context_length']
    tokens_per_step = args.batch_size * context_length

    avg_cpu = statistics.mean(times_cpu)
    std_cpu = statistics.stdev(times_cpu) if len(times_cpu) > 1 else 0.0
    throughput = tokens_per_step / avg_cpu

    flops = estimate_flops_per_step(
        n_params, num_layers, d_model, args.batch_size, context_length, args.mode)
    bytes_moved = estimate_bytes_per_step(
        n_params, num_layers, args.batch_size, context_length, d_model,
        args.mixed_precision, args.mode, args.attention)

    print(f'model={args.model_size} mode={args.mode} device={device} '
          f'mixed_precision={args.mixed_precision} attention={args.attention} '
          f'compile={args.compile} '
          f'batch={args.batch_size} '
          f'ctx={context_length} params={n_params / 1e6:.1f}M')

    print(f'[时间] cpu avg={avg_cpu:.4f}s std={std_cpu:.4f}s '
          f'({args.num_steps} steps after {args.warmup_steps} warmup)')
    if times_gpu:
        avg_gpu = statistics.mean(times_gpu)
        std_gpu = statistics.stdev(times_gpu) if len(times_gpu) > 1 else 0.0
        print(f'       gpu avg={avg_gpu:.4f}s std={std_gpu:.4f}s')

    print(f'[吞吐] {throughput:.0f} tokens/s')
    if times_gpu:
        # 用 GPU 时间算达成算力更真实(排除 launch 开销)
        print(f'       {flops / statistics.mean(times_gpu) / 1e12:.2f} TFLOP/s (by gpu time)')

    if device == 'cuda':
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        print(f'[显存] peak allocated={peak_alloc:.2f}GB '
              f'peak reserved={peak_reserved:.2f}GB')
    else:
        print('[显存] N/A (no CUDA)')

    print(f'[计算强度] est flops/step={flops / 1e9:.1f}GF '
          f'est bytes/step={bytes_moved / 1e9:.2f}GB '
          f'AI={flops / bytes_moved:.1f} FLOP/Byte (含激活流量, 保守估计)')


def main():
    parser = argparse.ArgumentParser(description='End-to-end benchmark of the basics Transformer model.')
    parser.add_argument('--model-size', choices=MODEL_SIZES.keys(), default='small')
    parser.add_argument('--mode', choices=['forward', 'forward-backward', 'full'], default='forward-backward')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--context-length', type=int, default=None,
                        help='Override the context length of the chosen model size (default: 512).')
    parser.add_argument('--warmup-steps', type=int, default=5)
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--mixed-precision', action='store_true', help='Use bf16 autocast for forward/backward.')
    parser.add_argument('--attention', choices=['naive', 'flash-pytorch', 'flash-triton'], default='naive',
                        help='Swap the A1 model attention for our FlashAttention implementation.')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--compile', action='store_true',
                        help='Wrap the model with torch.compile (default mode) before benchmarking.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    apply_attention_patch(args.attention)
    model = build_model(args).to(device)
    if args.compile:
        model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    vocab_size = MODEL_SIZES[args.model_size]['vocab_size']
    context_length = args.context_length or MODEL_SIZES[args.model_size]['context_length']
    x = torch.randint(0, vocab_size, (args.batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size, (args.batch_size, context_length), device=device)

    n_params = count_parameters(model)
    num_layers = MODEL_SIZES[args.model_size]['num_layers']
    d_model = MODEL_SIZES[args.model_size]['d_model']

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()  # 从 0 开始统计本轮峰值

    times_cpu, times_gpu = time_steps(
        model, optimizer, x, targets,
        args.mode, args.warmup_steps, args.num_steps, args.mixed_precision,
    )
    report(times_cpu, times_gpu, args, device, n_params, num_layers, d_model)


if __name__ == '__main__':
    main()
