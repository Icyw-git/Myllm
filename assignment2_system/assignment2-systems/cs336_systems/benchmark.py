import argparse
import statistics
from contextlib import nullcontext
import timeit
import torch

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


def build_model(args)->BasicsTransformerLM:
    config = dict(MODEL_SIZES[args.model_size])
    if args.context_length is not None:
        config['context_length'] = args.context_length
    return BasicsTransformerLM(**config)


def time_steps(model,optimizer,x,targets,mode,warmup,n,mixed_precision=False):
    use_cuda=torch.cuda.is_available()
    ctx=(torch.autocast('cuda',dtype=torch.bfloat16) if mixed_precision and use_cuda else nullcontext())
    def step():
        with ctx:
            if mode=='forward':
                with torch.no_grad():
                    model(x)
            elif mode=='forward-backward':
                logits=model(x)
                loss=cross_entropy(logits,targets)
                loss.backward()
            elif mode=='full':
                optimizer.zero_grad()
                logits=model(x)
                loss=cross_entropy(logits,targets)
                loss.backward()
                optimizer.step()
            else:
                raise ValueError(f"Unknown mode: {mode}")
        if use_cuda:
            torch.cuda.synchronize()

    for _ in range(warmup):
        step()
    times=[]
    for _ in range(n):
        t0=timeit.default_timer()
        step()
        times.append(timeit.default_timer()-t0)

    return times


def main():
    parser=argparse.ArgumentParser(description='End-to-end benchmark of the basics Transformer model.')
    parser.add_argument('--model-size', choices=MODEL_SIZES.keys(), default='small')
    parser.add_argument('--mode', choices=['forward','forward-backward','full'], default='forward-backward')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--context-length', type=int, default=None,
                        help='Override the context length of the chosen model size (default: 512).')
    parser.add_argument('--warmup-steps', type=int, default=5)
    parser.add_argument('--num-steps', type=int, default=10)
    parser.add_argument('--mixed-precision', action='store_true', help='Use bf16 autocast for forward/backward.')
    parser.add_argument('--seed', type=int, default=123)
    args=parser.parse_args()

    torch.manual_seed(args.seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'

    model=build_model(args).to(device)
    optimizer=AdamW(model.parameters(), lr=1e-4)

    vocab_size=MODEL_SIZES[args.model_size]['vocab_size']
    context_length=args.context_length or MODEL_SIZES[args.model_size]['context_length']
    x=torch.randint(0, vocab_size, (args.batch_size, context_length), device=device)
    targets=torch.randint(0, vocab_size, (args.batch_size, context_length), device=device)

    times=time_steps(model, optimizer, x, targets, args.mode, args.warmup_steps, args.num_steps, args.mixed_precision)
    avg=statistics.mean(times)
    std=statistics.stdev(times) if len(times) > 1 else 0.0
    tokens_per_sec=args.batch_size * context_length / avg
    print(f'model={args.model_size} mode={args.mode} device={device} '
          f'mixed_precision={args.mixed_precision} batch={args.batch_size} ctx={context_length}')
    print(f'avg={avg:.4f}s std={std:.4f}s ({args.num_steps} steps after {args.warmup_steps} warmup) '
          f'throughput={tokens_per_sec:.0f} tokens/s')


if __name__=='__main__':
    main()
                    