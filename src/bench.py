import argparse
import csv
import itertools
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F
import torch.utils.cpp_extension as cpp_ext

SRC_DIR = Path(__file__).resolve().parent

def setup_cuda_env():
    if not os.environ.get('CUDA_HOME') or cpp_ext.CUDA_HOME is None:
        for candidate in (cpp_ext.CUDA_HOME, '/usr/local/cuda', '/usr/local/cuda-12.2', '/usr/local/cuda-12'):
            if candidate and Path(candidate).exists():
                os.environ['CUDA_HOME'] = str(candidate)
                cpp_ext.CUDA_HOME = str(candidate)
                break

def load_custom_kernels():
    if not hasattr(load_custom_kernels, '_module'):
        setup_cuda_env()
        load_custom_kernels._module = cpp_ext.load(
            name='flash_attn',
            sources=[
                str(SRC_DIR / 'main.cpp'),
                str(SRC_DIR / 'flash_attn.cu'),
                str(SRC_DIR / 'naive_attn.cu'),
                str(SRC_DIR / 'flash_attn_unified.cu'),
                str(SRC_DIR / 'naive_attn_unified.cu'),
            ],
            extra_cuda_cflags=['-O2']
        )
    return load_custom_kernels._module

def torch_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def build_inputs(batch_size, n_head, seq_len, head_embd):
    q = torch.randn(batch_size, n_head, seq_len, head_embd, device='cuda')
    k = torch.randn(batch_size, n_head, seq_len, head_embd, device='cuda')
    v = torch.randn(batch_size, n_head, seq_len, head_embd, device='cuda')
    return q, k, v

def benchmark_impl(fn, q, k, v, warmup, iters):
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(q, k, v)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)

def check_correctness(fn, q, k, v, reference, rtol, atol):
    out = fn(q, k, v)
    torch.cuda.synchronize()
    return torch.allclose(out, reference, rtol=rtol, atol=atol)

def plot_results(records, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    if not records:
        return

    impls = sorted({r['impl'] for r in records})
    heads = sorted({r['n_head'] for r in records})
    base = Path(path)
    suffix = base.suffix or '.png'
    styles = [
        {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
        {'color': '#d62728', 'marker': 's', 'linestyle': '--'},
        {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},
        {'color': '#9467bd', 'marker': 'D', 'linestyle': ':'},
        {'color': '#ff7f0e', 'marker': 'v', 'linestyle': '-'},
    ]

    for nh in heads:
        fig, ax = plt.subplots(figsize=(5, 4))
        subset = [r for r in records if r['n_head'] == nh]
        seqs = sorted({r['seq_len'] for r in subset})
        for i, impl in enumerate(impls):
            ys = []
            xs = []
            for seq in seqs:
                match = [r for r in subset if r['seq_len'] == seq and r['impl'] == impl]
                if match:
                    xs.append(seq)
                    ys.append(match[0]['ms'])
            style = styles[i % len(styles)]
            ax.plot(xs, ys, label=impl, linewidth=2, markersize=5, **style)

        ax.set_title(f'Attention latency (nh={nh})')
        ax.set_xlabel('seq_len (N)')
        ax.set_ylabel('latency (ms)')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(title='Implementation', frameon=True)
        fig.tight_layout()

        out_path = base.parent / f"{base.stem}_nh{nh}{suffix}"
        plt.savefig(out_path, dpi=220)
        print(f'plot saved to {out_path}')

def run_profile(args, kernels):
    seq_len = args.seq_lens[0]
    n_head = args.n_heads[0]
    q, k, v = build_inputs(args.batch_size, n_head, seq_len, args.head_embd)

    print(f'=== single profile (N={seq_len}, nh={n_head}) ===')
    torch_result = torch_attn(q, k, v)
    impls = [
        ('torch', torch_attn),
        ('naive', kernels.naive_attention),
        ('flash', kernels.flash_attention),
        ('naive_unified', kernels.naive_attention_unified),
        ('flash_unified', kernels.flash_attention_unified),
    ]
    for name, fn in impls:
        ms = benchmark_impl(fn, q, k, v, args.warmup, args.iters)
        check = True if name == 'torch' else check_correctness(fn, q, k, v, torch_result, args.rtol, args.atol)
        print(f'{name:15s}: {ms:8.3f} ms | correct={check}')

def run_sweep(args, kernels):
    records = []
    impls = [
        ('torch', torch_attn),
        ('naive', kernels.naive_attention),
        ('flash', kernels.flash_attention),
        ('naive_unified', kernels.naive_attention_unified),
        ('flash_unified', kernels.flash_attention_unified),
    ]

    for seq_len, n_head in itertools.product(args.seq_lens, args.n_heads):
        q, k, v = build_inputs(args.batch_size, n_head, seq_len, args.head_embd)
        torch_result = torch_attn(q, k, v)

        print(f'\n--- N={seq_len}, nh={n_head} ---')
        for name, fn in impls:
            ms = benchmark_impl(fn, q, k, v, args.warmup, args.iters)
            ok = True if name == 'torch' else check_correctness(fn, q, k, v, torch_result, args.rtol, args.atol)
            records.append({'seq_len': seq_len, 'n_head': n_head, 'impl': name, 'ms': ms})
            print(f'{name:15s}: {ms:8.3f} ms | correct={ok}')

    csv_path = Path(args.csv)
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['seq_len', 'n_head', 'impl', 'ms'])
        writer.writeheader()
        writer.writerows(records)
    print(f'\nraw results saved to {csv_path}')

    plot_results(records, Path(args.plot))

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark attention kernels across N and nh.')
    parser.add_argument('--mode', choices=['profile', 'sweep'], default='sweep', help='profile single config or sweep grid')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seq-lens', type=int, nargs='+', default=[64, 128, 256], help='list of sequence lengths (N)')
    parser.add_argument('--n-heads', type=int, nargs='+', default=[1, 2, 4], help='list of head counts (nh)')
    parser.add_argument('--head-embd', type=int, default=64, help='head embedding dimension (d)')
    parser.add_argument('--warmup', type=int, default=1, help='warmup iterations before timing')
    parser.add_argument('--iters', type=int, default=5, help='timed iterations per implementation')
    parser.add_argument('--plot', type=str, default='bench_plot.png', help='output plot path')
    parser.add_argument('--csv', type=str, default='bench_results.csv', help='output csv path')
    parser.add_argument('--rtol', type=float, default=1e-2, help='relative tolerance for correctness check')
    parser.add_argument('--atol', type=float, default=1e-2, help='absolute tolerance for correctness check')
    return parser.parse_args()

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this benchmark.')

    torch.manual_seed(0)
    args = parse_args()
    kernels = load_custom_kernels()
    if args.mode == 'profile':
        run_profile(args, kernels)
    else:
        run_sweep(args, kernels)
