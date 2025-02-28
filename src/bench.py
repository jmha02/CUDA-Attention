import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

custom_kernels = load(name='flash_attn', sources=['main.cpp', 'flash_attn.cu', 'naive_attn.cu'], extra_cuda_cflags=['-O2'])

batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

def torch_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

print('=== profiling PyTorch attention ===')
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch_result = torch_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling CUDA naive attention ===')
with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
    naive_result = custom_kernels.naive_attention(q, k, v)
    naive_memory_allocated = torch.cuda.memory_allocated()

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
print('Naive attention memory allocated:', naive_memory_allocated / (1024 ** 2))
print('attn values sanity check:', torch.allclose(naive_result, torch_result, rtol=0, atol=1e-02))

print('=== profiling CUDA flash attention ===')
with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
    torch.cuda.reset_peak_memory_stats()
    flash_result = custom_kernels.flash_attention(q, k, v)
    flash_memory_allocated = torch.cuda.memory_allocated()

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
print('Flash attention memory allocated:', flash_memory_allocated / (1024 ** 2))
print('attn values sanity check:', torch.allclose(flash_result, torch_result, rtol=0, atol=1e-02))

