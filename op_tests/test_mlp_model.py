import torch
import torch.nn as nn
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose
from torch.profiler import profile, ProfilerActivity
import time
from aiter.tuned_gemm import tgemm

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.logsumexp(x, dim=1)
        return x

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.weight1 = nn.Parameter(torch.randn(hidden_size, input_size, dtype=dtypes.bf16))
        self.bias1 = nn.Parameter(torch.randn(hidden_size, dtype=dtypes.bf16))
        self.weight2 = nn.Parameter(torch.randn(output_size, hidden_size, dtype=dtypes.bf16))
        self.bias2 = nn.Parameter(torch.randn(output_size, dtype=dtypes.bf16))
    
    def forward(self, x):
        x = tgemm.mm(x, self.weight1, self.bias1, dtypes.bf16)
        x = aiter.sigmoid(x)
        x = tgemm.mm(x, self.weight2, self.bias2, dtypes.bf16)
        x = torch.logsumexp(x, dim=1)
        return x
    
    def load_from_original(self, original_model):
        self.weight1.data.copy_(original_model.linear1.weight.data.to(dtypes.bf16))
        self.bias1.data.copy_(original_model.linear1.bias.data.to(dtypes.bf16))
        self.weight2.data.copy_(original_model.linear2.weight.data.to(dtypes.bf16))
        self.bias2.data.copy_(original_model.linear2.bias.data.to(dtypes.bf16))

import triton
import triton.language as tl
import torch
import torch.nn as nn


@triton.jit
def linear_sigmoid_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_start = 0
    while k_start < K:
        k_idxs = k_start + offs_k
        mask_k = k_idxs < K

        # keep input tiles in fp16/bf16 for mma
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk)
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        w_ptrs = W_ptr + (offs_n[:, None] * stride_wn + k_idxs[None, :] * stride_wk)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        # dot does internal fp16->fp32 accumulation
        # acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)

        common_dtype = tl.bfloat16
        x = x.to(common_dtype)
        w = w.to(common_dtype)
        acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)


        k_start += BLOCK_K

    b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    b = tl.cast(b, tl.float32)
    acc = acc + b[None, :]

    acc = 1.0 / (1.0 + tl.exp(-acc))

    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_linear_sigmoid(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    N, Kw = weight.shape

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 128

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    linear_sigmoid_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


@triton.jit
def linear_logsumexp_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_b,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return

    running_max = tl.full((1,), -float('inf'), dtype=tl.float32)
    running_sum = tl.zeros((1,), dtype=tl.float32)

    n_start = 0
    while n_start < N:
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc_tile = tl.zeros((1, BLOCK_N), dtype=tl.float32)

        k_start = 0
        while k_start < K:
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            x_ptrs = X_ptr + (m * stride_xm + offs_k * stride_xk)
            x_row = tl.load(x_ptrs, mask=mask_k, other=0.0)[None, :]

            w_ptrs = W_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
            w_block = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

            # acc_tile += tl.dot(x_row, tl.trans(w_block), out_dtype=tl.float32)
            common_dtype = tl.bfloat16
            x_row = x_row.to(common_dtype)
            w_block = w_block.to(common_dtype)
            acc_tile += tl.dot(x_row, tl.trans(w_block), out_dtype=tl.float32)

            k_start += BLOCK_K

        b_block = tl.load(B_ptr + offs_n * stride_b, mask=mask_n, other=0.0)
        b_block = tl.cast(b_block, tl.float32)[None, :]
        z = acc_tile + b_block

        tile_max = tl.max(z, axis=1)
        new_max = tl.maximum(running_max, tile_max)

        running_sum = running_sum * tl.exp(running_max - new_max)
        exp_tile = tl.exp(z - new_max[:, None])
        tile_sum = tl.sum(exp_tile, axis=1)
        running_sum = running_sum + tile_sum
        running_max = new_max

        n_start += BLOCK_N

    out_val = tl.log(running_sum) + running_max
    tl.store(Out_ptr + m, tl.sum(out_val))


def triton_linear_logsumexp(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    N, Kw = weight.shape

    out = torch.empty((M,), device=x.device, dtype=torch.float32)

    BLOCK_N = 128
    BLOCK_K = 128

    grid = lambda meta: (triton.cdiv(M, 1),)

    linear_logsumexp_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        bias.stride(0),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


class ModelAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelAgent, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = triton_linear_sigmoid(x, self.linear1.weight, self.linear1.bias)
        x = triton_linear_logsumexp(x, self.linear2.weight, self.linear2.bias)
        return x


batch_size = 16384
input_size = 2048
hidden_size = 4096
output_size = 1024

def get_inputs():
    return [torch.rand(batch_size, input_size, dtype=dtypes.bf16, device="cuda")]

def get_init_inputs():
    return [input_size, hidden_size, output_size]

def test_correctness():
    init_inputs = get_init_inputs()
    model_orig = Model(*init_inputs).cuda()
    model_new = ModelNew(*init_inputs).cuda()
    model_new.load_from_original(model_orig)
    
    inputs = get_inputs()
    x = inputs[0]
    
    with torch.no_grad():
        output_orig = model_orig(x.to(dtypes.fp32)).to(dtypes.bf16)
        output_new = model_new(x)
    
    checkAllclose(output_orig, output_new, msg="gemm_sigmoid_gemm_logsumexp", rtol=1e-1, atol=1e-1)
    print("Correctness test passed!")

def test_speed():
    init_inputs = get_init_inputs()
    model_orig = Model(*init_inputs).cuda()
    model_new = ModelNew(*init_inputs).cuda()
    model_new.load_from_original(model_orig)
    model_agent = ModelAgent(*init_inputs).cuda()
    
    inputs = get_inputs()
    x = inputs[0]
    
    warmup = 10
    iterations = 100
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_orig(x.to(dtypes.fp32))
        torch.cuda.synchronize()
        
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            with_stack=True,
            with_modules=True,
            record_shapes=True,
        ) as prof_orig:
            for _ in range(iterations):
                _ = model_orig(x.to(dtypes.fp32))
            torch.cuda.synchronize()
    
    print("Original Model:")
    print(prof_orig.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_new(x)
        torch.cuda.synchronize()
        
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            with_stack=True,
            with_modules=True,
            record_shapes=True,
        ) as prof_new:
            for _ in range(iterations):
                _ = model_new(x)
            torch.cuda.synchronize()
    
    print("New Model (AITER):")
    print(prof_new.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_orig(x.to(dtypes.fp32))
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model_orig(x.to(dtypes.fp32))
        torch.cuda.synchronize()
        orig_time = (time.time() - start) / iterations
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_new(x)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model_new(x)
        torch.cuda.synchronize()
        new_time = (time.time() - start) / iterations
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_agent(x)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model_agent(x)
        torch.cuda.synchronize()
        agent_time = (time.time() - start) / iterations
    
    print(f"\nOriginal Model avg time: {orig_time*1000:.3f} ms")
    print(f"New Model (AITER) avg time: {new_time*1000:.3f} ms")
    print(f"NewNew Model (LLM) avg time: {agent_time*1000:.3f} ms")
    print(f"Speedup AITER: {orig_time/new_time:.2f}x")
    print(f"Speedup LLM: {orig_time/agent_time:.2f}x")

if __name__ == "__main__":
    test_correctness()
    test_speed()
