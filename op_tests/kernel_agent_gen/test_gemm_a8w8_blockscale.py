# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose
from torch.profiler import profile, ProfilerActivity
import time
from einops import rearrange

block_shape = (128, 128)

class Model(nn.Module):
    """Reference implementation using PyTorch operations."""
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor, weight: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
        """
        Args:
            x: [m, k] int8 input
            weight: [n, k] int8 weight
            x_scale: [m, scale_k] fp32 activation scales
            w_scale: [scale_n, scale_k] fp32 weight scales
        Returns:
            output: [m, n] in specified dtype
        """
        block_shape_n, block_shape_k = block_shape
        m, k = x.shape
        n = weight.shape[0]
        scale_n = (n + block_shape_n - 1) // block_shape_n
        scale_k = (k + block_shape_k - 1) // block_shape_k
        
        # Apply block-wise scaling to input
        x = x.to(x_scale.dtype).view(
            m, k // block_shape[1], block_shape[1]
        ) * x_scale.unsqueeze(-1)
        x = x.view(m, k)

        # Apply block-wise scaling to weight
        w_scale = rearrange(
            w_scale.view(-1, 1)
            .repeat(1, block_shape_n * block_shape_k)
            .view(scale_n, scale_k, block_shape_n, block_shape_k),
            "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
        )
        w_scale = w_scale[:n, :k]
        weight = weight.to(w_scale.dtype) * w_scale

        out = F.linear(x.to(torch.float32), weight.to(torch.float32))
        return out.to(dtype)


class ModelAiter(nn.Module):
    """AITER implementation using CK backend."""
    def __init__(self):
        super(ModelAiter, self).__init__()
    
    def forward(self, x: torch.Tensor, weight: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
        """
        Args:
            x: [m, k] int8 input
            weight: [n, k] int8 weight
            x_scale: [m, scale_k] fp32 activation scales
            w_scale: [scale_n, scale_k] fp32 weight scales
        Returns:
            output: [m, n] in specified dtype
        """
        return aiter.gemm_a8w8_blockscale(x, weight, x_scale, w_scale, dtype)


import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import subprocess

block_shape = (128, 128)

def get_fp8_dtype():
    """Auto-detect correct FP8 dtype for current AMD GPU."""
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True)
        output = result.stdout
        
        if 'gfx950' in output or 'gfx960' in output:
            return torch.float8_e4m3fn
        else:
            return getattr(torch, 'float8_e4m3fnuz', torch.float8_e4m3fn)
    except:
        return getattr(torch, 'float8_e4m3fnuz', torch.float8_e4m3fn)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=16),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, 
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, 
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, 
                      num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_scaled_matmul_kernel(
    x_ptr, weight_ptr, x_scale_ptr, w_scale_ptr, out_ptr,
    M, N, K,
    SCALE_BLOCK_N: tl.constexpr,
    SCALE_BLOCK_K: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_xscale_m, stride_xscale_k,
    stride_wscale_n, stride_wscale_k,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Optimized fused kernel for block-wise scaled fp8 matrix multiplication.
    Key optimizations:
    1. Efficient scale loading with precomputed indices
    2. Minimized scale extraction overhead
    3. Better memory access patterns
    4. Reduced arithmetic operations in hot loop
    """
    pid = tl.program_id(0)
    
    # Block swizzling for better L2 cache locality
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator in FP32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Precompute scale indices for n dimension (constant across K loop)
    scale_n_idx = offs_n // SCALE_BLOCK_N
    
    # Iterate over K dimension
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        
        # Load input block [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        x_mask = (offs_m[:, None] < M) & k_mask[None, :]
        x_fp8 = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load weight block [BLOCK_N, BLOCK_K]
        w_ptrs = weight_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
        w_mask = (offs_n[:, None] < N) & k_mask[None, :]
        w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Compute FP8 matmul with FP32 output
        result = tl.dot(x_fp8, tl.trans(w_fp8), out_dtype=tl.float32)
        
        # Compute scale indices for this K block
        scale_k_idx = k // SCALE_BLOCK_K
        
        # Load x_scale values (only need one per M row for this K block)
        x_scale_ptrs = x_scale_ptr + (offs_m * stride_xscale_m + scale_k_idx * stride_xscale_k)
        x_scale_mask = offs_m < M
        x_scale_broadcast = tl.load(x_scale_ptrs, mask=x_scale_mask, other=1.0)
        
        # Load w_scale values (only need one per N row for this K block)
        w_scale_ptrs = w_scale_ptr + (scale_n_idx * stride_wscale_n + scale_k_idx * stride_wscale_k)
        w_scale_mask = offs_n < N
        w_scale_broadcast = tl.load(w_scale_ptrs, mask=w_scale_mask, other=1.0)
        
        # Compute outer product of scales [BLOCK_M, 1] × [1, BLOCK_N] = [BLOCK_M, BLOCK_N]
        scale_factor = x_scale_broadcast[:, None] * w_scale_broadcast[None, :]
        
        # Apply scaling and accumulate
        acc += result * scale_factor
    
    # Store output
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Convert to output dtype (bfloat16)
    out = acc.to(tl.bfloat16)
    tl.store(out_ptrs, out, mask=out_mask)


def triton_fused_scaled_matmul(x: torch.Tensor, weight: torch.Tensor, 
                                x_scale: torch.Tensor, w_scale: torch.Tensor,
                                dtype=torch.bfloat16) -> torch.Tensor:
    """
    Wrapper function for fused scaled matmul kernel.
    
    Args:
        x: [M, K] int8/fp8 input
        weight: [N, K] int8/fp8 weight
        x_scale: [M, scale_k] fp32 activation scales
        w_scale: [scale_n, scale_k] fp32 weight scales
        dtype: output dtype
    
    Returns:
        output: [M, N] in specified dtype
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA"
    
    x = x.contiguous()
    weight = weight.contiguous()
    x_scale = x_scale.contiguous()
    w_scale = w_scale.contiguous()
    
    M, K = x.shape
    N = weight.shape[0]
    
    SCALE_BLOCK_N, SCALE_BLOCK_K = block_shape
    
    # Allocate output
    out = torch.empty((M, N), dtype=dtype, device=x.device)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )
    
    fused_scaled_matmul_kernel[grid](
        x, weight, x_scale, w_scale, out,
        M, N, K,
        SCALE_BLOCK_N, SCALE_BLOCK_K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        x_scale.stride(0), x_scale.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        out.stride(0), out.stride(1),
    )
    
    return out


class ModelNew(nn.Module):
    """Optimized implementation using custom Triton kernel."""
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor, weight: torch.Tensor, 
                x_scale: torch.Tensor, w_scale: torch.Tensor, 
                dtype=torch.bfloat16) -> torch.Tensor:
        """
        Args:
            x: [m, k] int8/fp8 input
            weight: [n, k] int8/fp8 weight
            x_scale: [m, scale_k] fp32 activation scales
            w_scale: [scale_n, scale_k] fp32 weight scales
        Returns:
            output: [m, n] in specified dtype
        """
        return triton_fused_scaled_matmul(x, weight, x_scale, w_scale, dtype)


# Test configuration
m = 1024
n = 4096
k = 4096
block_shape_n, block_shape_k = block_shape
scale_n = (n + block_shape_n - 1) // block_shape_n
scale_k = (k + block_shape_k - 1) // block_shape_k

def get_inputs():
    # Support both fp8 formats
    fp8_dtype = getattr(torch, 'float8_e4m3fnuz', torch.float8_e4m3fn)
    x = (torch.rand((m, k), dtype=torch.float16, device="cuda") / 10).to(fp8_dtype)
    weight = (torch.rand((n, k), dtype=torch.float16, device="cuda") / 10).to(fp8_dtype)
    x_scale = torch.rand([m, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")
    return [x, weight, x_scale, w_scale]

def get_init_inputs():
    return []

def test_correctness():
    """Test that all three implementations produce the same results."""
    model_orig = Model().cuda()
    model_aiter = ModelAiter().cuda()
    model_new = ModelNew().cuda()
    
    inputs = get_inputs()
    x, weight, x_scale, w_scale = inputs
    
    with torch.no_grad():
        output_orig = model_orig(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
        output_aiter = model_aiter(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
        output_new = model_new(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
    
    checkAllclose(output_orig, output_aiter, msg="gemm_a8w8_blockscale (ModelAiter)", rtol=1e-2, atol=0.01)
    checkAllclose(output_orig, output_new, msg="gemm_a8w8_blockscale (ModelNew)", rtol=1e-2, atol=0.01)
    print("✓ Correctness test passed for both ModelAiter and ModelNew!")

def test_speed():
    """Benchmark the performance of all three implementations."""
    model_orig = Model().cuda()
    model_aiter = ModelAiter().cuda()
    model_new = ModelNew().cuda()
    
    inputs = get_inputs()
    x, weight, x_scale, w_scale = inputs
    
    warmup = 10
    iterations = 100
    
    # Benchmark Original Model
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_orig(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
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
                _ = model_orig(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
            torch.cuda.synchronize()
    
    print("=" * 80)
    print("Original Model (PyTorch):")
    print("=" * 80)
    print(prof_orig.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Benchmark ModelAiter (AITER)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_aiter(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            with_stack=True,
            with_modules=True,
            record_shapes=True,
        ) as prof_aiter:
            for _ in range(iterations):
                _ = model_aiter(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
            torch.cuda.synchronize()
    
    print("\n" + "=" * 80)
    print("ModelAiter (AITER gemm_a8w8_blockscale):")
    print("=" * 80)
    print(prof_aiter.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Benchmark ModelNew (Triton)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_new(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
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
                _ = model_new(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
            torch.cuda.synchronize()
    
    print("\n" + "=" * 80)
    print("ModelNew (Triton Fused):")
    print("=" * 80)
    print(prof_new.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Simple timing measurements
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_orig(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model_orig(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        orig_time = (time.time() - start) / iterations
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_aiter(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model_aiter(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        aiter_time = (time.time() - start) / iterations
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_new(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model_new(x, weight, x_scale, w_scale, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        new_time = (time.time() - start) / iterations
    
    print("\n" + "=" * 80)
    print("Performance Summary:")
    print("=" * 80)
    print(f"Original Model (PyTorch) avg time:  {orig_time*1000:.3f} ms")
    print(f"ModelAiter (AITER) avg time:        {aiter_time*1000:.3f} ms")
    print(f"ModelNew (LLM-Triton) avg time:         {new_time*1000:.3f} ms")
    print(f"\nSpeedup AITER vs PyTorch:   {orig_time/aiter_time:.2f}x")
    print(f"Speedup Triton vs PyTorch:  {orig_time/new_time:.2f}x")
    print(f"Speedup Triton vs AITER:    {aiter_time/new_time:.2f}x")
    print("=" * 80)

if __name__ == "__main__":
    print("Testing INT8 Block-Scaled GEMM Implementations")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  M:             {m}")
    print(f"  N:             {n}")
    print(f"  K:             {k}")
    print(f"  Block shape:   {block_shape}")
    print(f"  Scale shape:   ({scale_n}, {scale_k})")
    print("=" * 80)
    print()
    
    test_correctness()
    print()
    test_speed()
