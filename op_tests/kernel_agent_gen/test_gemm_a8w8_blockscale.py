# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose
from torch.profiler import profile, ProfilerActivity
import time
from einops import rearrange

block_shape = (128, 128)

class Model(nn.Module):
    """Reference implementation using PyTorch operations."""
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor, weight: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor, dtype=dtypes.bf16) -> torch.Tensor:
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

        out = F.linear(x.to(dtypes.fp32), weight.to(dtypes.fp32))
        return out.to(dtype)


class ModelAiter(nn.Module):
    """AITER implementation using CK backend."""
    def __init__(self):
        super(ModelAiter, self).__init__()
    
    def forward(self, x: torch.Tensor, weight: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor, dtype=dtypes.bf16) -> torch.Tensor:
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


import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Optimized configs for AMD MFMA (Matrix Fused Multiply-Add) with FP16
        # Larger blocks for better matrix core utilization
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_a8w8_blockscale_kernel(
    x_ptr, weight_ptr, x_scale_ptr, w_scale_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_xscale_m, stride_xscale_k,
    stride_wscale_n, stride_wscale_k,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SCALE_BLOCK_K: tl.constexpr,
    SCALE_BLOCK_N: tl.constexpr,
):
    """
    Fused INT8 GEMM with block-wise scaling.
    Computes: out = (x * x_scale) @ (weight * w_scale)^T
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Compute block indices with swizzling for better L2 locality
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
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointer setup
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = weight_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
    
    # Accumulator in fp32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Boundary checks
        k_mask = (k + offs_k) < K
        x_mask = (offs_m[:, None] < M) & k_mask[None, :]
        w_mask = (offs_n[:, None] < N) & k_mask[None, :]
        
        # Load int8/fp8 values - keep in native format for matrix core
        x_fp8 = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Load scales for this K block
        # x_scale: [M, scale_k] where scale_k = K / SCALE_BLOCK_K
        # w_scale: [scale_n, scale_k] where scale_n = N / SCALE_BLOCK_N
        scale_k_idx = (k + offs_k) // SCALE_BLOCK_K
        
        # Load x_scale: [BLOCK_M, BLOCK_K]
        x_scale_ptrs = x_scale_ptr + (offs_m[:, None] * stride_xscale_m + scale_k_idx[None, :] * stride_xscale_k)
        x_scale_mask = (offs_m[:, None] < M) & k_mask[None, :]
        x_scale_fp32 = tl.load(x_scale_ptrs, mask=x_scale_mask, other=1.0)
        
        # Load w_scale: [BLOCK_N, BLOCK_K]
        scale_n_idx = offs_n // SCALE_BLOCK_N
        w_scale_ptrs = w_scale_ptr + (scale_n_idx[:, None] * stride_wscale_n + scale_k_idx[None, :] * stride_wscale_k)
        w_scale_mask = (offs_n[:, None] < N) & k_mask[None, :]
        w_scale_fp32 = tl.load(w_scale_ptrs, mask=w_scale_mask, other=1.0)
        
        # Convert scales to fp16 for efficient multiplication with fp8
        x_scale = x_scale_fp32.to(tl.float16)
        w_scale = w_scale_fp32.to(tl.float16)
        
        # Apply scales in fp16 (more efficient than fp32)
        # Cast fp8 to fp16 for scaling, keeping high precision
        x_scaled = x_fp8.to(tl.float16) * x_scale
        w_scaled = w_fp8.to(tl.float16) * w_scale
        
        # Use fp16 matrix multiplication with matrix cores (MFMA on AMD)
        # This utilizes hardware matrix cores for high throughput
        # Accumulate in fp32 for numerical stability
        acc += tl.dot(x_scaled, tl.trans(w_scaled), out_dtype=tl.float32)
        
        # Advance pointers
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    
    # Convert to output dtype and store
    out = acc.to(tl.float16)
    
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, out, mask=out_mask)


def triton_gemm_a8w8_blockscale(x: torch.Tensor, weight: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor, dtype=dtypes.bf16) -> torch.Tensor:
    """
    Triton implementation of block-scaled INT8 GEMM.
    
    Args:
        x: [M, K] int8 input
        weight: [N, K] int8 weight
        x_scale: [M, scale_k] fp32 activation scales
        w_scale: [scale_n, scale_k] fp32 weight scales
        dtype: output dtype
    
    Returns:
        output: [M, N] in specified dtype
    """
    assert x.is_cuda and weight.is_cuda
    # Support multiple fp8 formats
    fp8_dtypes = [torch.int8, torch.float8_e4m3fn]
    if hasattr(torch, 'float8_e4m3fnuz'):
        fp8_dtypes.append(torch.float8_e4m3fnuz)
    assert x.dtype in fp8_dtypes, f"x.dtype must be one of {fp8_dtypes}, got {x.dtype}"
    assert weight.dtype in fp8_dtypes, f"weight.dtype must be one of {fp8_dtypes}, got {weight.dtype}"
    assert x_scale.dtype == torch.float32
    assert w_scale.dtype == torch.float32
    
    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K
    
    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    x_scale = x_scale.contiguous()
    w_scale = w_scale.contiguous()
    
    # Allocate output
    out = torch.empty((M, N), device=x.device, dtype=torch.float16)
    
    # Grid configuration
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )
    
    # Launch kernel
    gemm_a8w8_blockscale_kernel[grid](
        x, weight, x_scale, w_scale, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        x_scale.stride(0), x_scale.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        out.stride(0), out.stride(1),
        SCALE_BLOCK_K=block_shape[1],
        SCALE_BLOCK_N=block_shape[0],
    )
    
    return out.to(dtype)


class ModelNew(nn.Module):
    """
    Optimized implementation using Triton kernel with fused block-wise scaling.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor, weight: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor, dtype=dtypes.bf16) -> torch.Tensor:
        """
        Args:
            x: [m, k] int8 input
            weight: [n, k] int8 weight
            x_scale: [m, scale_k] fp32 activation scales
            w_scale: [scale_n, scale_k] fp32 weight scales
        Returns:
            output: [m, n] in specified dtype
        """
        return triton_gemm_a8w8_blockscale(x, weight, x_scale, w_scale, dtype)


# Test configuration
m = 1024
n = 4096
k = 4096
block_shape_n, block_shape_k = block_shape
scale_n = (n + block_shape_n - 1) // block_shape_n
scale_k = (k + block_shape_k - 1) // block_shape_k

def get_inputs():
    # Support both fp8 formats
    fp8_dtype = getattr(torch, 'float8_e4m3fnuz', dtypes.fp8)
    x = (torch.rand((m, k), dtype=dtypes.fp16, device="cuda") / 10).to(fp8_dtype)
    weight = (torch.rand((n, k), dtype=dtypes.fp16, device="cuda") / 10).to(fp8_dtype)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")
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
        output_orig = model_orig(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
        output_aiter = model_aiter(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
        output_new = model_new(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
    
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
            _ = model_orig(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
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
                _ = model_orig(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
            torch.cuda.synchronize()
    
    print("=" * 80)
    print("Original Model (PyTorch):")
    print("=" * 80)
    print(prof_orig.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Benchmark ModelAiter (AITER)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_aiter(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
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
                _ = model_aiter(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
            torch.cuda.synchronize()
    
    print("\n" + "=" * 80)
    print("ModelAiter (AITER gemm_a8w8_blockscale):")
    print("=" * 80)
    print(prof_aiter.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Benchmark ModelNew (Triton)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_new(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
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
                _ = model_new(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
            torch.cuda.synchronize()
    
    print("\n" + "=" * 80)
    print("ModelNew (Triton Fused):")
    print("=" * 80)
    print(prof_new.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Simple timing measurements
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_orig(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model_orig(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
        torch.cuda.synchronize()
        orig_time = (time.time() - start) / iterations
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_aiter(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model_aiter(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
        torch.cuda.synchronize()
        aiter_time = (time.time() - start) / iterations
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_new(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model_new(x, weight, x_scale, w_scale, dtype=dtypes.bf16)
        torch.cuda.synchronize()
        new_time = (time.time() - start) / iterations
    
    print("\n" + "=" * 80)
    print("Performance Summary:")
    print("=" * 80)
    print(f"Original Model (PyTorch) avg time:  {orig_time*1000:.3f} ms")
    print(f"ModelAiter (AITER) avg time:        {aiter_time*1000:.3f} ms")
    print(f"ModelNew (Triton) avg time:         {new_time*1000:.3f} ms")
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
