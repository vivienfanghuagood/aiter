import torch
import torch.nn as nn
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose
from torch.profiler import profile, ProfilerActivity
import time

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.bmm(A, B)

class ModelAiter(nn.Module):
    def __init__(self):
        super(ModelAiter, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        B_transposed = B.transpose(1, 2)
        return aiter.batched_gemm_bf16_CK(A, B_transposed, bias=None, dtype=dtypes.bf16)

import triton
import triton.language as tl
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=16),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Get batch index
    batch_idx = tl.program_id(2)
    
    # Get program IDs for M and N dimensions
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Compute offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Compute base pointers for this batch
    A_batch_ptr = A_ptr + batch_idx * stride_ab
    B_batch_ptr = B_ptr + batch_idx * stride_bb
    C_batch_ptr = C_ptr + batch_idx * stride_cb
    
    # Initialize pointers to A and B for this block
    A_block_ptr = A_batch_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_block_ptr = B_batch_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension in BLOCK_K chunks
    for k in range(0, K, BLOCK_K):
        # Create masks for boundary conditions
        mask_a = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        mask_b = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        
        # Load A and B blocks
        a = tl.load(A_block_ptr, mask=mask_a, other=0.0)
        b = tl.load(B_block_ptr, mask=mask_b, other=0.0)
        
        # Accumulate
        acc += tl.dot(a, b)
        
        # Advance pointers
        A_block_ptr += BLOCK_K * stride_ak
        B_block_ptr += BLOCK_K * stride_bk
    
    # Convert accumulator to output dtype
    c = acc.to(C_ptr.dtype.element_ty)
    
    # Write output
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_block_ptr = C_batch_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(C_block_ptr, c, mask=mask_c)


def triton_batched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched matrix multiplication using Triton kernel.
    A: (batch_size, M, K)
    B: (batch_size, K, N)
    Returns: C of shape (batch_size, M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.shape[0] == B.shape[0], "Batch sizes must match"
    assert A.shape[2] == B.shape[1], "Inner dimensions must match"
    
    batch_size, M, K = A.shape
    _, _, N = B.shape
    
    # Ensure contiguous tensors
    A = A.contiguous()
    B = B.contiguous()
    
    # Allocate output
    C = torch.empty((batch_size, M, N), device=A.device, dtype=A.dtype)
    
    # Get strides
    stride_ab, stride_am, stride_ak = A.stride()
    stride_bb, stride_bk, stride_bn = B.stride()
    stride_cb, stride_cm, stride_cn = C.stride()
    
    # Launch kernel
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
            1,
            batch_size
        )
    
    batched_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
    )
    
    return C


class ModelNew(nn.Module):
    """
    Optimized batched matrix multiplication using custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return triton_batched_matmul(A, B)

# Test configurations: (batch_size, m, k, n)
test_shapes = [
    (16, 256, 512, 512),
    (32, 512, 512, 1024),
    (64, 512, 1024, 512),
    (128, 512, 1024, 2048),
    (256, 256, 512, 1024),
]

def get_inputs(batch_size, m, k, n):
    A = torch.rand(batch_size, m, k, dtype=dtypes.bf16, device="cuda")
    B = torch.rand(batch_size, k, n, dtype=dtypes.bf16, device="cuda")
    return [A, B]

def get_init_inputs():
    return []

def test_correctness():
    """Test that all three implementations produce the same results."""
    model_orig = Model().cuda()
    model_aiter = ModelAiter().cuda()
    model_new = ModelNew().cuda()
    
    batch_size, m, k, n = test_shapes[0]
    inputs = get_inputs(batch_size, m, k, n)
    A, B = inputs
    
    with torch.no_grad():
        output_orig = model_orig(A, B).float()
        output_new = model_aiter(A, B).float()
        output_agent = model_new(A, B).float()
    
    checkAllclose(output_orig, output_new, msg="batched_gemm (ModelAiter)", rtol=1e-2, atol=0.01)
    checkAllclose(output_orig, output_agent, msg="batched_gemm (ModelNew)", rtol=1e-2, atol=0.01)
    print("✓ Correctness test passed for both ModelAiter and ModelNew!")

def test_speed():
    """Benchmark the performance of all three implementations across multiple shapes."""
    import numpy as np
    
    model_orig = Model().cuda()
    model_aiter = ModelAiter().cuda()
    model_new = ModelNew().cuda()
    
    warmup = 10
    iterations = 100
    
    orig_times = []
    aiter_times = []
    new_times = []
    
    for batch_size, m, k, n in test_shapes:
        print(f"\n{'='*80}")
        print(f"Testing shape: batch={batch_size}, m={m}, k={k}, n={n}")
        print(f"{'='*80}")
        
        inputs = get_inputs(batch_size, m, k, n)
        A, B = inputs
    
        # Simple timing measurements
        with torch.no_grad():
            for _ in range(warmup):
                _ = model_orig(A, B)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(iterations):
                _ = model_orig(A, B)
            torch.cuda.synchronize()
            orig_time = (time.time() - start) / iterations
        
        with torch.no_grad():
            for _ in range(warmup):
                _ = model_aiter(A, B)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(iterations):
                _ = model_aiter(A, B)
            torch.cuda.synchronize()
            aiter_time = (time.time() - start) / iterations
        
        with torch.no_grad():
            for _ in range(warmup):
                _ = model_new(A, B)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(iterations):
                _ = model_new(A, B)
            torch.cuda.synchronize()
            new_time = (time.time() - start) / iterations
        
        orig_times.append(orig_time)
        aiter_times.append(aiter_time)
        new_times.append(new_time)
        
        print(f"Original Model (PyTorch) avg time:  {orig_time*1000:.3f} ms")
        print(f"ModelAiter (AITER) avg time:        {aiter_time*1000:.3f} ms")
        print(f"ModelNew (Triton) avg time:         {new_time*1000:.3f} ms")
        print(f"Speedup AITER vs PyTorch:   {orig_time/aiter_time:.2f}x")
        print(f"Speedup Triton vs PyTorch:  {orig_time/new_time:.2f}x")
        print(f"Speedup Triton vs AITER:    {aiter_time/new_time:.2f}x")
    
    # Calculate geometric mean
    orig_geomean = np.exp(np.mean(np.log(orig_times)))
    aiter_geomean = np.exp(np.mean(np.log(aiter_times)))
    new_geomean = np.exp(np.mean(np.log(new_times)))
    
    print("\n" + "=" * 80)
    print("Geometric Mean Performance Summary:")
    print("=" * 80)
    print(f"Original Model (PyTorch) geomean time:  {orig_geomean*1000:.3f} ms")
    print(f"ModelAiter (AITER) geomean time:        {aiter_geomean*1000:.3f} ms")
    print(f"ModelNew (Triton) geomean time:         {new_geomean*1000:.3f} ms")
    print(f"\nSpeedup AITER vs PyTorch:   {orig_geomean/aiter_geomean:.2f}x")
    print(f"Speedup Triton vs PyTorch:  {orig_geomean/new_geomean:.2f}x")
    print(f"Speedup Triton vs AITER:    {aiter_geomean/new_geomean:.2f}x")
    print("=" * 80)

if __name__ == "__main__":
    print("Testing Batched Matrix Multiplication Implementations")
    print("=" * 80)
    print(f"Test shapes (batch, m, k, n):")
    for shape in test_shapes:
        print(f"  {shape}")
    print("=" * 80)
    print()
    
    test_correctness()
    print()
    test_speed()
