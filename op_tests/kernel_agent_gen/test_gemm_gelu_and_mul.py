import torch
import torch.nn as nn
import torch.nn.functional as F
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose
from torch.profiler import profile, ProfilerActivity
import time
import triton
import triton.language as tl
from aiter.ops.triton.gemm_a16w16_gated import gemm_a16w16_gated


class Model(nn.Module):
    """
    Original PyTorch implementation using separate Linear, GELU, and multiplication.
    Input shape: [batch_size, in_features]
    Output shape: [batch_size, out_features]
    
    This performs: output = GELU(x @ W1) * (x @ W2)
    where W1, W2 are [in_features, out_features]
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features, bias=False)
        self.fc2 = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        # x: [batch_size, in_features]
        gate = self.fc1(x)  # [batch_size, out_features]
        up = self.fc2(x)    # [batch_size, out_features]
        return F.gelu(gate) * up


class ModelAiter(nn.Module):
    """
    Optimized implementation using AITER's gemm_a16w16_gated (fused GEMM+GELU+MUL).
    Uses Triton's gated GEMM kernel for best performance.
    """
    def __init__(self, in_features, out_features):
        super(ModelAiter, self).__init__()
        # Fused weight: [2*out_features, in_features] for gated gemm
        # First half for gate, second half for up projection
        self.weight = nn.Parameter(
            torch.randn(2 * out_features, in_features, dtype=dtypes.fp16)
        )
        self.out_features = out_features
    
    def forward(self, x):
        # x: [batch_size, in_features], dtype fp16
        # gemm_a16w16_gated: performs X @ W^T with gating
        # Returns GELU(X @ W[:N//2, :]^T) * (X @ W[N//2:, :]^T)
        return gemm_a16w16_gated(x, self.weight, dtype=dtypes.fp16, activation="gelu")
    
    def load_from_original(self, original_model):
        """Load weights from original model."""
        # fc1.weight is [out_features, in_features] for gate
        # fc2.weight is [out_features, in_features] for up
        w1 = original_model.fc1.weight.data.to(dtypes.fp16)  # [out_features, in_features]
        w2 = original_model.fc2.weight.data.to(dtypes.fp16)  # [out_features, in_features]
        # Concatenate along first dimension for gated gemm
        self.weight.data = torch.cat([w1, w2], dim=0)  # [2*out_features, in_features]


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        # AMD GPU optimized configs with smaller shared memory usage
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16, 'GROUP_M': 4}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_gelu_mul_kernel(
    a_ptr, w_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Fused matrix multiplication with GELU and element-wise multiplication.
    Computes: C = GELU(A @ W[:, :N]) * (A @ W[:, N:])
    
    A: [M, K]
    W: [K, 2N] (concatenated weights)
    C: [M, N]
    """
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers for A and W (gate part)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w_gate_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    
    # Accumulator for gate projection (will apply GELU)
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute A @ W[:, :N] for gate
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
        w_gate = tl.load(w_gate_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
        acc_gate += tl.dot(a, w_gate)
        a_ptrs += BLOCK_K * stride_ak
        w_gate_ptrs += BLOCK_K * stride_wk
    
    # Reset pointers for up projection
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # W[:, N:] starts at column N
    w_up_ptrs = w_ptr + N * stride_wn + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    
    # Accumulator for up projection
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute A @ W[:, N:] for up
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
        w_up = tl.load(w_up_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
        acc_up += tl.dot(a, w_up)
        a_ptrs += BLOCK_K * stride_ak
        w_up_ptrs += BLOCK_K * stride_wk
    
    # Apply GELU to gate: GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    # Using tanh approximation for better performance:
    # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2/pi)
    # gate_gelu = acc_gate * 0.5 * (1.0 + tl.libdevice.tanh(
    #     SQRT_2_OVER_PI * (acc_gate + 0.044715 * acc_gate * acc_gate * acc_gate)
    # ))

    gate_gelu =  0.5 * acc_gate * (1.0 + tl.erf(acc_gate * 0.70710678118654752440))
    
    # Element-wise multiplication: GELU(gate) * up
    result = gate_gelu * acc_up
    
    # Convert to fp16 and store
    c = result.to(tl.float16)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul_gelu_mul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Fused matrix multiplication with GELU and multiplication using Triton.
    
    Args:
        x: [M, K] input tensor (fp16)
        weight: [K, 2*N] weight tensor (fp16), concatenated [W_gate | W_up]
    
    Returns:
        output: [M, N] output tensor (fp16)
    """
    assert x.is_contiguous() and weight.is_contiguous()
    assert x.dtype == torch.float16 and weight.dtype == torch.float16
    
    M, K = x.shape
    K_w, N2 = weight.shape
    assert K == K_w, f"Shape mismatch: x has K={K}, weight has K={K_w}"
    assert N2 % 2 == 0, f"Weight second dimension must be even, got {N2}"
    N = N2 // 2
    
    output = torch.empty((M, N), dtype=torch.float16, device=x.device)
    
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )
    
    matmul_gelu_mul_kernel[grid](
        x, weight, output,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
    )
    
    return output


class ModelNew(nn.Module):
    """
    Triton-optimized implementation with fused GEMM + GELU + Mul.
    Optimized for AMD GPUs with consideration of shared memory limits.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # Fused weight: [in_features, 2 * out_features]
        self.weight = nn.Parameter(
            torch.randn(in_features, 2 * out_features, dtype=torch.float16, device="cuda")
        )
        self.out_features = out_features

    def forward(self, x):
        """
        Args:
            x: [batch_size, in_features], dtype float16, device cuda
        Returns:
            output: [batch_size, out_features], dtype float16
        """
        return triton_matmul_gelu_mul(x, self.weight)
    
    def load_from_original(self, original_model):
        """Load weights from original model."""
        w1 = original_model.fc1.weight.data.t().to(dtypes.fp16)  # [in_features, out_features]
        w2 = original_model.fc2.weight.data.t().to(dtypes.fp16)  # [in_features, out_features]
        self.weight.data = torch.cat([w1, w2], dim=1)  # [in_features, 2*out_features]


# Test configurations: (batch_size, in_features, out_features)
# Using larger shapes for better Triton performance
test_shapes = [
    (1024, 4096, 4096),
    (2048, 4096, 4096),
    (4096, 2048, 4096),
    (8192, 2048, 2048),
    (16384, 1024, 2048),
]


def get_inputs(batch_size, in_features):
    return [torch.rand(batch_size, in_features, dtype=dtypes.fp16, device="cuda")]


def get_init_inputs(in_features, out_features):
    return [in_features, out_features]


def test_correctness():
    """Test that all three implementations produce the same results."""
    batch_size, in_features, out_features = test_shapes[0]
    init_inputs = get_init_inputs(in_features, out_features)
    model_orig = Model(*init_inputs).cuda()
    model_aiter = ModelAiter(*init_inputs).cuda()
    model_new = ModelNew(*init_inputs).cuda()
    
    # Load weights from original model
    model_aiter.load_from_original(model_orig)
    model_new.load_from_original(model_orig)
    
    inputs = get_inputs(batch_size, in_features)
    x = inputs[0]
    
    with torch.no_grad():
        # Original model uses fp32 internally, convert output to fp16 for comparison
        output_orig = model_orig(x.to(dtypes.fp32)).to(dtypes.fp16)
        output_new = model_aiter(x)
        output_agent = model_new(x)
    
    checkAllclose(output_orig, output_new, msg="gelu_and_mul (ModelAiter)", rtol=1e-2, atol=0.01)
    checkAllclose(output_orig, output_agent, msg="gelu_and_mul (ModelNew)", rtol=1e-2, atol=0.01)
    print("✓ Correctness test passed for both ModelAiter and ModelNew!")


def test_speed():
    """Benchmark the performance of all three implementations across multiple shapes."""
    import numpy as np
    
    warmup = 10
    iterations = 100
    
    orig_times = []
    aiter_times = []
    new_times = []
    
    for batch_size, in_features, out_features in test_shapes:
        print(f"\n{'='*80}")
        print(f"Testing shape: batch={batch_size}, in_features={in_features}, out_features={out_features}")
        print(f"{'='*80}")
        
        init_inputs = get_init_inputs(in_features, out_features)
        model_orig = Model(*init_inputs).cuda()
        model_aiter = ModelAiter(*init_inputs).cuda()
        model_new = ModelNew(*init_inputs).cuda()
        
        model_aiter.load_from_original(model_orig)
        model_new.load_from_original(model_orig)
        
        inputs = get_inputs(batch_size, in_features)
        x = inputs[0]
        
        # Simple timing measurements
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
                _ = model_aiter(x)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(iterations):
                _ = model_aiter(x)
            torch.cuda.synchronize()
            aiter_time = (time.time() - start) / iterations
        
        with torch.no_grad():
            for _ in range(warmup):
                _ = model_new(x)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(iterations):
                _ = model_new(x)
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
    print("Testing GELU-and-Mul Implementations")
    print("=" * 80)
    print(f"Test shapes (batch_size, in_features, out_features):")
    for shape in test_shapes:
        print(f"  {shape}")
    print("=" * 80)
    print()
    
    test_correctness()
    print()
    test_speed()
