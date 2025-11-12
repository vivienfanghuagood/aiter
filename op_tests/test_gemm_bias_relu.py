import torch
import torch.nn as nn
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose
from torch.profiler import profile, ProfilerActivity
import time
from aiter.ops.triton.gemm_a16w16 import gemm_a16w16

class Model(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.gemm(x)
        x = x + self.bias
        x = torch.relu(x)
        return x

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=dtypes.fp16))
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=dtypes.fp16))
    
    def forward(self, x):
        return gemm_a16w16(x, self.weight, bias=self.bias, dtype=dtypes.fp16, activation="relu")
    
    def load_from_original(self, original_model):
        self.weight.data.copy_(original_model.gemm.weight.data.to(dtypes.fp16))
        self.bias.data.copy_(original_model.bias.data.to(dtypes.fp16))


import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_relu_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    a_stride_m,
    a_stride_k,
    b_stride_k,
    b_stride_n,
    c_stride_m,
    c_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Number of program blocks along M and N axes
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Block starting offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers to blocks
    a_ptrs = a_ptr + (offs_m[:, None] * a_stride_m + offs_k[None, :] * a_stride_k)
    b_ptrs = b_ptr + (offs_k[:, None] * b_stride_k + offs_n[None, :] * b_stride_n)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop
    for k in range(0, K, BLOCK_K):
        # Load A and B blocks
        a_mask = (offs_m[:, None] < M) & ((k + offs_k[None, :]) < K)
        b_mask = ((k + offs_k[:, None]) < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Matrix multiplication
        acc += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_K * a_stride_k
        b_ptrs += BLOCK_K * b_stride_k
    
    # Load bias and add
    bias_ptrs = bias_ptr + offs_n
    bias_mask = offs_n < N
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    acc += bias[None, :]
    
    # Apply ReLU
    acc = tl.maximum(acc, 0.0)
    
    # Convert to fp16 and store
    c = acc.to(tl.float16)
    
    # Store result
    c_ptrs = c_ptr + (offs_m[:, None] * c_stride_m + offs_n[None, :] * c_stride_n)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_linear_bias_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Compute (x @ weight + bias).relu() using Triton
    
    Args:
        x: [M, K] input tensor
        weight: [K, N] weight tensor
        bias: [N] bias tensor
    
    Returns:
        output: [M, N] output tensor
    """
    assert x.is_contiguous(), "x must be contiguous"
    assert weight.is_contiguous(), "weight must be contiguous"
    assert bias.is_contiguous(), "bias must be contiguous"
    assert x.dtype == torch.float16 and weight.dtype == torch.float16 and bias.dtype == torch.float16
    
    M, K = x.shape
    K_w, N = weight.shape
    assert K == K_w, f"Dimension mismatch: x has K={K}, weight has K={K_w}"
    assert bias.shape[0] == N, f"Bias shape mismatch: expected [{N}], got {bias.shape}"
    
    # Allocate output
    output = torch.empty((M, N), dtype=torch.float16, device=x.device)
    
    # Grid configuration
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )
    
    # Launch kernel
    matmul_bias_relu_kernel[grid](
        x, weight, bias, output,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
    )
    
    return output

class ModelAgent(nn.Module):
    """
    Optimized model that performs fused matrix multiplication + bias add + ReLU using Triton.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelAgent, self).__init__()
        # Store weight in (in_features, out_features) for A[M,K] @ W[K,N]
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features, dtype=torch.float16, device="cuda") * 0.02
        )
        self.bias = nn.Parameter(
            torch.randn(bias_shape, dtype=torch.float16, device="cuda")
        )

    def forward(self, x):
        """
        x: [batch_size, in_features], dtype float16, device cuda
        returns: [batch_size, out_features]
        """
        return triton_linear_bias_relu(x, self.weight, self.bias)
    
    def load_from_original(self, original_model):
        # Original model weight is [out_features, in_features]
        # We need [in_features, out_features] for A @ W
        self.weight.data.copy_(original_model.gemm.weight.data.t().to(dtypes.fp16))
        self.bias.data.copy_(original_model.bias.data.to(dtypes.fp16))

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features, dtype=dtypes.fp16, device="cuda")]

def get_init_inputs():
    return [in_features, out_features, bias_shape]

def test_correctness():
    """Test that all three implementations produce the same results."""
    init_inputs = get_init_inputs()
    model_orig = Model(*init_inputs).cuda()
    model_new = ModelNew(*init_inputs).cuda()
    model_new.load_from_original(model_orig)
    model_agent = ModelAgent(*init_inputs).cuda()
    model_agent.load_from_original(model_orig)
    
    inputs = get_inputs()
    x = inputs[0]
    
    with torch.no_grad():
        output_orig = model_orig(x.to(dtypes.fp32)).to(dtypes.fp16)
        output_new = model_new(x)
        output_agent = model_agent(x)
    
    checkAllclose(output_orig, output_new, msg="gemm_bias_relu (ModelNew)", rtol=1e-2, atol=0.01)
    checkAllclose(output_orig, output_agent, msg="gemm_bias_relu (ModelAgent)", rtol=1e-2, atol=0.01)
    print("✓ Correctness test passed for both ModelNew and ModelAgent!")

def test_speed():
    """Benchmark the performance of all three implementations."""
    init_inputs = get_init_inputs()
    model_orig = Model(*init_inputs).cuda()
    model_new = ModelNew(*init_inputs).cuda()
    model_new.load_from_original(model_orig)
    model_agent = ModelAgent(*init_inputs).cuda()
    model_agent.load_from_original(model_orig)
    
    inputs = get_inputs()
    x = inputs[0]
    
    warmup = 10
    iterations = 100
    
    # Benchmark Original Model
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
    
    print("=" * 80)
    print("Original Model (PyTorch):")
    print("=" * 80)
    print(prof_orig.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Benchmark ModelNew (AITER)
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
    
    print("\n" + "=" * 80)
    print("ModelNew (AITER gemm_a16w16):")
    print("=" * 80)
    print(prof_new.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Benchmark ModelAgent (Triton)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_agent(x)
        torch.cuda.synchronize()
        
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            with_stack=True,
            with_modules=True,
            record_shapes=True,
        ) as prof_agent:
            for _ in range(iterations):
                _ = model_agent(x)
            torch.cuda.synchronize()
    
    print("\n" + "=" * 80)
    print("ModelAgent (Triton Fused):")
    print("=" * 80)
    print(prof_agent.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
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
    
    print("\n" + "=" * 80)
    print("Performance Summary:")
    print("=" * 80)
    print(f"Original Model (PyTorch) avg time:  {orig_time*1000:.3f} ms")
    print(f"ModelNew (AITER) avg time:          {new_time*1000:.3f} ms")
    print(f"ModelAgent (Triton) avg time:       {agent_time*1000:.3f} ms")
    print(f"\nSpeedup AITER vs PyTorch:   {orig_time/new_time:.2f}x")
    print(f"Speedup Triton vs PyTorch:  {orig_time/agent_time:.2f}x")
    print(f"Speedup Triton vs AITER:    {new_time/agent_time:.2f}x")
    print("=" * 80)

if __name__ == "__main__":
    print("Testing GEMM + Bias + ReLU Implementations")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Batch size:    {batch_size}")
    print(f"  In features:   {in_features}")
    print(f"  Out features:  {out_features}")
    print("=" * 80)
    print()
    
    test_correctness()
    print()
    test_speed()
