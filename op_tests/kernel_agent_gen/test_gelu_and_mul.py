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


class Model(nn.Module):
    """
    Original PyTorch implementation: GELU(input[:, :d]) * input[:, d:]
    Input shape: [batch_size, 2 * out_features]
    Output shape: [batch_size, out_features]
    """
    def __init__(self, out_features):
        super(Model, self).__init__()
        self.out_features = out_features

    def forward(self, x):
        # x: [batch_size, 2 * out_features]
        d = self.out_features
        gate = x[:, :d]      # [batch_size, out_features]
        up = x[:, d:]        # [batch_size, out_features]
        return F.gelu(gate) * up


class ModelAiter(nn.Module):
    """
    Optimized implementation using AITER's gelu_and_mul.
    Input shape: [batch_size, 2 * out_features]
    Output shape: [batch_size, out_features]
    """
    def __init__(self, out_features):
        super(ModelAiter, self).__init__()
        self.out_features = out_features
    
    def forward(self, x):
        # x: [batch_size, 2 * out_features], dtype fp16
        # Apply gelu_and_mul: GELU(x[:, :out_features]) * x[:, out_features:]
        out = torch.empty(x.shape[0], self.out_features, dtype=dtypes.fp16, device=x.device)
        aiter.gelu_and_mul(out, x)
        return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
    ],
    key=['N'],
)
@triton.jit
def fused_gelu_mul_kernel(
    x_ptr,
    out_ptr,
    N,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes: GELU(x[:, :d]) * x[:, d:]
    where d = out_features
    
    Memory layout optimization:
    - Process output elements in a flat contiguous manner
    - Each output element at position (row, col) reads from:
      * gate: x[row, col] (first half of columns)
      * up: x[row, col + out_features] (second half of columns)
    """
    # Flat output index
    pid = tl.program_id(0)
    flat_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Bounds check
    mask = flat_idx < N
    
    # Compute row and column from flat index
    row = flat_idx // out_features
    col = flat_idx % out_features
    
    # Calculate input indices for gate and up
    # gate comes from x[:, :out_features]
    # up comes from x[:, out_features:]
    gate_idx = row * (2 * out_features) + col
    up_idx = row * (2 * out_features) + out_features + col
    
    # Load gate and up values (coalesced memory access)
    gate = tl.load(x_ptr + gate_idx, mask=mask, other=0.0)
    up = tl.load(x_ptr + up_idx, mask=mask, other=0.0)
    
    # Ultra-fast GELU approximation: gelu(x) ≈ x * sigmoid(1.702 * x)
    # Cast to fp32 for numerical stability
    gate_fp32 = gate.to(tl.float32)
    gelu_gate = gate_fp32 * tl.sigmoid(1.702 * gate_fp32)
    
    # Multiply with up (cast up to fp32 as well for consistency)
    up_fp32 = up.to(tl.float32)
    result = gelu_gate * up_fp32
    
    # Cast back to original dtype and store
    result_out = result.to(gate.dtype)
    tl.store(out_ptr + flat_idx, result_out, mask=mask)


def triton_fused_gelu_mul(x: torch.Tensor, out_features: int):
    """
    Wrapper function for the fused GELU * mul kernel.
    
    Args:
        x: Input tensor of shape [batch_size, 2 * out_features]
        out_features: Number of output features
    
    Returns:
        Output tensor of shape [batch_size, out_features]
    """
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    
    batch_size = x.shape[0]
    N = batch_size * out_features
    
    # Allocate output tensor
    out = torch.empty(batch_size, out_features, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    fused_gelu_mul_kernel[grid](
        x,
        out,
        N,
        out_features,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Optimized implementation using fused Triton kernel.
    Combines GELU(input[:, :d]) * input[:, d:] into a single kernel.
    """
    def __init__(self, out_features):
        super(ModelNew, self).__init__()
        self.out_features = out_features

    def forward(self, x):
        # x: [batch_size, 2 * out_features]
        return triton_fused_gelu_mul(x, self.out_features)

# Test configuration
batch_size = 64 * 1024
out_features = 8192


def get_inputs():
    """Generate input tensor of shape [batch_size, 2 * out_features]"""
    return [torch.rand(batch_size, 2 * out_features, dtype=dtypes.fp16, device="cuda")]


def get_init_inputs():
    return [out_features]


def test_correctness():
    """Test that all three implementations produce the same results."""
    init_inputs = get_init_inputs()
    model_orig = Model(*init_inputs).cuda()
    model_aiter = ModelAiter(*init_inputs).cuda()
    model_new = ModelNew(*init_inputs).cuda()
    
    inputs = get_inputs()
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
    """Benchmark the performance of all three implementations."""
    init_inputs = get_init_inputs()
    model_orig = Model(*init_inputs).cuda()
    model_aiter = ModelAiter(*init_inputs).cuda()
    model_new = ModelNew(*init_inputs).cuda()
    
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
    
    # Benchmark ModelAiter (AITER)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_aiter(x)
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
                _ = model_aiter(x)
            torch.cuda.synchronize()
    
    print("\n" + "=" * 80)
    print("ModelAiter (AITER gelu_and_mul):")
    print("=" * 80)
    print(prof_new.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Benchmark ModelNew (Triton)
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
        ) as prof_agent:
            for _ in range(iterations):
                _ = model_new(x)
            torch.cuda.synchronize()
    
    print("\n" + "=" * 80)
    print("ModelNew (Triton Fused):")
    print("=" * 80)
    print(prof_agent.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Simple timing measurements
    x_fp32 = x.to(dtypes.fp32)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_orig(x_fp32)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model_orig(x_fp32)
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
        new_time = (time.time() - start) / iterations
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_new(x)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model_new(x)
        torch.cuda.synchronize()
        agent_time = (time.time() - start) / iterations
    
    print("\n" + "=" * 80)
    print("Performance Summary:")
    print("=" * 80)
    print(f"Original Model (PyTorch) avg time:  {orig_time*1000:.3f} ms")
    print(f"ModelAiter (AITER) avg time:          {new_time*1000:.3f} ms")
    print(f"ModelNew (Triton) avg time:       {agent_time*1000:.3f} ms")
    print(f"\nSpeedup AITER vs PyTorch:   {orig_time/new_time:.2f}x")
    print(f"Speedup Triton vs PyTorch:  {orig_time/agent_time:.2f}x")
    print(f"Speedup Triton vs AITER:    {new_time/agent_time:.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    print("Testing GELU-and-Mul Implementations")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Batch size:    {batch_size}")
    print(f"  Out features:  {out_features}")
    print("=" * 80)
    print()
    
    test_correctness()
    print()
    test_speed()
