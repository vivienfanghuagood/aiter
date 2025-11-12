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


class ModelNew(nn.Module):
    """
    Optimized implementation using AITER's gelu_and_mul.
    Input shape: [batch_size, 2 * out_features]
    Output shape: [batch_size, out_features]
    """
    def __init__(self, out_features):
        super(ModelNew, self).__init__()
        self.out_features = out_features
    
    def forward(self, x):
        # x: [batch_size, 2 * out_features], dtype fp16
        # Apply gelu_and_mul: GELU(x[:, :out_features]) * x[:, out_features:]
        out = torch.empty(x.shape[0], self.out_features, dtype=dtypes.fp16, device=x.device)
        aiter.gelu_and_mul(out, x)
        return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=5, num_warps=2),
        # AMD GPU optimized configs with smaller blocks
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def gelu_and_mul_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused GELU and element-wise multiplication kernel.
    Computes: output = GELU(input[:, :N]) * input[:, N:]
    
    input: [M, 2*N]
    output: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load gate part (first half): input[:, :N]
    gate_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Load up part (second half): input[:, N:]
    up_ptrs = input_ptr + offs_m[:, None] * stride_im + (offs_n[None, :] + N) * stride_in
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Apply GELU: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    # Using M_SQRT1_2 = 1 / sqrt(2) = 0.70710678118654752440
    M_SQRT1_2 = 0.70710678118654752440
    gate_gelu = 0.5 * gate * (1.0 + tl.erf(gate * M_SQRT1_2))
    
    # Element-wise multiplication: GELU(gate) * up
    result = gate_gelu * up
    
    # Store result
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_ptrs, result.to(tl.float16), mask=mask)


def triton_gelu_and_mul(input: torch.Tensor) -> torch.Tensor:
    """
    Apply GELU and element-wise multiplication using Triton.
    
    Args:
        input: [M, 2*N] input tensor (fp16)
    
    Returns:
        output: [M, N] output tensor (fp16)
    """
    assert input.is_contiguous()
    assert input.dtype == torch.float16
    
    M, N2 = input.shape
    assert N2 % 2 == 0, f"Input second dimension must be even, got {N2}"
    N = N2 // 2
    
    output = torch.empty((M, N), dtype=torch.float16, device=input.device)
    
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )
    
    gelu_and_mul_kernel[grid](
        input, output,
        M, N,
        input.stride(0), input.stride(1),
        output.stride(0), output.stride(1),
    )
    
    return output


class ModelAgent(nn.Module):
    """
    Triton-optimized implementation with fused GELU + Mul.
    Optimized for AMD GPUs with consideration of shared memory limits.
    Input shape: [batch_size, 2 * out_features]
    Output shape: [batch_size, out_features]
    """
    def __init__(self, out_features):
        super(ModelAgent, self).__init__()
        self.out_features = out_features

    def forward(self, x):
        """
        Args:
            x: [batch_size, 2 * out_features], dtype float16, device cuda
        Returns:
            output: [batch_size, out_features], dtype float16
        """
        return triton_gelu_and_mul(x)

@triton.jit
def gelu_mul_kernel(
    x_ptr,          # *ptr to input [B, 2D]
    out_ptr,        # *ptr to output [B, D]
    B,              # batch size
    D,              # out_features
    stride_x,       # stride between rows in x (in elements)
    stride_out,     # stride between rows in out (in elements)
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    # column offsets for this tile
    col_offset = pid_n * BLOCK_N
    offs_n = col_offset + tl.arange(0, BLOCK_N)
    # mask within [0, D)
    mask = offs_n < D

    # base row pointers
    row_x = x_ptr + pid_b * stride_x
    row_out = out_ptr + pid_b * stride_out

    # load gate and up halves
    gate = tl.load(row_x + offs_n, mask=mask, other=0.0)
    up = tl.load(row_x + D + offs_n, mask=mask, other=0.0)

    # cast to f32 for computation
    gate_f32 = gate.to(tl.float32)
    up_f32 = up.to(tl.float32)

    # GELU approximation: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    c0 = 0.044715
    c1 = 0.7978845608028654  # sqrt(2/pi)
    x3 = gate_f32 * gate_f32 * gate_f32
    t = c1 * (gate_f32 + c0 * x3)

    # tanh(t) = (exp(2t) - 1) / (exp(2t) + 1)
    e2t = tl.exp(2.0 * t)
    tanh_t = (e2t - 1.0) / (e2t + 1.0)

    gelu = 0.5 * gate_f32 * (1.0 + tanh_t)

    out_vals = gelu * up_f32
    out_vals = out_vals.to(tl.float16)

    tl.store(row_out + offs_n, out_vals, mask=mask)


def triton_gelu_mul(x: torch.Tensor, out_features: int):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    B = x.shape[0]
    D = out_features
    assert x.shape[1] == 2 * D, "Input second dimension must be 2 * out_features."

    out = torch.empty((B, D), device=x.device, dtype=torch.float16)

    stride_x = x.stride(0)
    stride_out = out.stride(0)

    BLOCK_N = 256
    grid = (B, triton.cdiv(D, BLOCK_N))

    gelu_mul_kernel[grid](x, out, B, D, stride_x, stride_out, BLOCK_N=BLOCK_N)
    return out


class ModelAgentNew(nn.Module):
    def __init__(self, out_features):
        super(ModelAgentNew, self).__init__()
        self.out_features = out_features

    def forward(self, x):
        return triton_gelu_mul(x, self.out_features)


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
    model_new = ModelNew(*init_inputs).cuda()
    model_agent = ModelAgent(*init_inputs).cuda()
    
    inputs = get_inputs()
    x = inputs[0]
    
    with torch.no_grad():
        # Original model uses fp32 internally, convert output to fp16 for comparison
        output_orig = model_orig(x.to(dtypes.fp32)).to(dtypes.fp16)
        output_new = model_new(x)
        output_agent = model_agent(x)
    
    checkAllclose(output_orig, output_new, msg="gelu_and_mul (ModelNew)", rtol=1e-2, atol=0.01)
    checkAllclose(output_orig, output_agent, msg="gelu_and_mul (ModelAgent)", rtol=1e-2, atol=0.01)
    print("✓ Correctness test passed for both ModelNew and ModelAgent!")


def test_speed():
    """Benchmark the performance of all three implementations."""
    init_inputs = get_init_inputs()
    model_orig = Model(*init_inputs).cuda()
    model_new = ModelNew(*init_inputs).cuda()
    model_agent = ModelAgent(*init_inputs).cuda()
    
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
    print("ModelNew (AITER gelu_and_mul):")
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
