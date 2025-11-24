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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class ModelAiter(nn.Module):
    def __init__(self):
        super(ModelAiter, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigmoid_x = aiter.sigmoid(x)
        return aiter.mul(x, sigmoid_x)

import triton
import triton.language as tl
@triton.jit
def swish_kernel(
    x_ptr,        # *pointer* to input tensor
    out_ptr,      # *pointer* to output tensor
    n_elements,   # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_f32 = x.to(tl.float32)

    # sigmoid(x) = 1 / (1 + exp(-x))
    s = 1.0 / (1.0 + tl.exp(-x_f32))
    y = x_f32 * s

    # Store; Triton will cast to out dtype if needed
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_swish(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA device."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 4096

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a Swish activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_swish(x)

# Test configurations: (batch_size, dim)
# Using common model hidden sizes
test_shapes = [
    (1024, 4096),   # Common hidden size
    (2048, 4096),   # Common hidden size
    (4096, 4096),   # Common hidden size
    (8192, 2048),   # Common hidden size
    (16384, 1024),  # Smaller hidden size
]

def get_inputs(batch_size, dim):
    x = torch.rand(batch_size, dim, dtype=dtypes.fp16, device="cuda")
    return [x]

def get_init_inputs():
    return []

def test_correctness():
    """Test that all three implementations produce the same results."""
    model_orig = Model().cuda()
    model_aiter = ModelAiter().cuda()
    model_new = ModelNew().cuda()
    
    batch_size, dim = test_shapes[0]
    inputs = get_inputs(batch_size, dim)
    x = inputs[0]
    
    with torch.no_grad():
        output_orig = model_orig(x)
        output_new = model_aiter(x)
        output_agent = model_new(x)
    
    checkAllclose(output_orig, output_new, msg="swish (ModelAiter)", rtol=1e-2, atol=0.01)
    checkAllclose(output_orig, output_agent, msg="swish (ModelNew)", rtol=1e-2, atol=0.01)
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
    
    for batch_size, dim in test_shapes:
        print(f"\n{'='*80}")
        print(f"Testing shape: batch={batch_size}, dim={dim}")
        print(f"{'='*80}")
        
        inputs = get_inputs(batch_size, dim)
        x = inputs[0]
        
        # Simple timing measurements
        with torch.no_grad():
            for _ in range(warmup):
                _ = model_orig(x)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(iterations):
                _ = model_orig(x)
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
    print("Testing Swish Activation Implementations")
    print("=" * 80)
    print(f"Test shapes (batch_size, dim):")
    for shape in test_shapes:
        print(f"  {shape}")
    print("=" * 80)
    print()
    
    test_correctness()
    print()
    test_speed()
