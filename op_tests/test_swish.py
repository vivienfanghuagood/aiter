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

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
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


class ModelAgent(nn.Module):
    """
    Optimized model that performs a Swish activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelAgent, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_swish(x)

batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim, dtype=dtypes.fp16, device="cuda")
    return [x]

def get_init_inputs():
    return []

def test_correctness():
    model_orig = Model().cuda()
    model_new = ModelNew().cuda()
    
    inputs = get_inputs()
    x = inputs[0]
    
    with torch.no_grad():
        output_orig = model_orig(x)
        output_new = model_new(x)
    
    checkAllclose(output_orig, output_new, msg="swish")
    print("Correctness test passed!")

def test_speed():
    model_orig = Model().cuda()
    model_new = ModelNew().cuda()
    model_agent = ModelAgent().cuda()
    
    inputs = get_inputs()
    x = inputs[0]
    
    warmup = 10
    iterations = 100
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_orig(x)
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
                _ = model_orig(x)
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
            _ = model_orig(x)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model_orig(x)
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
    print(f"Agent Model (LLM) avg time: {agent_time*1000:.3f} ms")
    print(f"Speedup AITER: {orig_time/new_time:.2f}x")
    print(f"Speedup LLM: {orig_time/agent_time:.2f}x")

if __name__ == "__main__":
    test_correctness()
    test_speed()
