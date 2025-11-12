import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
 
import triton
import triton.language as tl
 
import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose
from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.gemm_a16w16_atomic import gemm_a16w16_atomic
 
 
 
class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
   
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.
 
        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).
 
        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A, B.T)
 
@triton.autotune(
    configs=[
        # Optimized configurations for AMD GPUs (MI300/gfx942, gfx950)
        # AMD GPUs have 64KB LDS (shared memory) per CU
        # Wave size is 64 (equivalent to warp size on NVIDIA)
        
        # Large block configs - better for larger matrices
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        
        # Medium block configs - balanced
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        
        # K-dimension focused configs
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8),
        
        # Higher stage configs for better pipeline
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=6, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=6, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,  # A is [M, K]
    stride_bk, stride_bn,  # B is logically [K, N] (may be transposed view)
    stride_cm, stride_cn,  # C is [M, N]
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    # Map program ids to (pid_m, pid_n)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_id = pid // (GROUP_M * num_pid_n)
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % (GROUP_M * num_pid_n)
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = (pid_in_group // group_size_m)
 
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
 
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
 
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
    # Optimized loop with better memory access patterns
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Check if we're in bounds for K dimension
        k_remaining = K - k * BLOCK_K
        
        # Load with masking only when necessary
        if k_remaining >= BLOCK_K:
            a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
            b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        else:
            k_mask = offs_k < k_remaining
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_mask[None, :]), other=0.0)
            b = tl.load(b_ptrs, mask=(k_mask[:, None]) & (offs_n[None, :] < N), other=0.0)
 
        acc += tl.dot(a, b)
 
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
 
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
 
 
def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B.T where A is (M, K) and B is (N, K) using a Triton GEMM kernel.
    On non-CUDA tensors, falls back to torch.matmul for correctness.
    """
    # Fallback to torch on CPU or other devices
    if not (A.is_cuda and B.is_cuda):
        return torch.matmul(A, B.T)
 
    # Ensure dtype is float32 for this kernel
    # assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 is supported in this kernel."
    assert A.dtype == torch.float16 and B.dtype == torch.float16, "Only float16 is supported in this kernel."
 
    # Make inputs contiguous to simplify stride handling
    Ac = A.contiguous()
    Bc = B.contiguous()
 
    M, K = Ac.shape
    N = Bc.shape[0]  # B is (N, K) but we will treat it as (K, N) via strides
 
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
 
    # Strides in elements
    stride_am, stride_ak = Ac.stride()
    # For B: logical shape we want is (K, N) without materializing transpose.
    # Using original B (N, K), the strides for (K, N) view are (stride along K dim, stride along N dim) = (stride1, stride0).
    stride_bn_orig, stride_bk_orig = Bc.stride()  # for (N, K)
    stride_bk = stride_bk_orig
    stride_bn = stride_bn_orig
 
    stride_cm, stride_cn = C.stride()
 
    # Grid function - autotuning will pick BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )
 
    matmul_kernel[grid](
        Ac, Bc, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    return C
 
 
class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton GEMM kernel to compute C = A @ B.T
    where A is (M, K) and B is (N, K).
    """
    def __init__(self):
        super(ModelNew, self).__init__()
 
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # If on CUDA, use Triton kernel. Otherwise, fallback to torch.matmul for correctness.
        if A.is_cuda and B.is_cuda:
            return triton_matmul(A, B)
        else:
            return torch.matmul(A, B.T)


class ModelAgent(nn.Module):
    """
    Optimized model that performs a matmul using AITER's built-in operations.
    """
    def __init__(self):
        super(ModelAgent, self).__init__()
 
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return gemm_a16w16(x, w, None, x.dtype)
 
 
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2
 
def get_inputs():
    A = torch.rand(M, K, dtype=torch.float16, device="cuda")
    B = torch.rand(N, K, dtype=torch.float16, device="cuda")
    return [A, B]
 
def get_init_inputs():
    return []  # No special initialization inputs needed
 
 
def test_correctness():
    """Test that all three implementations produce the same results."""
    model_orig = Model().cuda()
    model_new = ModelNew().cuda()
    model_agent = ModelAgent().cuda()
   
    inputs = get_inputs()
    x = inputs[0]
    w = inputs[1]
   
    with torch.no_grad():
        output_orig = model_orig(x, w)
        output_new = model_new(x, w)
        output_agent = model_agent(x, w)
   
    checkAllclose(output_orig, output_new, msg="matmul with transposed B (ModelNew)", rtol=1e-2, atol=0.01)
    checkAllclose(output_orig, output_agent, msg="matmul with transposed B (ModelAgent)", rtol=1e-2, atol=0.01)
    print("✓ Correctness test passed for both ModelNew and ModelAgent!")
 
def test_speed():
    """Benchmark the performance of all three implementations."""
    model_orig = Model().cuda()
    model_new = ModelNew().cuda()
    model_agent = ModelAgent().cuda()
   
    inputs = get_inputs()
    x = inputs[0]
    w = inputs[1]
   
    warmup = 10
    iterations = 100
   
    # Benchmark Original Model
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_orig(x, w)
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
                _ = model_orig(x, w)
            torch.cuda.synchronize()
   
    print("=" * 80)
    print("Original Model (PyTorch):")
    print("=" * 80)
    print(prof_orig.key_averages().table(sort_by="cuda_time_total", row_limit=10))
   
    # Benchmark ModelNew (Triton)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_new(x, w)
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
                _ = model_new(x, w)
            torch.cuda.synchronize()
   
    print("\n" + "=" * 80)
    print("ModelNew (Triton):")
    print("=" * 80)
    print(prof_new.key_averages().table(sort_by="cuda_time_total", row_limit=10))
 
    # Benchmark ModelAgent (AITER)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_agent(x, w)
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
                _ = model_agent(x, w)
            torch.cuda.synchronize()
   
    print("\n" + "=" * 80)
    print("ModelAgent (AITER gemm_a16w16):")
    print("=" * 80)
    print(prof_agent.key_averages().table(sort_by="cuda_time_total", row_limit=10))
   
    # Simple timing measurements
    orig_elapsed_gpu_times = []
    for _ in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        _ = model_orig(x, w)
        end_event.record()
        torch.cuda.synchronize()
        orig_elapsed_gpu_times.append(start_event.elapsed_time(end_event))
    orig_time = sum(orig_elapsed_gpu_times) / iterations
 
    new_elapsed_gpu_times = []
    for _ in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        _ = model_new(x, w)
        end_event.record()
        torch.cuda.synchronize()
        new_elapsed_gpu_times.append(start_event.elapsed_time(end_event))
    new_time = sum(new_elapsed_gpu_times) / iterations
 
    agent_elapsed_gpu_times = []
    for _ in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        _ = model_agent(x, w)
        end_event.record()
        torch.cuda.synchronize()
        agent_elapsed_gpu_times.append(start_event.elapsed_time(end_event))
    agent_time = sum(agent_elapsed_gpu_times) / iterations
 
    print("\n" + "=" * 80)
    print("Performance Summary:")
    print("=" * 80)
    print(f"Original Model (PyTorch) avg time:  {orig_time:.3f} ms")
    print(f"ModelNew (Triton) avg time:         {new_time:.3f} ms")
    print(f"ModelAgent (AITER) avg time:        {agent_time:.3f} ms")
    print(f"\nSpeedup Triton vs PyTorch:  {orig_time/new_time:.2f}x")
    print(f"Speedup AITER vs PyTorch:   {orig_time/agent_time:.2f}x")
    print(f"Speedup AITER vs Triton:    {new_time/agent_time:.2f}x")
    print("=" * 80)
 
if __name__ == "__main__":
    print("Testing Matrix Multiplication with Transpose Implementations")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  M:             {M}")
    print(f"  K:             {K}")
    print(f"  N:             {N}")
    print("=" * 80)
    print()
    
    test_correctness()
    print()
    test_speed()