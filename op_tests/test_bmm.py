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

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        B_transposed = B.transpose(1, 2)
        return aiter.batched_gemm_bf16_CK(A, B_transposed, bias=None, dtype=dtypes.bf16)

import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'num_warps': 8, 'num_stages': 3}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'num_warps': 4, 'num_stages': 3}),
    ],
    key=['M','N','K'],
)
@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    BATCH, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_batch = tl.program_id(1)
    pid = tl.program_id(0)

    # Number of blocks
    num_blk_m = tl.cdiv(M, BLOCK_M)
    num_blk_n = tl.cdiv(N, BLOCK_N)
    num_blks = num_blk_m * num_blk_n

    # Grouped ordering to improve L2 locality
    group_id = pid // (GROUP_M * num_blk_n)
    first_m = group_id * GROUP_M
    group_size_m = min(num_blk_m - first_m, GROUP_M)
    pid_in_group = pid % (group_size_m * num_blk_n)
    blk_m = first_m + (pid_in_group // num_blk_n)
    blk_n = pid_in_group % num_blk_n

    offs_m = blk_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = blk_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_batch_ptr = A_ptr + pid_batch * stride_ab
    B_batch_ptr = B_ptr + pid_batch * stride_bb
    C_batch_ptr = C_ptr + pid_batch * stride_cb

    a_ptrs = A_batch_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_batch_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        k_remaining = K - k_iter
        k_mask = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_iter += BLOCK_K

    # Write back
    c_ptrs = C_batch_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # assert A.dim() == 3 and B.dim() == 3, ""A and B must be 3D tensors (B, M, K) and (B, K, N)""
    # assert A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1], ""Shapes must be (B,M,K) x (B,K,N)""
    # If not on CUDA, fallback to torch.bmm
    if not (A.is_cuda and B.is_cuda):
        return torch.bmm(A, B)

    BATCH, M, K = A.shape
    _, _, N = B.shape

    # Make contiguous for simpler strides
    A_c = A.contiguous()
    B_c = B.contiguous()
    C = torch.empty((BATCH, M, N), device=A.device, dtype=torch.float32)

    # Extract strides in elements
    stride_ab, stride_am, stride_ak = A_c.stride()
    stride_bb, stride_bk, stride_bn = B_c.stride()
    stride_cb, stride_cm, stride_cn = C.stride()

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_M = 8
    num_warps = 8
    num_stages = 4


    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (grid_m * grid_n, BATCH)



    bmm_kernel[grid](
        A_c, B_c, C,
        BATCH, M, N, K,
        stride_ab, stride_am, stride_ak,
        stride_bb, stride_bk, stride_bn,
        stride_cb, stride_cm, stride_cn,
        GROUP_M=GROUP_M,
    )
    return C


class ModelAgent(nn.Module):
    # """"""
    # Optimized batched matrix multiplication using a custom Triton kernel.
    # """"""
    def __init__(self):
        super(ModelAgent, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_bmm(A, B)

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_inputs():
    A = torch.rand(batch_size, m, k, dtype=dtypes.bf16, device="cuda")
    B = torch.rand(batch_size, k, n, dtype=dtypes.bf16, device="cuda")
    return [A, B]

def get_init_inputs():
    return []

def test_correctness():
    model_orig = Model().cuda()
    model_new = ModelNew().cuda()
    
    inputs = get_inputs()
    A, B = inputs
    
    with torch.no_grad():
        output_orig = model_orig(A, B)
        output_new = model_new(A, B)
    
    checkAllclose(output_orig, output_new, msg="batched_gemm", rtol=1e-2, atol=0.01)
    print("Correctness test passed!")

def test_speed():
    model_orig = Model().cuda()
    model_new = ModelNew().cuda()
    model_agent = ModelAgent().cuda()
    
    inputs = get_inputs()
    A, B = inputs
    
    warmup = 10
    iterations = 100
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_orig(A, B)
        torch.cuda.synchronize()
        
    # with torch.no_grad():
    #     with profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         profile_memory=True,
    #         with_stack=True,
    #         with_modules=True,
    #         record_shapes=True,
    #     ) as prof_orig:
    #         for _ in range(iterations):
    #             _ = model_orig(A, B)
    #         torch.cuda.synchronize()
    
    print("Original Model:")
    print(prof_orig.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_new(A, B)
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
                _ = model_new(A, B)
            torch.cuda.synchronize()
    
    print("New Model (AITER):")
    print(prof_new.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
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
            _ = model_new(A, B)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model_new(A, B)
        torch.cuda.synchronize()
        new_time = (time.time() - start) / iterations
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_agent(A, B)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            _ = model_agent(A, B)
        torch.cuda.synchronize()
        agent_time = (time.time() - start) / iterations
    
    print(f"\nOriginal Model avg time: {orig_time*1000:.3f} ms")
    print(f"New Model (AITER) avg time: {new_time*1000:.3f} ms")
    print(f"NewNew Model (LLM) avg time: {agent_time*1000:.3f} ms")
    print(f"Speedup AITER: {orig_time/new_time:.2f}x")
    print(f"Speedup LLM: {orig_time/agent_time:.2f}x")

if __name__ == "__main__":
    test_correctness()
    test_speed()
