# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import aiter.fused_moe as fused_moe_mod
import pytest
import torch

from aiter import ActivationType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, moe_sorting, torch_moe
from aiter.jit.utils.chip_info import get_gfx


pytestmark = pytest.mark.skipif(
    get_gfx() not in {"gfx1200", "gfx1201"},
    reason="RDNA4-only regression coverage",
)


def _moe_sorting_native(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_size: int = 32,
):
    device = topk_ids.device
    token_num, topk = topk_ids.shape
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    sentinel = (topk << 24) | token_num

    sorted_ids = torch.full(
        (max_num_tokens_padded,), sentinel, dtype=dtypes.i32, device=device
    )
    sorted_weights = torch.zeros(
        (max_num_tokens_padded,), dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,), -1, dtype=dtypes.i32, device=device
    )
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)

    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    for expert_id in range(num_experts):
        token_id, topk_id = torch.where(topk_ids == expert_id)
        token_count = token_id.numel()
        expert_block_count = (token_count + block_size - 1) // block_size
        tokens_padded = expert_block_count * block_size
        sorted_ids[sorted_ids_begin : sorted_ids_begin + token_count] = (
            topk_id << 24
        ) | token_id
        sorted_weights[sorted_ids_begin : sorted_ids_begin + token_count] = (
            topk_weights[token_id, topk_id]
        )
        sorted_expert_ids[
            sorted_expert_ids_begin : sorted_expert_ids_begin + expert_block_count
        ] = expert_id
        sorted_ids_begin += tokens_padded
        sorted_expert_ids_begin += expert_block_count

    num_valid_ids[0] = sorted_ids_begin
    num_valid_ids[1] = token_num
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids


@torch.inference_mode()
def test_rdna4_moe_sorting_matches_native():
    torch.manual_seed(0)

    token_num = 31
    num_experts = 256
    topk = 8
    model_dim = 4096
    dtype = torch.bfloat16

    hidden_states = torch.randn((token_num, model_dim), dtype=dtype, device="cuda")
    score = torch.randn((token_num, num_experts), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(hidden_states, score, topk, True)

    ref = _moe_sorting_native(topk_ids, topk_weights, num_experts)
    out = moe_sorting(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        dtype,
        32,
        None,
        None,
        0,
    )

    valid_len = int(ref[3][0].item())
    sentinel = (topk << 24) | token_num
    weight_mask = ref[0] != sentinel
    expert_mask = ref[2] != -1

    assert torch.equal(ref[3].cpu(), out[3].cpu())
    assert torch.equal(ref[0][:valid_len].cpu(), out[0][:valid_len].cpu())
    assert torch.allclose(
        ref[1][weight_mask].cpu(), out[1][weight_mask].cpu(), atol=0, rtol=0
    )
    assert torch.equal(ref[2][expert_mask].cpu(), out[2][expert_mask].cpu())


def test_rdna4_native_sorting_gate():
    assert (
        fused_moe_mod._rdna4_pick_moe_sorting_backend(
            token_num=1,
            num_experts=60,
            topk=4,
            block_size=16,
            moebuf_dtype=dtypes.fp16,
            expert_mask=None,
            num_local_tokens=None,
            dispatch_policy=0,
            use_opus=False,
            mode="auto",
        )
        == "native"
    )
    assert (
        fused_moe_mod._rdna4_pick_moe_sorting_backend(
            token_num=1,
            num_experts=60,
            topk=4,
            block_size=16,
            moebuf_dtype=dtypes.bf16,
            expert_mask=None,
            num_local_tokens=None,
            dispatch_policy=0,
            use_opus=False,
            mode="auto",
        )
        == "fallback"
    )
    assert (
        fused_moe_mod._rdna4_pick_moe_sorting_backend(
            token_num=32,
            num_experts=60,
            topk=4,
            block_size=32,
            moebuf_dtype=dtypes.fp16,
            expert_mask=None,
            num_local_tokens=None,
            dispatch_policy=0,
            use_opus=False,
            mode="auto",
        )
        == "fallback"
    )


@pytest.mark.parametrize(
    ("token_num", "num_experts", "block_size"),
    [
        (1, 60, 16),
        (8, 80, 64),
    ],
)
@torch.inference_mode()
def test_rdna4_native_sorting_matches_rdna4_fallback(
    monkeypatch: pytest.MonkeyPatch,
    token_num: int,
    num_experts: int,
    block_size: int,
):
    torch.manual_seed(0)

    topk = 4
    model_dim = 2048
    dtype = torch.float16

    hidden_states = torch.randn((token_num, model_dim), dtype=dtype, device="cuda")
    score = torch.randn((token_num, num_experts), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(hidden_states, score, topk, True)

    ref = fused_moe_mod._moe_sorting_rdna4_fallback(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        dtype,
        block_size,
        None,
        None,
    )

    monkeypatch.setenv("AITER_RDNA4_MOE_SORTING_BACKEND", "native")
    out = moe_sorting(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        dtype,
        block_size,
        None,
        None,
        0,
    )

    assert torch.equal(ref[0].cpu(), out[0].cpu())
    assert torch.equal(ref[2].cpu(), out[2].cpu())
    assert torch.equal(ref[3].cpu(), out[3].cpu())
    assert torch.allclose(ref[1].cpu(), out[1].cpu(), atol=0, rtol=0)


@torch.inference_mode()
def test_rdna4_moe_sorting_supports_cuda_graph_capture():
    torch.manual_seed(0)

    token_num = 2
    num_experts = 60
    topk = 4
    model_dim = 2048
    block_size = 16
    dtype = torch.float16

    topk_ids = torch.randint(
        0, num_experts, (token_num, topk), dtype=torch.int32, device="cuda"
    )
    topk_weights = torch.rand((token_num, topk), dtype=torch.float32, device="cuda")

    eager = moe_sorting(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        dtype,
        block_size,
        None,
        None,
        0,
    )

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = moe_sorting(
            topk_ids,
            topk_weights,
            num_experts,
            model_dim,
            dtype,
            block_size,
            None,
            None,
            0,
        )
    graph.replay()
    torch.cuda.synchronize()

    valid_len = int(eager[3][0].item())
    expert_len = (valid_len + block_size - 1) // block_size
    assert torch.equal(eager[3].cpu(), captured[3].cpu())
    assert torch.equal(eager[0][:valid_len].cpu(), captured[0][:valid_len].cpu())
    assert torch.allclose(
        eager[1][:valid_len].cpu(), captured[1][:valid_len].cpu(), atol=0, rtol=0
    )
    assert torch.equal(
        eager[2][:expert_len].cpu(),
        captured[2][:expert_len].cpu(),
    )


@torch.inference_mode()
def test_rdna4_fused_moe_native_sorting_decode_regression(
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(0)
    monkeypatch.setenv("AITER_RDNA4_MOE_SORTING_BACKEND", "native")

    token_num = 1
    num_experts = 60
    topk = 4
    model_dim = 2048
    inter_dim = 1408
    dtype = torch.float16

    hidden_states = torch.randn((token_num, model_dim), dtype=dtype, device="cuda")
    w1 = torch.randn(
        (num_experts, inter_dim * 2, model_dim), dtype=dtype, device="cuda"
    ) / 10
    w2 = torch.randn(
        (num_experts, model_dim, inter_dim), dtype=dtype, device="cuda"
    ) / 10
    score = torch.randn((token_num, num_experts), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(hidden_states, score, topk, True)

    ref = torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=ActivationType.Silu,
    )
    out = fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=ActivationType.Silu,
    )

    assert torch.allclose(ref, out, atol=0.125, rtol=1e-2)


@pytest.mark.parametrize(
    ("token_num", "num_experts", "topk", "model_dim", "inter_dim", "atol", "rtol"),
    [
        (2048, 32, 2, 5120, 1536, 1.0, 1e-2),
        (2048, 32, 2, 7168, 2048, 2.0, 2e-2),
    ],
)
@torch.inference_mode()
def test_rdna4_fused_moe_regression(
    token_num: int,
    num_experts: int,
    topk: int,
    model_dim: int,
    inter_dim: int,
    atol: float,
    rtol: float,
):
    torch.manual_seed(0)
    dtype = torch.float16

    hidden_states = torch.randn((token_num, model_dim), dtype=dtype, device="cuda")
    w1 = torch.randn(
        (num_experts, inter_dim * 2, model_dim), dtype=dtype, device="cuda"
    ) / 10
    w2 = torch.randn(
        (num_experts, model_dim, inter_dim), dtype=dtype, device="cuda"
    ) / 10
    score = torch.randn((token_num, num_experts), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(hidden_states, score, topk, True)

    ref = torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=ActivationType.Silu,
    )
    out = fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=ActivationType.Silu,
    )

    assert torch.allclose(ref, out, atol=atol, rtol=rtol)
