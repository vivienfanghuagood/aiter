# AITER Operators Usage in vLLM

This document provides a comprehensive reference for AITER library operators used in vLLM, including detailed Python interfaces and usage examples.

---

## Overview

vLLM integrates AITER operators through the `vllm/_aiter_ops.py` module, with environment variable controls for enabling specific operators. All AITER operators are exclusively used on **AMD ROCm** platforms.

---

## 1. Paged Attention Operators

### 1.1 Use Cases
- **Decode stage** KV Cache management
- Support for FP8/INT8 quantized KV Cache

### 1.2 Python Interface

#### Write to KV Cache
```python
# vllm/attention/ops/rocm_aiter_paged_attn.py
from aiter import reshape_and_cache_with_pertoken_quant

rocm_aiter.reshape_and_cache_with_pertoken_quant(
    key,           # torch.Tensor: Key tensor
    value,         # torch.Tensor: Value tensor 
    key_cache,     # torch.Tensor: Key cache buffer (FP8/INT8)
    value_cache,   # torch.Tensor: Value cache buffer
    k_scale,       # torch.Tensor: Key quantization scale
    v_scale,       # torch.Tensor: Value quantization scale
    slot_mapping,  # torch.Tensor: Slot mapping indices
    True,          # bool: enable per-token quantization
)
```

#### Decode Forward Pass
```python
# vllm/attention/ops/rocm_aiter_paged_attn.py
from aiter import pa_fwd_asm

output = torch.empty_like(query)
rocm_aiter.pa_fwd_asm(
    query,                    # torch.Tensor: [num_seqs, num_heads, head_dim]
    key_cache,                # torch.Tensor: Paged key cache
    value_cache,              # torch.Tensor: Paged value cache
    block_tables,             # torch.Tensor: Block table indices
    seq_lens,                 # torch.Tensor: Sequence lengths
    max_num_blocks_per_seq,   # int: Max blocks per sequence
    k_scale,                  # torch.Tensor: K quantization scale
    v_scale,                  # torch.Tensor: V quantization scale
    output,                   # torch.Tensor: Output buffer
)
```

### 1.3 Environment Variables
```bash
VLLM_ROCM_USE_AITER_PAGED_ATTN=1  # Enable Paged Attention
```

### 1.4 Features
- ✅ Memory-efficient KV cache management
- ✅ Dynamic sequence length support
- ✅ FP8/INT8 quantization support
- ✅ Per-token quantization

---

## 2. Flash Attention / MHA Operators

### 2.1 Use Cases
- **Prefill stage** (processing prompts)
- **Training** mode

### 2.2 Python Interface

#### Standard Flash Attention
```python
# vllm/v1/attention/backends/rocm_aiter_fa.py
from aiter import flash_attn_varlen_func

output = flash_attn_varlen_func(
    q,                # torch.Tensor: Query [total_q, num_heads, head_dim]
    k,                # torch.Tensor: Key
    v,                # torch.Tensor: Value
    cu_seqlens_q,     # torch.Tensor: Cumulative sequence lengths for q
    cu_seqlens_k,     # torch.Tensor: Cumulative sequence lengths for k
    max_seqlen_q,     # int: Max sequence length for q
    max_seqlen_k,     # int: Max sequence length for k
    dropout_p=0.0,    # float: Dropout probability
    softmax_scale=None, # float: Attention scale (default: 1/sqrt(d))
    causal=False,     # bool: Apply causal mask
    window_size=(-1, -1), # tuple: Sliding window size
    alibi_slopes=None,# torch.Tensor: ALiBi slopes
    return_attn_probs=False,
)
```

### 2.3 Environment Variables
```bash
VLLM_ROCM_USE_AITER_MHA=1  # Enable MHA/Flash Attention
```

### 2.4 Features
- ✅ Variable-length sequence support
- ✅ Causal masking
- ✅ Sliding window attention
- ✅ ALiBi position bias
- ✅ FP16/BF16/FP8 support

---

## 3. MLA (Multi-Head Latent Attention) Operators

### 3.1 Use Cases
- **DeepSeek-V2/V3** models
- Models with latent KV cache compression

### 3.2 Python Interface

#### MLA Decode Forward
```python
# vllm/_aiter_ops.py
torch.ops.vllm.rocm_aiter_mla_decode_fwd(
    q,                  # torch.Tensor: Query
    kv_buffer,          # torch.Tensor: Latent KV buffer
    o,                  # torch.Tensor: Output buffer (in-place)
    qo_indptr,          # torch.Tensor: Query/output indptr
    max_seqlen_qo,      # int: Max sequence length
    kv_indptr=None,     # torch.Tensor: KV indptr (optional)
    kv_indices=None,    # torch.Tensor: KV page indices (optional)
    kv_last_page_lens=None, # torch.Tensor: Last page lengths
    sm_scale=1.0,       # float: Softmax scale
    logit_cap=0.0,      # float: Logit capping
)
```

#### Direct Call from aiter.mla
```python
# vllm/v1/attention/backends/mla/rocm_aiter_mla.py
from aiter.mla import mla_decode_fwd

mla_decode_fwd(
    q,
    kv_buffer.view(-1, 1, 1, q.shape[-1]),
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_qo,
    sm_scale=sm_scale,
    logit_cap=logit_cap,
)
```

### 3.3 Environment Variables
```bash
VLLM_ROCM_USE_AITER_MLA=1  # Enable MLA
```

### 3.4 Supported Backends
- `ROCM_AITER_MLA` - ROCm AITER MLA
- `ROCM_AITER_TRITON_MLA` - Triton-based MLA
- `ROCM_AITER_MLA_SPARSE` - Sparse MLA variant

---

## 4. MoE (Mixture of Experts) Operators

### 4.1 Use Cases
- **Mixtral**, **DeepSeek-MoE**, **Qwen-MoE** models
- FP16/BF16/FP8 weight support

### 4.2 Python Interface

#### Fused MoE
```python
# vllm/_aiter_ops.py
from aiter import ActivationType, QuantType, fused_moe

output = torch.ops.vllm.rocm_aiter_fused_moe(
    hidden_states,     # torch.Tensor: [num_tokens, hidden_size]
    w1,                # torch.Tensor: Expert W1 weights [num_experts, ...]
    w2,                # torch.Tensor: Expert W2 weights
    topk_weight,       # torch.Tensor: TopK weights [num_tokens, topk]
    topk_ids,          # torch.Tensor: TopK expert IDs
    expert_mask=None,  # torch.Tensor: Expert mask (optional)
    activation_method=0, # int: 0=SILU, 1=GELU, 2=ReLU
    quant_method=0,    # int: QuantType enum (0=NO, 1=INT8, 2=FP8...)
    doweight_stage1=False,
    w1_scale=None,     # torch.Tensor: W1 quantization scale
    w2_scale=None,     # torch.Tensor: W2 quantization scale
    a1_scale=None,     # torch.Tensor: Activation1 scale
    a2_scale=None,     # torch.Tensor: Activation2 scale
)
```

#### ASM MoE with TopK Weighting
```python
# vllm/_aiter_ops.py
from aiter.fused_moe_bf16_asm import asm_moe_tkw1

output = torch.ops.vllm.rocm_aiter_asm_moe_tkw1(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    fc1_scale=None,
    fc2_scale=None,
    fc1_smooth_scale=None,
    fc2_smooth_scale=None,
    a16=False,         # bool: Use FP16 activation
    per_tensor_quant_scale=None,
    expert_mask=None,
    activation_method=0,
)
```

#### TopK Softmax (Expert Routing)
```python
# vllm/_aiter_ops.py
from aiter import topk_softmax

torch.ops.vllm.rocm_aiter_topk_softmax(
    topk_weights,          # torch.Tensor: Output weights (in-place)
    topk_indices,          # torch.Tensor: Output indices (in-place)
    token_expert_indices,  # torch.Tensor: Token-expert mapping (in-place)
    gating_output,         # torch.Tensor: Router logits
    renormalize,           # bool: Renormalize weights
)
```

#### Grouped TopK (Group-wise Routing)
```python
# vllm/_aiter_ops.py
from aiter import grouped_topk

torch.ops.vllm.rocm_aiter_grouped_topk(
    gating_output,         # torch.Tensor: Router logits
    topk_weights,          # torch.Tensor: Output weights (in-place)
    topk_ids,              # torch.Tensor: Output expert IDs (in-place)
    num_expert_group,      # int: Number of expert groups
    topk_group,            # int: TopK per group
    need_renorm,           # bool: Renormalize weights
    scoring_func="softmax", # str: "softmax" or "sigmoid"
    routed_scaling_factor=1.0,
)
```

#### Biased Grouped TopK
```python
# vllm/_aiter_ops.py
from aiter import biased_grouped_topk

torch.ops.vllm.rocm_aiter_biased_grouped_topk(
    gating_output,
    correction_bias,       # torch.Tensor: Bias correction
    topk_weights,
    topk_ids,
    num_expert_group,
    topk_group,
    need_renorm,
    routed_scaling_factor=1.0,
)
```

### 4.3 Environment Variables
```bash
VLLM_ROCM_USE_AITER_MOE=1                 # Enable MoE
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 # Enable shared expert fusion
```

### 4.4 Activation Types
- `0` - SILU (Swish)
- `1` - GELU
- `2` - ReLU

### 4.5 Quantization Types
- `0` - NO (no quantization)
- `1` - INT8
- `2` - FP8
- Additional types defined in `QuantType` enum

---

## 5. RoPE (Rotary Position Embedding) Operators

### 5.1 Use Cases
- All Transformer models using RoPE
- Llama, Qwen, DeepSeek, etc.

### 5.2 Python Interface

```python
# vllm/_aiter_ops.py
from aiter.ops.triton.rope import rope_cached_thd_positions_2c_fwd_inplace

# ROCm AITER Triton RoPE implementation
num_tokens = positions.numel()
cos, sin = cos_sin_cache.chunk(2, dim=-1)
query_shape = query.shape
key_shape = key.shape
rotate_style = 0 if is_neox_style else 1  # 0=neox, 1=llama

query = query.view(num_tokens, -1, head_size)
key = key.view(num_tokens, -1, head_size)
query_ = query[..., :rotary_dim]
key_ = key[..., :rotary_dim]
positions = positions.view(*query.shape[:1])

rope_cached_thd_positions_2c_fwd_inplace(
    positions,             # torch.Tensor: Position IDs
    sin,                   # torch.Tensor: Sin cache
    cos,                   # torch.Tensor: Cos cache
    query_,                # torch.Tensor: Query (in-place, rotary dims only)
    key_,                  # torch.Tensor: Key (in-place)
    rotate_style,          # int: 0=neox_style, 1=llama_style
    reuse_freqs_front_part=True,
    is_nope_first=False,
)

query = query.view(query_shape)
key = key.view(key_shape)
```

### 5.3 Environment Variables
```bash
VLLM_ROCM_USE_AITER_TRITON_ROPE=1  # Enable Triton RoPE
```

### 5.4 Rotation Styles
- `0` - **NeoX style**: Rotate pairs across feature dimension
- `1` - **LLaMA style**: Rotate pairs within first/second half

---

## 6. Quantization Operators

### 6.1 FP8 Quantization

#### Group FP8 Quantization
```python
# vllm/_aiter_ops.py
from aiter import QuantType, get_hip_quant

aiter_per1x128_quant = get_hip_quant(QuantType.per_1x128)
x_fp8, out_bs = torch.ops.vllm.rocm_aiter_group_fp8_quant(
    input_2d,      # torch.Tensor: Input tensor [M, N]
    group_size=128 # int: Quantization group size (must be 128)
)
# Returns:
# x_fp8: torch.Tensor [M, N] in FP8
# out_bs: torch.Tensor [M, N//128] block scales
```

#### FP8 GEMM (INT8 Weight, INT8 Activation)
```python
# vllm/_aiter_ops.py
from aiter import gemm_a8w8_CK

Y = torch.ops.vllm.rocm_aiter_gemm_a8w8(
    A,             # torch.Tensor: Activation [M, K] in INT8
    B,             # torch.Tensor: Weight [N, K] in INT8
    As,            # torch.Tensor: A scales
    Bs,            # torch.Tensor: B scales
    bias=None,     # torch.Tensor: Bias (optional)
    output_dtype=torch.float16,
)
```

#### FP8 GEMM with Block Scale
```python
# vllm/_aiter_ops.py
from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale

Y = torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
    A,                      # torch.Tensor: [M, K]
    B,                      # torch.Tensor: [N, K]
    As,                     # torch.Tensor: A block scales
    Bs,                     # torch.Tensor: B block scales
    output_dtype=torch.float16,
)
```

#### FP8 Batched GEMM
```python
# vllm/_aiter_ops.py
from aiter.ops.triton.batched_gemm_a8w8_... import batched_gemm_a8w8_...

Y = rocm_aiter_ops.triton_fp8_bmm(
    X,              # torch.Tensor: Batch of activations
    WQ,             # torch.Tensor: Quantized weights
    w_scale,        # torch.Tensor: Weight scales
    group_size=128,
    bias=None,
    dtype=torch.bfloat16,
    splitK=None,
    YQ=None,
    transpose_bm=False,
    config=None,
)
```

### 6.2 FP4 Dynamic Quantization GEMM
```python
# vllm/_aiter_ops.py
from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.quant import dynamic_mxfp4_quant

y = rocm_aiter_ops.triton_fp4_gemm_dynamic_qaunt(
    x,                  # torch.Tensor: Input
    weight,             # torch.Tensor: FP4 quantized weight
    weight_scale,       # torch.Tensor: Weight scales
    out_dtype=torch.bfloat16,
    x_scales=None,      # torch.Tensor: Pre-computed x scales (optional)
)
```

### 6.3 Environment Variables
```bash
VLLM_ROCM_USE_AITER_LINEAR=1             # Enable quantized Linear
VLLM_ROCM_USE_AITER_FP8BMM=1             # Enable FP8 BatchedGEMM
VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1       # Enable FP4 ASM GEMM
VLLM_ROCM_USE_AITER_TRITON_GEMM=1        # Enable Triton GEMM
```

### 6.4 Quantization Granularities
- **Per-tensor**: Single scale for entire tensor
- **Per-channel**: One scale per output channel
- **Per-token**: One scale per token (activation)
- **Per-group** (1x128): Block-wise quantization

---

## 7. Normalization Operators

### 7.1 RMSNorm

#### Standard RMSNorm
```python
# vllm/_aiter_ops.py
from aiter import rms_norm

output = torch.ops.vllm.rocm_aiter_rms_norm(
    x,                  # torch.Tensor: Input tensor
    weight,             # torch.Tensor: RMS weight
    variance_epsilon,   # float: Epsilon for numerical stability
)
```

#### RMSNorm with Residual Add
```python
# vllm/_aiter_ops.py
from aiter import rmsnorm2d_fwd_with_add

output, residual_out = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
    x,                  # torch.Tensor: Input
    residual,           # torch.Tensor: Residual to add
    weight,             # torch.Tensor: RMS weight
    variance_epsilon,   # float: Epsilon
)
```

### 7.2 Environment Variables
```bash
VLLM_ROCM_USE_AITER_RMSNORM=1  # Enable RMSNorm
```

### 7.3 Formula
RMSNorm computes: `x * weight / sqrt(mean(x^2) + epsilon)`

---

## 8. Unified Attention Operators

### 8.1 Use Cases
- Unified handling of Prefill + Decode in single kernel
- Reduces kernel launch overhead

### 8.2 Python Interface
```python
# vllm/v1/attention/backends/rocm_aiter_unified_attn.py
from aiter.ops.triton.unified_attention import unified_attention

output = unified_attention(
    q,                  # torch.Tensor: Query
    k,                  # torch.Tensor: Key  
    v,                  # torch.Tensor: Value
    o,                  # torch.Tensor: Output buffer
    cu_seqlens_q,       # torch.Tensor: Cumulative Q sequence lengths
    cu_seqlens_k,       # torch.Tensor: Cumulative KV sequence lengths
    max_seqlen_q,       # int: Max Q sequence length
    max_seqlen_k,       # int: Max KV sequence length
    sm_scale,           # float: Softmax scale
    # Additional parameters...
)
```

### 8.3 Environment Variables
```bash
VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1  # Enable Unified Attention
```

### 8.4 Advantages
- ✅ Single kernel for both prefill and decode
- ✅ Reduced kernel launch overhead
- ✅ Better GPU utilization for mixed batches

---

## 9. Sampling Operators

### 9.1 TopK/TopP Sampling
```python
# vllm/v1/sample/ops/topk_topp_sampler.py
from aiter import top_p_top_k

# Efficient TopK/TopP sampling using AITER
# Used for token generation in decode stage
```

### 9.2 Features
- ✅ Fused TopK and TopP operations
-  Optimized for ROCm hardware
- ✅ Low latency sampling

---

## 10. Auxiliary Operators

### 10.1 Weight Shuffle (Memory Layout Optimization)
```python
# vllm/_aiter_ops.py
from aiter.ops.shuffle import shuffle_weight

shuffled_weight = rocm_aiter_ops.shuffle_weight(
    tensor,             # torch.Tensor: Weight tensor
    layout=(16, 16)     # tuple: Block layout for shuffling
)

# Batch shuffle multiple weights
shuffled_tensors = rocm_aiter_ops.shuffle_weights(
    *tensors,           # Variable number of tensors
    layout=(16, 16)
)
```

### 10.2 Purpose
Weight shuffling rearranges memory layout to optimize cache access patterns and improve GEMM performance on AMD GPUs.

---

## Environment Variables Reference

### Master Switch
```bash
VLLM_ROCM_USE_AITER=1  # Master switch - must be enabled first
```

### Module-Specific Switches
```bash
# Attention Modules
VLLM_ROCM_USE_AITER_MHA=1                # Flash Attention / MHA
VLLM_ROCM_USE_AITER_PAGED_ATTN=1         # Paged Attention
VLLM_ROCM_USE_AITER_MLA=1                # Multi-Head Latent Attention
VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1  # Unified Attention

# Linear/GEMM Modules
VLLM_ROCM_USE_AITER_LINEAR=1             # Quantized Linear layers
VLLM_ROCM_USE_AITER_FP8BMM=1             # FP8 Batched GEMM
VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1       # FP4 Assembly GEMM
VLLM_ROCM_USE_AITER_TRITON_GEMM=1        # Triton-based GEMM

# MoE Modules
VLLM_ROCM_USE_AITER_MOE=1                       # Fused MoE operators
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1     # Shared expert fusion

# Other Modules
VLLM_ROCM_USE_AITER_RMSNORM=1            # RMSNorm operators
VLLM_ROCM_USE_AITER_TRITON_ROPE=1        # Triton-based RoPE
```

---

## Supported Models

AITER operators are used in the following models:

### Language Models
- **Llama** (all versions): RoPE, Flash Attn, RMSNorm, Paged Attn
- **Qwen** (all versions): RoPE, Flash Attn, MoE
- **DeepSeek-V2/V3**: MLA, MoE, RoPE, all optimizations
- **Mixtral**: MoE, RoPE, Flash Attn
- **GPT-NeoX**: RoPE (NeoX style), Flash Attn
- **Yi**: RoPE, Flash Attn

### MoE Models
- **Mixtral-8x7B/8x22B**
- **DeepSeek-MoE**
- **Qwen-MoE**
- **DBRX**

### All ROCm-supported Transformer Models
Any model running on AMD ROCm can potentially benefit from AITER operators.

---

## Performance Optimization Guide

### 1. Hardware-Specific Recommendations

#### AMD MI300 Series
```bash
# Enable all AITER operators for best performance
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_ROCM_USE_AITER_PAGED_ATTN=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_FP8BMM=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
```

#### AMD MI250 Series
```bash
# Enable core operators
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_ROCM_USE_AITER_PAGED_ATTN=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
```

### 2. Model-Specific Recommendations

#### Large Language Models (70B+)
- ✅ Enable Paged Attention for memory efficiency
- ✅ Use FP8 quantization on MI300
- ✅ Enable unified attention for mixed batches

#### MoE Models
- ✅ Enable `AITER_MOE` for significant speedup
- ✅ Use `FUSION_SHARED_EXPERTS` if applicable
- ✅ Enable grouped TopK for better routing

#### Long Context (>32K tokens)
- ✅ Paged Attention is essential
- ✅ Consider FP8 KV cache quantization
- ✅ Use unified attention for efficiency

### 3. Quantization Strategy

#### FP8 Quantization (Recommended for MI300)
```bash
export VLLM_ROCM_USE_AITER_FP8BMM=1
export VLLM_ROCM_USE_AITER_LINEAR=1
```
- **Benefits**: 2x memory reduction, 1.5-2x throughput increase
- **Trade-off**: Minimal accuracy loss (<0.5% perplexity increase)

#### FP4 Quantization (Maximum compression)
```bash
export VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1
```
- **Benefits**: 4x memory reduction
- **Trade-off**: More accuracy loss, use with calibration

### 4. Batch Size Tuning

- **Small batches (1-8)**: Enable all optimizations, latency-sensitive
- **Large batches (16+)**: Focus on throughput optimizations
- **Mixed batches**: Use unified attention

---

## Debugging and Troubleshooting

### Enable Debug Logging
```bash
export VLLM_LOGGING_LEVEL=DEBUG
```

### Common Issues

#### 1. AITER not found
```
Error: AITER library not found
```
**Solution**: Install AITER library
```bash
pip install aiter
```

#### 2. Operator not supported on architecture
```
Warning: AITER operator not supported on gfx1xxx
```
**Solution**: AITER operators require gfx9 architectures (MI250/MI300). Disable AITER on unsupported hardware.

#### 3. Quantization errors
```
Error: Dimension not divisible by group size
```
**Solution**: Ensure input dimensions are compatible with quantization group size (typically 128).

---

## Architecture Support

### Supported AMD GPU Architectures
- ✅ **gfx90a** (MI250X)
- ✅ **gfx942** (MI300A/X)
- ❌ **gfx1xxx** (RDNA, not supported)

### Check Your Architecture
```bash
rocminfo | grep gfx
```

---

## API Reference Summary

### Attention Operations
| Operator | Function | Purpose |
|----------|----------|---------|
| `flash_attn_varlen_func` | Variable-length Flash Attention | Prefill stage |
| `pa_fwd_asm` | Paged Attention forward | Decode stage |
| `mla_decode_fwd` | MLA decode forward | DeepSeek models |
| `unified_attention` | Unified prefill+decode | Mixed batches |

### MoE Operations
| Operator | Function | Purpose |
|----------|----------|---------|
| `fused_moe` | Fused MoE computation | Expert forward pass |
| `topk_softmax` | TopK with softmax | Expert routing |
| `grouped_topk` | Grouped TopK routing | Multi-group routing |
| `biased_grouped_topk` | Biased grouped routing | Load balancing |

### Quantization Operations
| Operator | Function | Purpose |
|----------|----------|---------|
| `group_fp8_quant` | FP8 group quantization | Activation quantization |
| `gemm_a8w8_CK` | INT8 GEMM | Quantized matmul |
| `gemm_a8w8_blockscale` | Block-scale GEMM | Fine-grained quant |
| `triton_fp4_gemm_dynamic_qaunt` | FP4 dynamic GEMM | Extreme compression |

### Other Operations
| Operator | Function | Purpose |
|----------|----------|---------|
| `rope_cached_thd_positions_2c_fwd_inplace` | RoPE forward | Position encoding |
| `rms_norm` | RMSNorm forward | Layer normalization |
| `shuffle_weight` | Weight shuffling | Memory optimization |

---

## Contributing

When adding new AITER operators to vLLM:

1. **Add operator wrapper** in `vllm/_aiter_ops.py`
2. **Register custom op** using `direct_register_custom_op`
3. **Add environment variable** control
4. **Update documentation** with interface and usage
5. **Add tests** in `tests/kernels/`

### Example Template
```python
def _rocm_aiter_new_op_impl(
    input: torch.Tensor,
    param: int,
) -> torch.Tensor:
    from aiter import new_operator
    return new_operator(input, param)

def _rocm_aiter_new_op_fake(
    input: torch.Tensor,
    param: int,
) -> torch.Tensor:
    return torch.empty_like(input)

# Register operator
direct_register_custom_op(
    "rocm_aiter_new_op",
    _rocm_aiter_new_op_impl,
    mutates_args=[],
    fake_impl=_rocm_aiter_new_op_fake,
)
```

---

## Summary

AITER provides a comprehensive library of optimized operators for AMD ROCm:

### Coverage
- ✅ **Attention**: Paged Attn, Flash Attn, MLA, Unified Attn
- ✅ **MoE**: Fused MoE, TopK routing, Grouped routing
- ✅ **GEMM**: FP8, FP4, Block-scale quantization
- ✅ **Normalization**: RMSNorm, LayerNorm with residual
-  **Position Encoding**: RoPE (Triton-optimized)
- ✅ **Sampling**: TopK/TopP sampling

### Key Benefits
- 🚀 **Performance**: 1.5-3x speedup on AMD hardware
- 💾 **Memory**: 50-75% KV cache reduction with Paged Attention
- 🎯 **Accuracy**: Minimal loss with FP8 quantization (<0.5%)
- 🔧 **Flexibility**: Environment variable controls for fine-tuning

### Integration
All operators are seamlessly integrated into vLLM through the `_aiter_ops.py` module with automatic fallback to default implementations when not available or disabled.

---

## License

AITER operators are part of the vLLM project and follow the same Apache 2.0 license.

## References

- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [AITER GitHub Repository](https://github.com/ROCm/aiter)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD Instinct™ Accelerators](https://www.amd.com/en/products/accelerators/instinct.html)

---

**Last Updated**: 2024-11-21
