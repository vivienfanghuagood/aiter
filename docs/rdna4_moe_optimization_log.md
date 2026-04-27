# RDNA4 MoE integration and optimization log

This note records the RDNA4 standard-weight MoE bring-up work across AITER,
FlyDSL, and vLLM. It intentionally keeps both accepted and rejected
experiments so later work can reproduce the good results and avoid redoing
paths that already failed end-to-end.

## Scope

Target:
- expose a FlyDSL-backed RDNA4 MoE path through AITER
- let vLLM select that path as a proper ROCm AITER backend on RDNA4
- preserve numerical correctness
- improve decode performance relative to the initial RDNA4 bring-up

Current status:
- correctness is in good shape for the validated fp16 path
- the accepted RDNA4 path is still behind Triton on the standard vLLM decode
  benchmark, so more kernel-body work is still needed

## Repositories and branches

The coordinated bring-up was split across three repositories:
- `AITER`: `vivienfanghuagood/aiter`, branch `fhq/rdna4-moe`
- `vLLM`: `vivienfanghuagood/vllm`, branch `fhq/rdna4-moe`
- `FlyDSL`: `vivienfanghuagood/FlyDSL`, branch `fhq/rdna4-moe`

This log lives in AITER because that repo contains the main RDNA4 fused-MoE
dispatch logic and the regression tests.

## Files that matter in AITER

Accepted implementation lives primarily in:
- `aiter/fused_moe.py`
- `aiter/ops/moe_op.py`
- `aiter/ops/topk.py`
- `aiter/ops/flydsl/moe_kernels.py`
- `aiter/ops/flydsl/kernels/moe_gemm_2stage.py`
- `aiter/ops/flydsl/kernels/rdna_moe_gemm_2stage.py`
- `aiter/ops/flydsl/kernels/rdna_moe_gemm_2stage_common.py`
- `op_tests/test_rdna4_moe_backend.py`

## Environment used during bring-up

Working layout:
- `/work/aiter`
- `/work/vllm`
- `/work/FlyDSL`

Python environment:
- `/tmp/vllm-rocm-std`

Model used for vLLM decode validation:
- `/tmp/hf-cache/Qwen1.5-MoE-A2.7B`

Common environment flags:

```bash
export PYTHONPATH=/work/vllm:/work/aiter
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export AITER_RDNA4_MOE_SORTING_BACKEND=auto
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
```

## Accepted changes

### 1. RDNA4 standard MoE kernel path

A dedicated RDNA4 fp16/bf16 standard-weight MoE builder was added through:
- `rdna_moe_gemm_2stage.py`
- `rdna_moe_gemm_2stage_common.py`

This is separate from the gfx1250 path and is selected only for gfx120x
standard MoE shapes.

### 2. RDNA4 topk and grouped-topk fallback

RDNA4 does not use the existing HIP topk path directly. The accepted solution
is a conservative torch fallback for:
- `topk_softmax`
- `topk_sigmoid`
- `grouped_topk`
- `biased_grouped_topk`

This keeps the routing side correct while the expert kernels use the new RDNA4
path.

### 3. RDNA4 native sorting with a conservative safety gate

The accepted sorting policy is:
- default mode: `AITER_RDNA4_MOE_SORTING_BACKEND=auto`
- use native sorting only for shapes that were validated as numerically safe
- fall back to the graph-safe Triton-aligned path otherwise

The current `auto` gate only enables native sorting when all of the following
are true:
- `moebuf_dtype == fp16`
- `token_num <= 16`
- `num_experts <= 128`
- `topk <= 4`
- `block_size in {16, 32, 64, 128}`
- `expert_mask is None`
- `num_local_tokens is None`
- `dispatch_policy == 0`
- OPUS sorting is disabled

Important:
- `bf16` is intentionally excluded from the native auto path
- validation showed systematic bf16 mismatches on small decode shapes

### 4. Graph-safe sorting output contract

For the RDNA4 path, the sorting outputs are prefilled with:
- sentinel token ids
- zero weights
- `-1` expert ids

That contract matters because stage1/stage2 scan padded buffers and the graph
capture path must avoid host-side reads from `num_valid_ids`.

### 5. Stage2 temporary allocation fix

For `flydsl_moe_stage2(..., mode="reduce")`, the temporary reduction target is
allocated with `torch.empty(...)` instead of `torch.zeros(...)`.

This change was kept because the reduce kernel writes the full temporary buffer
and the extra zero-fill only added overhead.

## Validation commands

### AITER regression tests

```bash
PYTHONPATH=/work/aiter \
  /tmp/vllm-rocm-std/bin/python -m pytest -q \
  /work/aiter/op_tests/test_rdna4_moe_backend.py
```

### vLLM integration tests

```bash
PYTHONPATH=/work/vllm \
  /tmp/vllm-rocm-std/bin/python -m pytest --noconftest -q \
  /work/vllm/tests/test_unquantized_moe_backend_fallback.py \
  /work/vllm/tests/test_aiter_ops_registration.py \
  /work/vllm/tests/test_rocm_platform_detection.py
```

### Standard vLLM decode benchmark

This note used a local-only vLLM harness that was intentionally not committed.
To reproduce the same class of measurement, keep the following workload shape
stable in your own local harness:
- model: `Qwen1.5-MoE-A2.7B`
- dtype: `fp16`
- `max_model_len=64`
- `max_num_seqs=1`
- prefix caching disabled
- decode-oriented single-request run
- repeated timed runs over the same prompt/output shape

If you still have the local harness from the original bring-up, the exact
command looked like this:

```bash
AITER_RDNA4_MOE_SORTING_BACKEND=auto \
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_MOE=1 \
TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
PYTHONPATH=/work/vllm:/work/aiter \
  /tmp/vllm-rocm-std/bin/python /work/vllm/tmp_bench_stage1_ab.py
```

## Performance timeline

Measured on the standard vLLM decode benchmark with
`Qwen1.5-MoE-A2.7B`:

- Triton baseline: about `65.02 tok/s`
- early AITER RDNA4 path with poor sorting behavior: about `44.89 tok/s`
- accepted AITER RDNA4 path after sorting and temp-buffer fixes:
  about `58.45 tok/s`

Expected noise on repeated runs is non-zero. Later reruns of the same accepted
tree were typically in the `58.0` to `58.5 tok/s` range.

## Optimization chronology

### Accepted: native sorting for safe fp16 decode shapes

What changed:
- added a native RDNA4 sorting path
- added a graph-safe fallback path
- gated native sorting conservatively

Why it stayed:
- large improvement over the initial fallback-heavy path
- correctness stayed within the validated fp16 envelope

Why the gate is conservative:
- bf16 mismatches were too large to leave enabled by default

### Accepted: remove useless zero-fill in stage2 reduce temp

What changed:
- stage2 reduce temporary buffer switched from `zeros` to `empty`

Why it stayed:
- the buffer is fully overwritten by the kernel
- it improved the hot path without changing numerics

### Rejected: direct cached stage1/stage2 compiled wrappers

What changed:
- bypassed the generic FlyDSL stage1/stage2 wrapper path
- called cached compiled executables directly

Why it looked promising:
- standalone microbench improved
- stage1 wrapper overhead dropped
- stage2 wrapper overhead dropped

Why it was rejected:
- vLLM decode throughput regressed from about `58.50 tok/s` to
  about `58.00 tok/s`
- the complexity increase was not justified by the real workload

Action taken:
- reverted from the committed path

### Rejected: global stage2 tile and waves tuning

What changed:
- tried `stage2 tile_n=64`
- tried stage2 waves and related launch variants

Why it looked promising:
- some eager microbench cases improved slightly

Why it was rejected:
- standard vLLM decode dropped to about `58.05 tok/s`
- no stable end-to-end gain

Action taken:
- reverted from the committed path

### Rejected: shape-aware stage1 heuristic for `M=1`

What changed:
- tried `stage1 tile_n=64, waves_per_eu=2` only for decode-like `M=1`

Why it looked promising:
- eager `M=1` microbench improved

Why it was rejected:
- eager improvement did not survive graph replay
- a direct CUDAGraph replay test of `fused_moe` at `M=1` regressed from about
  `0.0881 ms` to about `0.0909 ms`
- standard vLLM decode dropped to about `57.96 tok/s`

Action taken:
- reverted from the committed path

## Main pitfalls

### Do not trust eager microbench alone

This was the biggest trap during bring-up.

Observed failure mode:
- an eager `M=1` microbench can show an improvement
- the same change can be neutral or worse under CUDAGraph replay
- the same change can still be worse in full vLLM decode

Practical rule:
- never keep an RDNA4 MoE optimization based only on eager timing
- require at least:
  - correctness
  - eager microbench
  - CUDAGraph replay microbench
  - standard vLLM decode benchmark

### Do not enable bf16 native sorting by default

Even when the shape is small and decode-like, the native sorting path showed
systematic bf16 mismatches. The safe default is still fallback for bf16.

### Do not batch multiple tuning ideas into one change

The best workflow for this path was:
- change one point
- benchmark
- keep only if the end-to-end result improves
- otherwise revert immediately

This keeps the tree readable and makes regressions easy to attribute.

### Do not commit local research harnesses

The local research flow used temporary scripts such as:
- `/work/aiter/tmp_rdna4_moe_research.py`
- `/work/vllm/tmp_bench_stage1_ab.py`

Those were intentionally kept out of git. Keep future ad-hoc harnesses local
unless they are cleaned up into a durable benchmark utility.

## Recommended reproduction workflow

When changing the RDNA4 MoE path again, use this order:

1. Run the AITER RDNA4 regression tests.
2. Run the vLLM RDNA4 integration tests.
3. Measure eager standalone behavior only as a coarse filter.
4. Measure CUDAGraph replay for the exact `fused_moe` shape you are trying to
   improve.
5. Run the standard vLLM decode benchmark.
6. Keep the change only if step 5 is better.

## Where the next work should focus

After the accepted sorting fixes, sorting stopped being the main bottleneck.

The next likely productive directions are:
- stage1 kernel body efficiency under graph replay
- stage2 kernel body efficiency under graph replay
- launch behavior that specifically matches the graph-captured decode path

The least productive directions so far were:
- wrapper-level refactors
- broad env-override tuning
- eager-only microbench optimization

## Summary

The accepted RDNA4 path is the result of keeping only the changes that survived
real vLLM decode measurement:
- conservative native sorting for safe fp16 decode shapes
- graph-safe fallback sorting for everything else
- RDNA4 standard-weight FlyDSL MoE builder integration
- stage2 temp-buffer zero-fill removal

Anything beyond that should be treated as experimental until it wins under
CUDAGraph replay and the standard vLLM decode benchmark.
