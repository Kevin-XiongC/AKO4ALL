# Custom Benchmark for Fused QK-Norm + RoPE + FP8-Cast + KV-Store

## Usage

```bash
python bench/bench.py --solution solution/kernel.py --ref input/reference.py --verbose
```

## Target Shape

GLM4 TP8 decode configuration:
- `head_dim = 128`
- `num_heads_q = 12, num_heads_k = 1, num_heads_v = 1`
- `partial_rotary_factor = 0.5` → `rotary_dim = 64`
- NeoX-style RoPE
- FP8 E4M3 quantization with `k_scale = v_scale = 1.0`
- Primary batch: 32 tokens

## Output Format

```
COMPILED: True/False
CORRECT: True/False
RUNTIME: <ms>         # solution mean at 32 tokens
REF_RUNTIME: <ms>     # reference mean at 32 tokens
SPEEDUP: <x>
```

With `--verbose`, a full batch-size sweep table is also printed.

Exit code: `0` = correct, `1` = incorrect or failed.

## Correctness Criteria

- **Q output**: FP8 E4M3 as uint8, max diff ≤ 1 ULP
- **K cache**: FP8 E4M3 as uint8, max diff ≤ 1 ULP
- **V cache**: FP8 E4M3 as uint8, max diff ≤ 1 ULP

Tested across token counts: {1, 4, 32, 128, 256, 512} × 5 trials each.

## Baseline (Reference)

`sgl_kernel.fused_qk_norm_rope` (existing CUDA kernel for norm+rope only)
+ PyTorch `.float().div(scale).to(float8_e4m3fn)` cast for Q, K, V
+ `q_output.copy_(q_fp8)` + index scatter write K/V to KV cache

This is the actual unfused production path. All three outputs (Q, K, V) are FP8 E4M3.

## What's Measured

GPU-side kernel time via CUDA events. L2 cache is thrashed before each trial
to measure cold-cache performance. Input tensors are cloned per trial since
the kernel operates in-place.

## CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--solution` | (required) | Path to solution kernel.py |
| `--ref` | (required) | Path to reference.py |
| `--verbose` | off | Detailed correctness + batch sweep |
| `--skip-perf` | off | Only run correctness checks |
| `--num-perf-trials` | 100 | Number of timing trials |

## Required Exports from Solution

The solution `kernel.py` must export:
- `fused_qk_norm_rope_store(qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps, q_weight, k_weight, base, is_neox, position_ids, factor, low, high, attention_factor, rotary_dim, q_output, q_scale, k_cache, v_cache, out_loc, k_scale, v_scale)`

Args added vs original: `q_output` (uint8 buffer for FP8 Q), `q_scale` (float).

The CUDA kernel source (`fused_qknorm_rope_store_kernel.cu`) must be in the same directory as `kernel.py`.
