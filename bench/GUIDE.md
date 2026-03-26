# Custom Benchmark for MoE Scatter/Gather

## Usage

```bash
python bench/bench.py --solution solution/kernel.py --ref input/reference.py --verbose
```

## Output Format

Each run prints structured lines:

```
COMPILED: True/False
CORRECT: True/False
SCATTER_RUNTIME: <ms>       # kernel-only at bs=1024 (pre-allocated buffers)
GATHER_RUNTIME: <ms>        # at bs=1024
TOTAL_RUNTIME: <ms>         # scatter_kernel + gather at bs=1024
```

With `--verbose`, a full batch-size sweep table is printed before the structured output.

Exit code: `0` = correct, `1` = incorrect or failed.

## What's Measured

- **SCATTER_RUNTIME**: Only the two Triton kernels (count+layout and scatter), with
  pre-allocated buffers. No `torch.zeros`/allocation overhead. This is the
  production-realistic scatter cost.
- **GATHER_RUNTIME**: The gather kernel (weighted accumulation back to token order).
- **TOTAL_RUNTIME**: `SCATTER_RUNTIME + GATHER_RUNTIME` — the total overhead added
  by the scatter/gather pipeline around the GEMM.

All times are GPU-side via CUDA events (`triton.testing.do_bench`). Kernel launch
overhead is NOT included (the GPU pipeline is saturated during the benchmark loop).

## Primary Metric

`TOTAL_RUNTIME` at `bs=1024` is the primary optimization target. This represents
the overhead added around the GEMM for a typical serving batch.

## Default Config

- DeepSeek-V3 style: 160 experts, EP=8, hidden=5120, topk=8
- Local experts per device: 20
- Primary batch size: 1024

## Comparison Baseline

sglang's fused_moe pipeline overhead for ~1024 tokens (GPU-side, from trace):
- `moe_align_block_size_kernel`: ~11μs
- `moe_sum_reduce_kernel`: ~16μs
- **Total: ~27μs** — this is the bar to beat.

## CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--solution` | (required) | Path to optimized kernel.py |
| `--ref` | (required) | Path to reference.py |
| `--verbose` | off | Print detailed correctness results + full sweep table |
| `--skip-perf` | off | Only run correctness checks, skip benchmarks |

## Required Exports from Solution

The solution `kernel.py` must export:
- `moe_align_and_scatter(hidden_states, topk_ids, num_groups, start_expert, max_total_M)` → `(sorted_hidden, packed_layout, output_index)`
- `moe_gather(gemm_output, topk_weights, output_index)` → `output`
- `ALIGNMENT` (int, default 128)
- `_count_and_compute_layout_kernel` (Triton JIT function)
- `_scatter_tokens_kernel` (Triton JIT function)
