# Scatter/Gather Design Notes

## Architecture

The pipeline has 3 Triton kernels total:

### Scatter: 2 kernels

1. **`_count_and_compute_layout_kernel`** (grid=1, single-program)
   - Scans `topk_ids` to count tokens per local expert
   - Computes aligned prefix-sum → `packed_layout`
   - Cost is negligible (~2μs), single SM

2. **`_scatter_tokens_kernel`** (grid=min(bs, 8192))
   - Each program handles one token via grid-stride loop
   - Loads full row once (power-of-2 padded to next_power_of_2(hidden_dim))
   - Writes to each assigned expert's region using atomic_add for position
   - Generates `output_index` reverse mapping for gather

### Gather: 1 kernel

3. **`_gather_tokens_kernel`** (grid=(hidden_chunks, min(bs, 1024)))
   - 2D grid: dim-0 tiles hidden dimension in BLOCK_D chunks, dim-1 tiles tokens
   - For each token, iterates over topk slots
   - Uses `output_index >= 0` as validity check (no topk_ids dependency)
   - Accumulates `weight * gemm_output[src_row]` in float32, stores bf16

## Design Choices (mirrors sglang ep_scatter/ep_gather)

- **Per-token processing**: each program owns one token, loads the full row once,
  writes to potentially multiple experts → amortizes the read over topk writes.
- **Grid-stride loops**: handles arbitrary batch sizes with fixed grid.
- **`output_index` as reverse mapping**: scatter writes the destination row index
  for each (token, topk) pair; gather uses this directly.
- **No `topk_ids` in gather**: `output_index == -1` means "non-local expert",
  saving one global load + two int comparisons per topk iteration.

## Performance Characteristics

- **Scatter bottleneck**: random write pattern. Each token writes to one of 20
  non-contiguous expert regions across a ~100MB buffer. L2 miss rate is high.
  Effective bandwidth is ~15× below HBM peak.
- **Gather is cheaper**: reads from scattered positions but writes sequentially
  (each token accumulates to its own contiguous output row).
- **Count+layout kernel**: trivial cost (~2μs), bottleneck is the scatter.

## sglang Comparison

sglang's `fused_moe` uses a different approach:
- `moe_align_block_size_kernel` (~11μs): sorts tokens into block-aligned groups
- GEMM is fused into a single kernel that reads directly from original positions
- `moe_sum_reduce_kernel` (~16μs): weighted reduction back to token order
- Total overhead: ~27μs at ~1024 tokens

The sglang approach avoids explicit scatter entirely by having the GEMM kernel
do indirect indexing. Our scatter/gather approach trades this for compatibility
with DeepGEMM's m-offset layout, which requires physically reordered data.
