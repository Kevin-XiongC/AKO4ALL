# Iteration Log

## Summary

| Iter | Title | Speedup | Runtime (geomean ms) | Status |
|------|-------|---------|---------------------|--------|
| 1 | Reduce twoshot NVLink writes | 0.91x | 0.1449 | regression |
| 2 | Native bf16 vec_add + allreduce_sum | 1.02x | 0.1302 | improved |
| 3 | Sequential poll-and-accumulate | 0.97x | 0.1365 | regression |
| 4 | __launch_bounds__(640,2) | 0.97x | 0.1360 | regression |
| 5 | Split read/write phases + inline fused_op | 0.99x | 0.1339 | regression |
| 6 | Merge write+clear + read from allreduce_in | 0.97x | 0.1363 | regression |
| 7 | Move oneshot clear after poll+fused_op | 1.01x | 0.1312 | no-change |
| 8 | Lower oneshot threshold for 8 GPUs | **1.23x** | **0.1078** | **improved** |

## Key Improvement: Iter 8

The Python-side oneshot/twoshot threshold for 8 GPUs was 42MB (≈268 tokens), making token 64-256 use the expensive Lamport oneshot protocol. Lowered to 5MB (≈32 tokens), switching medium tokens to the more efficient twoshot. Results:

- Token 128: 0.107ms → 0.045ms (**2.4x faster**)
- Token 256: 0.186ms → 0.054ms (**3.4x faster**)
- Token 64: 0.055ms → 0.048ms (13% faster)
- Large tokens (512+): unchanged

The 8-rank Lamport polling is fundamentally inefficient: each rank writes to 8 buffers then polls 8 remote entries per element with volatile loads. The twoshot's structured scatter-reduce-allgather with barriers is much more efficient for these message sizes.

## Iterations 1-7 (see git history)
Only iter 2 (native bf16 __hadd2) improved (1.5%). All NVLink restructuring attempts (iters 1,3,4,5,6,7) regressed or were neutral, confirming the twoshot algorithm is well-optimized.

