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

## Key Learnings After 6 Iterations

This kernel is extremely well-optimized by the TRT-LLM team. The main bottlenecks are:

1. **NVLink bandwidth** for twoshot large tokens (cannot exceed hardware limit)
2. **NVLink latency + Lamport polling overhead** for oneshot small tokens
3. **Phase serialization** (copy → barrier → reduce → barrier → fused_op) is inherent to the algorithm
4. **RMS norm's __syncthreads** prevents fusing computation into communication-heavy loops

What works: native bf16 __hadd2 packed operations (reduces instruction count for compute-bound oneshot).
What doesn't: restructuring NVLink access patterns, forcing occupancy, or merging/splitting phases.

The kernel achieves ~236 GB/s bus BW on 8xH200 for large tokens, vs theoretical ~450 GB/s per direction. The 2x gap is from the allreduce requiring both reads AND writes (each direction sees ~236/2 = 118 GB/s effective, with the other direction used for the complementary operation).

## Iteration Details

### Iter 1-6 (see git history for details)
All iterations except iter 2 regressed or were neutral. Only iter 2 (native bf16 __hadd2) improved by 1.5%.

