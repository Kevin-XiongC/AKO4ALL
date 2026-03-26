# Iteration Log — trtllm_allreduce_fusion (kARResidualRMSNorm, 8xH200)

## Final Result: 1.29x speedup (geomean 0.1322 → 0.1024 ms)

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
| 9 | Fine-tune threshold (3MB, 10MB) | — | — | threshold=5MB optimal |
| 10 | Increase grid for small twoshot | **1.29x** | **0.1029** | **improved** |

## Three Successful Optimizations

### 1. Native bf16 packed add (Iter 2) — 1.5% improvement
Replaced float-conversion vec_add with `__hadd2` packed bf16 operations in both `allreduce_sum` and residual add paths. Reduces instruction count by ~60% for these operations. Main impact on small oneshot tokens where compute is a larger fraction.

### 2. Oneshot threshold tuning (Iter 8) — 23% improvement (the big win)
The Python-side oneshot heuristic for 8 GPUs used 42MB threshold (≈268 tokens). The Lamport protocol with 8-rank polling is fundamentally expensive: each element requires 8 NVLink writes + 8 volatile reads per poll attempt. Lowered to 5MB (≈32 tokens), switching medium tokens to twoshot which uses structured scatter-reduce-allgather with explicit barriers.
- Token 128: 0.107ms → 0.045ms (2.4x faster)
- Token 256: 0.186ms → 0.054ms (3.4x faster)

### 3. Grid utilization for small twoshot (Iter 10) — 5% improvement
With the lower threshold, tokens 64-256 use twoshot but with small grids (8-32 blocks vs 132 SMs). Changed grid_size to use `max(token_per_rank, token_num)`, giving full SM utilization for phases 1 (copy) and 3 (fused_op). Extra blocks skip phase 2 but participate in barriers.
- Token 64: 0.048ms → 0.035ms (25% faster)
- Token 128: 0.045ms → 0.040ms (12% faster)

## Key Learnings

1. **NVLink asymmetry**: Remote writes are fire-and-forget (pipelined), remote reads stall threads. "Write to all, read local" is optimal.
2. **NVLink full-duplex**: Already utilized in the interleaved read+write pattern. Separating phases doesn't help.
3. **Register pressure**: 64-72 regs/thread, only 1 block/SM possible. Reducing registers causes spilling that's worse than low occupancy.
4. **RMS norm __syncthreads**: Prevents fusing fused_op into NVLink-heavy loops.
5. **Compiler sensitivity**: Even "branch-free" approaches (pointer swizzling) can regress if they change the compiler's optimization decisions.
6. **Heuristic tuning** was the biggest win — algorithmic parameters matter more than micro-optimizations for this kernel.

