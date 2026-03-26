# Iteration Log

## Summary

| Iter | Title | Speedup | Runtime (geomean ms) | Status |
|------|-------|---------|---------------------|--------|
| 1 | Reduce twoshot NVLink writes | 0.91x | 0.1449 | regression |
| 2 | Native bf16 vec_add + allreduce_sum | 1.02x | 0.1302 | improved |

## Iterations

### Iter 1 — Reduce twoshot NVLink writes

- **Hypothesis:** Phase 2 writes to all N rank buffers (7 remote NVLink writes). Writing only to own buffer saves NVLink bandwidth; phase 3 reads from each rank's buffer instead.
- **Changes:** Twoshot phase 2: write to own comm_buf only. Phase 3: read from comm_bufs[r] for rank r's portion.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 0.1449 ms (geomean)
  - Speedup: 0.91x (regression)
- **Analysis:** Remote NVLink reads in phase 3 are much slower than local HBM reads. Remote writes are fire-and-forget (pipelined), while remote reads stall threads. The original "write to all, read local" pattern is fundamentally better on NVLink.
- **Next:** Optimize compute path.

### Iter 2 — Native bf16 vec_add + allreduce_sum

- **Hypothesis:** vec_add and allreduce_sum convert bf16→float→bf16 for each add. Native bf16 packed add (__hadd2) uses 1 instruction instead of 5 per pair.
- **Changes:** Specialized vec_add and allreduce_sum for bf16/fp16 using __hadd2 packed operations.
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 0.1302 ms (geomean)
  - Speedup: 1.02x (improved)
- **Analysis:** Main benefit for small oneshot tokens (token=16: 18% faster) where compute fraction is higher. Large twoshot tokens are memory-bound, compute optimization has minimal impact.
- **Next:** Focus on memory access patterns or launch config to improve memory-bound large token performance.

