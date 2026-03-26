# Iteration Log

## Summary

| Iter | Title | Speedup | Runtime (geomean ms) | Status |
|------|-------|---------|---------------------|--------|
| 1 | Reduce twoshot NVLink writes | 0.91x | 0.1449 | regression |

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
- **Next:** Try different approaches — optimize the compute path (RMS norm, vec_add) or improve launch configuration.

