# Iteration Log (H200, Per-Token FP8 Scatter/Gather Optimization)

## Summary

| Iter | Title | Scatter(ms) | Gather(ms) | Total(ms) | Status |
|------|-------|-------------|------------|-----------|--------|
| 0 | Per-token baseline (post per-group support) | 0.0237 | 0.0118 | 0.0355 | baseline |
| 1 | Eviction policies (evict_first writes, evict_last reads) | 0.0230 | 0.0115 | 0.0346 | improved -2.5% |
| 2 | Autotune scatter num_warps | — | — | — | CUDA error, reverted |
