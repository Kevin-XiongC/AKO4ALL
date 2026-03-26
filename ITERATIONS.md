# Iteration Log (H200, Per-Token FP8 Scatter/Gather Optimization)

## Summary

| Iter | Title | Scatter(ms) | Gather(ms) | Total(ms) | Status |
|------|-------|-------------|------------|-----------|--------|
| 0 | Per-token baseline (post per-group support) | 0.0237 | 0.0118 | 0.0355 | baseline |
| 1 | Eviction policies (evict_first writes, evict_last reads) | 0.0230 | 0.0115 | 0.0346 | improved -2.5% |
| 2 | Autotune scatter num_warps | — | — | — | CUDA error, reverted |
| 3 | Gather grid (5, 1024) — more token parallelism | 0.0227 | 0.0113 | 0.0340 | improved -1.7% |
| 4 | 2-tile scatter (reduce reg pressure) | 0.0251 | 0.0114 | 0.0365 | worse — double load overhead, reverted |
| 5 | Gather num_warps=8 | 0.0231 | 0.0136 | 0.0367 | worse — too few blocks/SM, reverted |
