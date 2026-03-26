# Iteration Log (H200)

## Summary

| Iter | Title | Scatter(ms) | Gather(ms) | Total(ms) | Status |
|------|-------|-------------|------------|-----------|--------|
| 0 | Baseline (prev optimized from 4090) | 0.0238 | 0.0136 | 0.0374 | baseline |
| 1 | 2D scatter (split position alloc + tiled data copy) | 0.0222 | 0.0132 | 0.0354 | improved |
| 2 | Bigger count BLOCK_SIZE + tuned num_warps | 0.0227 | 0.0136 | 0.0362 | noise/regression |
