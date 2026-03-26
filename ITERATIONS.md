# Iteration Log (H200)

## Summary

| Iter | Title | Scatter(ms) | Gather(ms) | Total(ms) | Status |
|------|-------|-------------|------------|-----------|--------|
| 0 | Baseline (prev optimized from 4090) | 0.0238 | 0.0136 | 0.0374 | baseline |
| 1 | 2D scatter (split position alloc + tiled data copy) | 0.0222 | 0.0132 | 0.0354 | improved |
| 2 | Bigger count BLOCK_SIZE + tuned num_warps | 0.0227 | 0.0136 | 0.0362 | noise/regression |
| 3 | BLOCK_SIZE=8192 count + num_warps=8 data copy | 0.0226 | 0.0136 | 0.0362 | no change (bench uses hardcoded params for scatter_kern) |
| 4 | Gather grid tokens=512 (2 tokens/block) | 0.0226 | 0.0128 | 0.0354 | improved gather |
| 5 | Gather num_warps=4 | 0.0239 | 0.0115 | 0.0354 | **major gather improvement** |
| 6 | Remove scatter/count eviction policies + static_range topk | 0.0237 | 0.0113 | 0.0351 | improved both |
