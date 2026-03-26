# Iteration Log

## Summary

| Iter | Title | Scatter(ms) | Gather(ms) | Total(ms) | Status |
|------|-------|-------------|------------|-----------|--------|
| 0 | Baseline | 0.0643 | 0.0347 | 0.0990 | baseline |
| 1 | tl.histogram in count kernel | 0.0408 | 0.0346 | 0.0754 | improved |
| 2 | gather num_warps=4 | 0.0398 | 0.0338 | 0.0736 | improved |
| 3 | tiled gather (1D grid) | 0.0407 | 0.0435 | 0.0842 | regression — reverted |
| 4 | gather BLOCK_D=512, num_warps=4 | 0.0407 | 0.0329 | 0.0736 | improved |
| 5 | wrapper optimizations | 0.0407 | 0.0331 | 0.0738 | no change |
| 6 | torch.empty + BLOCK_D tuning | 0.0408 | 0.0309 | 0.0718 | improved |
| 7 | skip data load for non-local tokens | 0.0370 | 0.0305 | 0.0675 | improved |
| 8 | gather BLOCK_D=1024, num_warps=8 | 0.0372 | 0.0301 | 0.0673 | marginal |
| 9 | scatter eviction_policy=evict_first | 0.0364 | 0.0306 | 0.0670 | marginal |
| 10 | CUDA graph/tiled scatter - reverted | 0.0372 | 0.0307 | 0.0679 | no change |
| 11 | remove eviction policy + BLOCK_D=1024 | 0.0369 | 0.0302 | 0.0671 | best |

## Iterations

### Iter 1 — tl.histogram in count kernel
Replaced the O(BLOCK_SIZE × BLOCK_G) inner loop with tl.histogram, eliminating 32 tl.sum+tl.where per batch.
Result: scatter 0.0643→0.0408ms (-36.5%), total 0.0990→0.0754ms (-23.8%)
