# Iteration Log

## Summary

| Iter | Title | Scatter(ms) | Gather(ms) | Total(ms) | Status |
|------|-------|-------------|------------|-----------|--------|
| 0 | Baseline | 0.0643 | 0.0347 | 0.0990 | baseline |
| 1 | tl.histogram in count kernel | 0.0408 | 0.0346 | 0.0754 | improved |
| 2 | gather num_warps=4 | 0.0398 | 0.0338 | 0.0736 | improved |

## Iterations

### Iter 1 — tl.histogram in count kernel
Replaced the O(BLOCK_SIZE × BLOCK_G) inner loop with tl.histogram, eliminating 32 tl.sum+tl.where per batch.
Result: scatter 0.0643→0.0408ms (-36.5%), total 0.0990→0.0754ms (-23.8%)
