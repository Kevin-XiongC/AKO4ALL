# Iteration Log

## Summary

<!-- Append one row per iteration. Status: improved / no-change / regression / failed -->

| Iter | Title | SUM_SPEEDUP | Mean Speedup | Status |
|------|-------|-------------|--------------|--------|
| 0 | Baseline | 118.76 | 3.71x | baseline |
| 1 | Vectorized uint32 FP8 stores + vec weight loads | 120.39 | 3.76x | improved |
| 2 | Precomputed cos_sin_cache (CUDA-computed) | 116.69 | 3.65x | regression |
| 3 | __launch_bounds__(256,8) + remove BF16 round-trip | 124.16 | 3.88x | improved |

## Iterations

<!-- Template — copy for each new iteration:

### Iter N — Short title

- **Hypothesis:** Why this change is expected to help
- **Changes:** What was modified
- **Bench:**
  - Compiled: True/False
  - Correct: True/False
  - Runtime: ___ ms (mean), ___ ~ ___ ms (min ~ max)
  - Speedup: ___x (mean), ___ ~ ___x (min ~ max)
- **Analysis:** Why it worked or failed
- **Next:** What to try next
-->
