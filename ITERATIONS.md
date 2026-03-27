# Iteration Log

## Summary

<!-- Append one row per iteration. Status: improved / no-change / regression / failed -->

| Iter | Title | SUM_SPEEDUP | Mean Speedup | Status |
|------|-------|-------------|--------------|--------|
| 0 | Baseline | 118.76 | 3.71x | baseline |
| 1 | Vectorized uint32 FP8 stores + vec weight loads | 120.39 | 3.76x | improved |
| 2 | Precomputed cos_sin_cache (CUDA-computed) | 116.69 | 3.65x | regression |
| 3 | __launch_bounds__(256,8) + remove BF16 round-trip | 124.16 | 3.88x | improved |
| 4 | cos_sin_cache (CUDA-computed) + __restrict__ | 124.32 | 3.89x | no-change |
| 5 | Vectorized float4 cos_sin_cache loads | 144.08 | 4.50x | improved |
| 6 | Skip sumOfSquares for V heads | 144.38 | 4.51x | no-change |
| 7 | blockSize=128 (launch_bounds 128,16) | 145.69 | 4.55x | improved |
| 8 | Unified Q/K output path + eliminated elements2 array | 146.47 | 4.58x | improved |
| 9 | Hybrid inline/cache + skip attn_factor | 142.56 | 4.46x | regression |
| 9b | Skip attn_factor only (no hybrid) | 144.83 | 4.53x | regression |
| 10 | Pre-bake attention_factor into cos_sin_cache | 148.84 | 4.65x | improved |

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
