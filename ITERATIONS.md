# Iteration Log (H200, FP8 Fused Scatter)

## Summary

| Iter | Title | Scatter(ms) | Gather(ms) | Total(ms) | Status |
|------|-------|-------------|------------|-----------|--------|
| 0 | FP8 baseline (scatter+quant fused) | 0.0241 | 0.0120 | 0.0361 | baseline |
| 1 | bf16 max / fewer intermediates / range(topk) | 0.0240 | 0.0113 | 0.0353 | marginal |
