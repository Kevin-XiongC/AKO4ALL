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
| 6 | Gather num_stages=2 | 0.0228 | 0.0113 | 0.0340 | no change |
| 7 | static_range + remove redundant -1 stores | 0.0239 | 0.0115 | 0.0354 | worse — excessive unrolling, reverted |
| 8 | Single-pass count kernel (TOTAL<=16384) | 0.0203 | 0.0113 | 0.0316 | improved -7.1% — count kernel much faster |
| 9 | Atomic on packed_layout (remove m_offset load) | 0.0212 | 0.0113 | 0.0325 | slightly worse, reverted |
| 10 | Gather evict_first output stores | 0.0208 | 0.0116 | 0.0324 | gather worse, reverted |
| 11 | Gather BLOCK_D=512 | 0.0202 | 0.0134 | 0.0336 | gather worse, reverted |
| 12 | Autotune count kernel num_warps=8 | 0.0197 | 0.0115 | 0.0312 | improved -1.3% |
| 13 | Autotune scatter single config | — | — | — | num_warps conflict with bench, reverted |
| 14 | Count num_warps=16, gather num_stages=3 | 0.0206 | 0.0114 | 0.0321 | combined worse; count=16 alone ~0.0310 avg — kept num_warps=16 |
| 15 | Count autotune 16 vs 32 warps | — | — | 0.0314 avg | multi-config overhead, kept single 16 |
| 16 | Speculative hidden_states load before pre-check | 0.0200 | 0.0116 | 0.0317 avg | wasted reads offset latency hiding, reverted |
