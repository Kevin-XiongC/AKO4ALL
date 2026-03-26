# Iteration Log

## Summary

| Iter | Title | Speedup | Runtime (geomean ms) | Status |
|------|-------|---------|---------------------|--------|
| 1 | Reduce twoshot NVLink writes | 0.91x | 0.1449 | regression |
| 2 | Native bf16 vec_add + allreduce_sum | 1.02x | 0.1302 | improved |
| 3 | Sequential poll-and-accumulate | 0.97x | 0.1365 | regression |
| 4 | __launch_bounds__(640,2) | 0.97x | 0.1360 | regression |
| 5 | Split read/write phases + inline fused_op | 0.99x | 0.1339 | regression |

## Iterations

### Iter 1 — Reduce twoshot NVLink writes
- **Bench:** Speedup=0.91x. Remote reads stall threads. Reverted.

### Iter 2 — Native bf16 vec_add + allreduce_sum
- **Bench:** Speedup=1.02x. __hadd2 helps small oneshot tokens.

### Iter 3 — Sequential poll-and-accumulate
- **Bench:** Speedup=0.97x. Serializes rank detection. Reverted.

### Iter 4 — __launch_bounds__(640,2)
- **Bench:** Speedup=0.97x. Register spilling. Reverted.

### Iter 5 — Split read/write phases + inline fused_op
- **Hypothesis 5a:** Separate NVLink reads and writes into distinct phases for full-duplex utilization (inspired by SHARP/MSCCL++ flat allreduce reaching 460 GB/s).
- **Result 5a:** Speedup=1.001x — NVLink already handles interleaved reads/writes efficiently.
- **Hypothesis 5b:** Fuse own portion's fused_op into phase 2 to save local reads.
- **Result 5b:** Speedup=0.99x — fused_op's __syncthreads slows NVLink pipeline. Reverted.
- **Analysis:** NVLink full-duplex IS utilized in interleaved approach. The RMS norm's __syncthreads is incompatible with NVLink-heavy loops.

### Key Learnings
- NVLink: Writes > reads (fire-and-forget vs stall). Full-duplex works in interleaved mode.
- Compute optimizations help small tokens only (memory-bound for large).
- Register pressure can't be reduced without algorithm change.
- __syncthreads from RMS norm prevents fusing fused_op into NVLink-heavy loops.
- The kernel's 3.8x gap from theoretical is primarily from phase serialization + barriers.

### Next Direction
Try reducing the Lamport clear overhead for oneshot (clear is NRanks * msg_size) and optimizing the oneshot write pattern. Also try reading own data from allreduce_in directly in phase 2 to reduce phase 1 copy.

