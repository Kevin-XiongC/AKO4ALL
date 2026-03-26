# Iteration Log

## Summary

| Iter | Title | Speedup | Runtime (geomean ms) | Status |
|------|-------|---------|---------------------|--------|
| 1 | Reduce twoshot NVLink writes | 0.91x | 0.1449 | regression |
| 2 | Native bf16 vec_add + allreduce_sum | 1.02x | 0.1302 | improved |
| 3 | Sequential poll-and-accumulate | 0.97x | 0.1365 | regression |
| 4 | __launch_bounds__(640,2) | 0.97x | 0.1360 | regression |

## Iterations

### Iter 1 — Reduce twoshot NVLink writes

- **Hypothesis:** Writing reduced data only to own buffer saves NVLink bandwidth.
- **Changes:** Phase 2 write to own comm_buf only. Phase 3 read from comm_bufs[r].
- **Bench:** CORRECT=True, Runtime=0.1449ms, Speedup=0.91x
- **Analysis:** Remote reads stall threads. "Write to all, read local" is better on NVLink.

### Iter 2 — Native bf16 vec_add + allreduce_sum

- **Hypothesis:** __hadd2 packed add avoids float conversion overhead.
- **Changes:** Specialized vec_add/allreduce_sum for bf16/fp16 using __hadd2.
- **Bench:** CORRECT=True, Runtime=0.1302ms, Speedup=1.02x
- **Analysis:** Helps small oneshot tokens (18% for token=16). Memory-bound large tokens unaffected.

### Iter 3 — Sequential poll-and-accumulate

- **Hypothesis:** Reduce register pressure from 64→40 regs by eliminating vals[NRanks].
- **Changes:** Sequential per-rank polling, accumulate inline. Reverted.
- **Bench:** CORRECT=True, Runtime=0.1365ms, Speedup=0.97x
- **Analysis:** Sequential polling serializes rank detection. Parallel is critical.

### Iter 4 — __launch_bounds__(640,2)

- **Hypothesis:** Force compiler to limit regs for 2 blocks/SM occupancy.
- **Changes:** Added __launch_bounds__(640, 2) to both kernels. Reverted.
- **Bench:** CORRECT=True, Runtime=0.1360ms, Speedup=0.97x
- **Analysis:** Compiler spills to local memory (STACK increases), worse than 1 block/SM. The kernel's register usage is inherent to the algorithm — occupancy can't be improved without algorithmic changes.

### Observations after 4 iterations

**What works:** Native bf16 compute ops reduce instruction overhead.
**What doesn't:** Changing NVLink access patterns, serializing rank polling, forcing occupancy.
**Root cause:** Kernel is fundamentally NVLink-bandwidth-bound for twoshot (large tokens) and NVLink-latency-bound for oneshot (small tokens). The phased execution model (copy→barrier→reduce→barrier→fused_op) adds unavoidable overhead.
**Next directions:**
1. Optimize the ONESHOT threshold per NRanks — current 128 may not be optimal for 8 GPUs
2. Reduce phase 1 copy size by reading own data from allreduce_in in phase 2
3. Merge phase 1 copy with phase 2 using Lamport-style write detection (eliminate barrier 1)
4. Explore deeper fusion: apply fused_op to own portion inline in phase 2 before barrier 2

