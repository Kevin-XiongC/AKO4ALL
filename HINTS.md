# Hints

## Kernel Overview

MoE scatter/gather for DeepGEMM m-offset layout, targeting Expert Parallelism (EP).
Three Triton kernels: count+layout (grid=1), scatter (per-token), gather (2D-grid).
See `context/` for full design notes and m-offset specification.

## Key Constraints

- Output `packed_layout` format must match DeepGEMM: `[m_offsets | m_counts]`, int32, offsets aligned to 128.
- `output_index` encodes scatter destination per (token, topk) pair; -1 for non-local experts.
- `moe_gather` signature must remain `(gemm_output, topk_weights, output_index) → output`.
- bf16 input/output throughout; float32 accumulation in gather.

## Optimization Target

Primary metric: `TOTAL_RUNTIME` at bs=1024 (scatter kernel-only + gather).
Baseline sglang overhead for ~1024 tokens: ~27μs GPU. Target: approach or beat this.

## Known Bottlenecks

1. **Scatter random writes** — the dominant cost. Each token writes to a different expert's
   region in the output buffer, causing poor L2 utilization. The effective bandwidth is
   ~15× worse than sequential writes.
2. **Atomic contention** — `atomic_add` on per-expert write counters. Low contention with
   20 experts, but still serializes position allocation per expert.

## High-Value Optimization Directions

1. **Fuse scatter + per-token FP8 quantization** — the scatter kernel already loads the
   full hidden row; quantizing in the same kernel avoids a second data pass. However,
   this changes the output type and requires the GEMM to accept FP8 input.
2. **Sort tokens by expert before scatter** — converts random writes to sequential,
   dramatically improving bandwidth utilization. Trade-off: adds a radix-sort overhead.
3. **Shared-memory write buffering** — batch writes to the same expert in shared memory
   before flushing to global memory. Improves write coalescing.
4. **Two-pass scatter** — first pass allocates positions (small writes), second pass does
   the data copy in expert-sorted order.
5. **Gather: increase BLOCK_D** — currently 128 for non-1024-divisible dims; tuning this
   and num_warps may help.

## Profiling

- Before Iter 1, run `ncu` on the baseline kernel to confirm the bottleneck.
- If 3 consecutive iterations show no improvement, run `ncu` to re-profile, use WebSearch
  for new ideas, and review `ITERATIONS.md` for patterns. Plan before continuing.

## Rules

- Do NOT switch languages (stay in Triton/Python).
- Do NOT modify `bench/bench.py` or `input/reference.py`.
- The `solution/kernel.py` must export the same API as `input/kernel.py`.
