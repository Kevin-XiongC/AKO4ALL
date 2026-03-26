# trtllm_allreduce_fusion Architecture

## Overview

`trtllm_allreduce_fusion` is a fused multi-GPU communication + compute CUDA kernel from flashinfer/TensorRT-LLM. It combines all-reduce with post-processing operations (residual add, RMS norm, quantization) into a single kernel launch to minimize latency.

## Fusion Patterns

| Pattern | Code | Operations |
|---------|------|------------|
| kAllReduce | 0 | allreduce only |
| kARResidualRMSNorm | 1 | allreduce → residual add → RMS norm |
| kARResidualRMSNormFP8Quant | 2 | + FP8 quantization |
| kARResidualRMSNormFP4Quant | 3 | + FP4 quantization |
| kARResidualRMSNormOutFP8Quant | 4 | + output FP8 quant |
| kARResidualRMSNormOutFP4Quant | 5 | + output FP4 quant |

Primary optimization target: **kARResidualRMSNorm** (pattern=1).

## Communication Strategies

### Oneshot (Lamport)
- Uses Lamport-style triple buffering with negative-zero sentinel protocol
- Single kernel: each rank writes its data to shared buffer, reads all ranks' data, reduces in-register
- Good for small messages (< ~2MB per rank)
- Heuristic thresholds: TP2=2MB, TP4=0.5MB, TP8=0.25MB

### Twoshot (Barrier-based)
- Phase 1: Scatter-reduce — each rank reduces a slice of the data from all ranks
- Phase 2: Allgather — each rank broadcasts its reduced slice
- Uses explicit barriers between phases
- Better for large messages

## Workspace Layout

Three IPC shared buffers:
1. **Buffer**: `tp_size * max_token_num * hidden_dim * 2` bytes — data exchange area
2. **Flags**: `tp_size * 256 * 4` bytes — barrier synchronization flags
3. **Lamport**: `tp_size * max_token_num * hidden_dim * 2 * 3` bytes — triple buffer for Lamport protocol

Plus 5 int32 flag words: `[atomic_counter, non_lamport_flag, lamport_flag, lamport_comm_size, clear_size]`

## Key Source Files

- Kernel: `/root/flashinfer/include/flashinfer/comm/trtllm_allreduce_fusion.cuh`
- FFI binding: `/root/flashinfer/csrc/trtllm_allreduce_fusion.cu`
- Python API: `/root/flashinfer/flashinfer/comm/trtllm_ar.py`

## Performance-Relevant Parameters

| Parameter | Effect |
|-----------|--------|
| `use_oneshot` | Oneshot vs twoshot strategy (None = auto heuristic) |
| `launch_with_pdl` | Use PDL (Programmatic Dependent Launch) for kernel scheduling |
| `fp32_acc` | Use FP32 accumulation in reduction (accuracy vs speed) |
| `trigger_completion_at_end` | Signal completion at kernel end (for pipelining) |
| `hidden_dim` | Must be divisible by block size; affects register usage |

## Kernel Launch Configuration

- Block size: typically 512 or 1024 threads
- Grid: based on token_num and SM count
- Cluster size: 1-16 (SM 9.0+ cluster groups)
- The kernel uses `__cluster_dims__` for cooperative launch on Hopper+
