"""
Correctness + bandwidth test for trtllm_allreduce_fusion (flashinfer.comm).

Focuses on kARResidualRMSNorm pattern (allreduce -> residual add -> RMS norm).

Usage (single-node, 8 GPUs by default):
  python tests/test_allreduce_fusion.py

Usage (single-node, 2 GPUs):
  LOCAL_WORLD_SIZE=2 python tests/test_allreduce_fusion.py

Usage (multi-node, NCCL):
  Node 0: MASTER_ADDR=10.0.0.1 MASTER_PORT=8361 WORLD_SIZE=2 RANK=0 python tests/test_allreduce_fusion.py
  Node 1: MASTER_ADDR=10.0.0.1 MASTER_PORT=8361 WORLD_SIZE=2 RANK=1 python tests/test_allreduce_fusion.py
"""

import multiprocessing as mp
import os
import socket
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

import flashinfer.comm as comm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TOKEN_NUM = 16384
HIDDEN_DIM = 5120

CORRECTNESS_CONFIGS: List[Tuple[int, int, torch.dtype, int, Optional[bool]]] = [
    # (token_num, hidden_dim, dtype, pattern_code, use_oneshot)
    # kAllReduce (pattern=0) — basic sanity
    (128, HIDDEN_DIM, torch.bfloat16, 0, True),
    (128, HIDDEN_DIM, torch.bfloat16, 0, False),
    # kARResidualRMSNorm (pattern=1) — primary target
    (16,    HIDDEN_DIM, torch.bfloat16, 1, True),
    (128,   HIDDEN_DIM, torch.bfloat16, 1, True),
    (128,   HIDDEN_DIM, torch.bfloat16, 1, False),
    (128,   HIDDEN_DIM, torch.float16,  1, True),
    (128,   HIDDEN_DIM, torch.float16,  1, False),
    (1024,  HIDDEN_DIM, torch.bfloat16, 1, True),
    (1024,  HIDDEN_DIM, torch.bfloat16, 1, False),
    (4096,  HIDDEN_DIM, torch.bfloat16, 1, True),
    (4096,  HIDDEN_DIM, torch.bfloat16, 1, False),
    (16384, HIDDEN_DIM, torch.bfloat16, 1, True),
    (16384, HIDDEN_DIM, torch.bfloat16, 1, False),
]

BENCH_TOKEN_NUMS = [16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 8192, 16384]
BENCH_HIDDEN_DIM = HIDDEN_DIM
BENCH_DTYPE = torch.bfloat16
BENCH_PATTERN = comm.AllReduceFusionPattern.kARResidualRMSNorm
BENCH_WARMUP = 5
BENCH_ITERS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ref_rms_norm(hidden_states: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMS norm in float32."""
    h = hidden_states.float()
    variance = h.pow(2).mean(dim=-1, keepdim=True)
    normed = h * torch.rsqrt(variance + eps)
    return gamma.float() * normed


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correctness(
    rank: int,
    world_size: int,
    workspace_tensor: torch.Tensor,
    workspace_metadata: dict,
    group,
    device: torch.device,
) -> bool:
    all_pass = True

    for token_num, hidden_dim, dtype, pattern_code, use_oneshot in CORRECTNESS_CONFIGS:
        # Skip twoshot when token_num < world_size (trtllm constraint)
        if not use_oneshot and token_num < world_size:
            continue

        # Need workspace big enough
        if token_num * hidden_dim > workspace_metadata["max_token_num"] * workspace_metadata["hidden_dim"]:
            continue

        tag = (f"tok={token_num} hid={hidden_dim} dt={dtype} "
               f"pat={pattern_code} oneshot={use_oneshot}")

        dist.barrier(group=group)

        message_size = token_num * hidden_dim

        # Allocate inputs
        allreduce_in = torch.randn(message_size, dtype=dtype, device=device)
        allreduce_in_clone = allreduce_in.clone()

        allreduce_out = torch.zeros(message_size, dtype=dtype, device=device)
        residual_in = torch.randn(message_size, dtype=dtype, device=device)
        residual_in_clone = residual_in.clone()
        residual_out = torch.empty_like(residual_in)
        norm_out = torch.empty_like(residual_in)

        rms_gamma = torch.randn(hidden_dim, dtype=dtype, device=device)
        rms_eps = 1e-5

        # Run kernel
        comm.trtllm_allreduce_fusion(
            allreduce_in=allreduce_in,
            world_size=world_size,
            world_rank=rank,
            token_num=token_num,
            hidden_dim=hidden_dim,
            workspace_ptrs=workspace_tensor,
            launch_with_pdl=True,
            use_oneshot=use_oneshot,
            trigger_completion_at_end=True,
            fp32_acc=False,
            pattern_code=pattern_code,
            allreduce_out=allreduce_out,
            residual_in=residual_in,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=None,
            scale_out=None,
            rms_gamma=rms_gamma,
            rms_eps=rms_eps,
            scale_factor=None,
            layout_code=None,
            metadata=workspace_metadata,
        )
        torch.cuda.synchronize()

        # Compute reference
        dist.all_reduce(allreduce_in_clone, group=group)
        ref_ar = allreduce_in_clone.view(token_num, hidden_dim).float()

        tolerance = 8e-2 if dtype == torch.float16 else 8e-1

        if pattern_code == comm.AllReduceFusionPattern.kAllReduce:
            try:
                torch.testing.assert_close(
                    allreduce_out.view(token_num, hidden_dim).float(),
                    ref_ar, atol=tolerance, rtol=1e-2,
                )
                status = "PASS"
            except AssertionError as e:
                status = "FAIL"
                all_pass = False
        elif pattern_code == comm.AllReduceFusionPattern.kARResidualRMSNorm:
            ref_residual = ref_ar + residual_in_clone.view(token_num, hidden_dim).float()
            ref_norm = ref_rms_norm(ref_residual, rms_gamma, rms_eps)
            try:
                torch.testing.assert_close(
                    residual_out.view(token_num, hidden_dim).float(),
                    ref_residual, atol=tolerance, rtol=1e-2,
                )
                torch.testing.assert_close(
                    norm_out.view(token_num, hidden_dim).float(),
                    ref_norm, atol=tolerance, rtol=1e-2,
                )
                status = "PASS"
            except AssertionError as e:
                status = "FAIL"
                all_pass = False
        else:
            status = "SKIP"

        if rank == 0:
            print(f"  [{status}] {tag}", flush=True)

        dist.barrier(group=group)

    return all_pass


# ---------------------------------------------------------------------------
# Bandwidth benchmark
# ---------------------------------------------------------------------------

def bench_bandwidth(
    rank: int,
    world_size: int,
    workspace_tensor: torch.Tensor,
    workspace_metadata: dict,
    group,
    device: torch.device,
) -> Dict[int, float]:
    """Run bandwidth sweep, return {token_num: elapsed_ms}."""
    results: Dict[int, float] = {}
    dtype = BENCH_DTYPE
    hidden_dim = BENCH_HIDDEN_DIM
    rms_eps = 1e-5

    for token_num in BENCH_TOKEN_NUMS:
        if token_num * hidden_dim > workspace_metadata["max_token_num"] * workspace_metadata["hidden_dim"]:
            continue

        message_size = token_num * hidden_dim

        allreduce_in = torch.randn(message_size, dtype=dtype, device=device)
        residual_in = torch.randn(message_size, dtype=dtype, device=device)
        residual_out = torch.empty_like(residual_in)
        norm_out = torch.empty_like(residual_in)
        rms_gamma = torch.randn(hidden_dim, dtype=dtype, device=device)

        use_oneshot = None  # let heuristics decide

        def run_kernel():
            comm.trtllm_allreduce_fusion(
                allreduce_in=allreduce_in,
                world_size=world_size,
                world_rank=rank,
                token_num=token_num,
                hidden_dim=hidden_dim,
                workspace_ptrs=workspace_tensor,
                launch_with_pdl=True,
                use_oneshot=use_oneshot,
                trigger_completion_at_end=True,
                fp32_acc=False,
                pattern_code=BENCH_PATTERN,
                allreduce_out=None,
                residual_in=residual_in,
                residual_out=residual_out,
                norm_out=norm_out,
                quant_out=None,
                scale_out=None,
                rms_gamma=rms_gamma,
                rms_eps=rms_eps,
                scale_factor=None,
                layout_code=None,
                metadata=workspace_metadata,
            )

        # Warmup
        for _ in range(BENCH_WARMUP):
            run_kernel()
        torch.cuda.synchronize()
        dist.barrier(group=group)

        # Measure
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(BENCH_ITERS):
            run_kernel()
        end_event.record()
        torch.cuda.synchronize()

        total_ms = start_event.elapsed_time(end_event)
        avg_ms = total_ms / BENCH_ITERS

        # Bus bandwidth: for allreduce, effective data moved = 2*(N-1)/N * msg_bytes
        msg_bytes = message_size * dtype.itemsize
        bus_bw = msg_bytes * 2.0 * (world_size - 1) / world_size * BENCH_ITERS / (total_ms / 1000.0) / 1e9

        results[token_num] = avg_ms

        if rank == 0:
            print(f"  token_num={token_num:5d}  avg={avg_ms:.4f} ms  "
                  f"total={total_ms:.1f} ms  bus_bw={bus_bw:.2f} GB/s",
                  flush=True)

        dist.barrier(group=group)

    return results


# ---------------------------------------------------------------------------
# Per-rank worker
# ---------------------------------------------------------------------------

def worker(
    local_rank: int,
    num_local_ranks: int,
    port: int,
    node_rank: int = 0,
    num_nodes: int = 1,
    master_addr: str = "127.0.0.1",
):
    global_rank = node_rank * num_local_ranks + local_rank
    world_size = num_nodes * num_local_ranks

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"[rank {global_rank}] Initializing (node={node_rank}, local={local_rank}, "
          f"world={world_size}, master={master_addr}:{port})", flush=True)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{port}",
        world_size=world_size,
        rank=global_rank,
    )
    group = dist.group.WORLD

    # Create workspace
    ipc_handles, workspace_tensor, workspace_metadata = (
        comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
            global_rank,
            world_size,
            MAX_TOKEN_NUM,
            HIDDEN_DIM,
            group=group,
            use_fp32_lamport=False,
            create_metadata=True,
        )
    )

    try:
        # --- Correctness ---
        if global_rank == 0:
            print("\n=== Correctness Tests ===", flush=True)
        dist.barrier(group=group)

        correct = check_correctness(
            global_rank, world_size, workspace_tensor, workspace_metadata, group, device
        )

        if global_rank == 0:
            print(f"\nCorrectness: {'ALL PASS' if correct else 'SOME FAILED'}", flush=True)

        # --- Bandwidth ---
        if global_rank == 0:
            print(f"\n=== Bandwidth Benchmark (pattern=kARResidualRMSNorm, "
                  f"hidden={BENCH_HIDDEN_DIM}, dtype={BENCH_DTYPE}, "
                  f"world_size={world_size}) ===", flush=True)
        dist.barrier(group=group)

        bw_results = bench_bandwidth(
            global_rank, world_size, workspace_tensor, workspace_metadata, group, device
        )

    finally:
        dist.barrier(group=group)
        comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(ipc_handles, group=group)
        dist.destroy_process_group()

    if global_rank == 0:
        print(f"\n[rank 0] Done.", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if all(k in os.environ for k in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]):
        # Multi-node mode: one process per invocation
        master_addr = os.environ["MASTER_ADDR"]
        port = int(os.environ["MASTER_PORT"])
        num_nodes = int(os.environ["WORLD_SIZE"])
        node_rank = int(os.environ["RANK"])
        num_local = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))

        torch.multiprocessing.spawn(
            worker,
            args=(num_local, port, node_rank, num_nodes, master_addr),
            nprocs=num_local,
        )
    else:
        # Single-node mode
        num_local = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
        available = torch.cuda.device_count()
        if num_local > available:
            print(f"LOCAL_WORLD_SIZE={num_local} > available GPUs={available}, "
                  f"using {available}", flush=True)
            num_local = available

        port = get_open_port()
        print(f"Single-node mode: {num_local} GPUs, port={port}", flush=True)

        torch.multiprocessing.spawn(
            worker,
            args=(num_local, port),
            nprocs=num_local,
        )


if __name__ == "__main__":
    main()
