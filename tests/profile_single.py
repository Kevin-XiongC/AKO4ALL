#!/usr/bin/env python3
"""
Single-process profiling helper for ncu.
Launches 8 processes, only rank 0 is profiled.

Usage:
  ncu --target-processes all -k regex:allreduce_fusion -o profile \
    python tests/profile_single.py [--tokens 128] [--oneshot]
"""
import argparse
import torch.multiprocessing as mp
import os
import socket
import torch
import torch.distributed as dist
import flashinfer.comm as comm

HIDDEN_DIM = 5120

def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def worker(local_rank, num_gpus, port, token_num, use_oneshot):
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=num_gpus,
        rank=local_rank,
    )
    group = dist.group.WORLD
    ipc_handles, workspace_tensor, workspace_metadata = (
        comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
            local_rank, num_gpus, max(token_num, 16384), HIDDEN_DIM,
            group=group, use_fp32_lamport=False, create_metadata=True,
        )
    )
    msg_size = token_num * HIDDEN_DIM
    allreduce_in = torch.randn(msg_size, dtype=torch.bfloat16, device=device)
    residual_in = torch.randn(msg_size, dtype=torch.bfloat16, device=device)
    residual_out = torch.empty_like(residual_in)
    norm_out = torch.empty_like(residual_in)
    rms_gamma = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(3):
        comm.trtllm_allreduce_fusion(
            allreduce_in=allreduce_in, world_size=num_gpus, world_rank=local_rank,
            token_num=token_num, hidden_dim=HIDDEN_DIM, workspace_ptrs=workspace_tensor,
            launch_with_pdl=True, use_oneshot=use_oneshot,
            trigger_completion_at_end=True, fp32_acc=False,
            pattern_code=comm.AllReduceFusionPattern.kARResidualRMSNorm,
            allreduce_out=None, residual_in=residual_in, residual_out=residual_out,
            norm_out=norm_out, quant_out=None, scale_out=None,
            rms_gamma=rms_gamma, rms_eps=1e-5, scale_factor=None,
            layout_code=None, metadata=workspace_metadata,
        )
    torch.cuda.synchronize()
    dist.barrier(group=group)

    # Profiled run
    comm.trtllm_allreduce_fusion(
        allreduce_in=allreduce_in, world_size=num_gpus, world_rank=local_rank,
        token_num=token_num, hidden_dim=HIDDEN_DIM, workspace_ptrs=workspace_tensor,
        launch_with_pdl=True, use_oneshot=use_oneshot,
        trigger_completion_at_end=True, fp32_acc=False,
        pattern_code=comm.AllReduceFusionPattern.kARResidualRMSNorm,
        allreduce_out=None, residual_in=residual_in, residual_out=residual_out,
        norm_out=norm_out, quant_out=None, scale_out=None,
        rms_gamma=rms_gamma, rms_eps=1e-5, scale_factor=None,
        layout_code=None, metadata=workspace_metadata,
    )
    torch.cuda.synchronize()
    dist.barrier(group=group)
    comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(ipc_handles, group=group)
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--oneshot", action="store_true")
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()
    port = get_open_port()
    print(f"Profiling: tokens={args.tokens}, oneshot={args.oneshot}, gpus={args.gpus}")
    mp.spawn(worker, args=(args.gpus, port, args.tokens, args.oneshot if args.oneshot else None), nprocs=args.gpus)

if __name__ == "__main__":
    main()
