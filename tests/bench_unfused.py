"""
Benchmark: fused allreduce+residual+RMSNorm vs unfused (NCCL allreduce + fused_add_rmsnorm).
"""
import os, socket, time, torch, torch.distributed as dist
import torch.multiprocessing as mp
import flashinfer.comm as comm

HIDDEN_DIM = 5120
TOKEN_NUMS = [16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 8192, 16384]
MAX_TOKEN_NUM = 16384
DTYPE = torch.bfloat16
WARMUP = 5
ITERS = 20

def ref_rms_norm(x, gamma, eps):
    h = x.float()
    var = h.pow(2).mean(dim=-1, keepdim=True)
    return (gamma.float() * h * torch.rsqrt(var + eps)).to(x.dtype)

def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def worker(local_rank, num_gpus, port):
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                            world_size=num_gpus, rank=local_rank)
    group = dist.group.WORLD

    ipc_handles, ws_tensor, ws_meta = comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
        local_rank, num_gpus, MAX_TOKEN_NUM, HIDDEN_DIM,
        group=group, use_fp32_lamport=False, create_metadata=True)

    rms_gamma = torch.randn(HIDDEN_DIM, dtype=DTYPE, device=device)
    rms_eps = 1e-5

    if local_rank == 0:
        print(f"\n{'token':>6s}  {'fused':>10s}  {'unfused':>10s}  {'speedup':>8s}")
        print(f"{'':>6s}  {'(ms)':>10s}  {'(ms)':>10s}  {'':>8s}")
        print("-" * 42)

    for token_num in TOKEN_NUMS:
        msg_size = token_num * HIDDEN_DIM
        allreduce_in = torch.randn(msg_size, dtype=DTYPE, device=device)
        residual_in = torch.randn(msg_size, dtype=DTYPE, device=device)
        residual_out = torch.empty_like(residual_in)
        norm_out = torch.empty_like(residual_in)

        # --- Fused kernel ---
        def run_fused():
            comm.trtllm_allreduce_fusion(
                allreduce_in=allreduce_in, world_size=num_gpus, world_rank=local_rank,
                token_num=token_num, hidden_dim=HIDDEN_DIM, workspace_ptrs=ws_tensor,
                launch_with_pdl=True, use_oneshot=None, trigger_completion_at_end=True,
                fp32_acc=False, pattern_code=comm.AllReduceFusionPattern.kARResidualRMSNorm,
                allreduce_out=None, residual_in=residual_in, residual_out=residual_out,
                norm_out=norm_out, quant_out=None, scale_out=None,
                rms_gamma=rms_gamma, rms_eps=rms_eps, scale_factor=None,
                layout_code=None, metadata=ws_meta)

        # --- Unfused: NCCL allreduce + manual residual+RMSNorm ---
        ar_out = torch.empty_like(allreduce_in)
        def run_unfused():
            ar_out.copy_(allreduce_in)
            dist.all_reduce(ar_out, op=dist.ReduceOp.SUM, group=group)
            ar_2d = ar_out.view(token_num, HIDDEN_DIM)
            res_2d = residual_in.view(token_num, HIDDEN_DIM)
            res_result = ar_2d + res_2d
            residual_out.view(token_num, HIDDEN_DIM).copy_(res_result)
            h = res_result.float()
            var = h.pow(2).mean(dim=-1, keepdim=True)
            norm_out.view(token_num, HIDDEN_DIM).copy_(
                (rms_gamma.float() * h * torch.rsqrt(var + rms_eps)).to(DTYPE))

        # Warmup
        for _ in range(WARMUP):
            run_fused()
            run_unfused()
        torch.cuda.synchronize()
        dist.barrier(group=group)

        # Measure fused
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITERS):
            run_fused()
        e.record()
        torch.cuda.synchronize()
        fused_ms = s.elapsed_time(e) / ITERS

        dist.barrier(group=group)

        # Measure unfused
        s.record()
        for _ in range(ITERS):
            run_unfused()
        e.record()
        torch.cuda.synchronize()
        unfused_ms = s.elapsed_time(e) / ITERS

        if local_rank == 0:
            sp = unfused_ms / fused_ms if fused_ms > 0 else 0
            print(f"{token_num:6d}  {fused_ms:10.4f}  {unfused_ms:10.4f}  {sp:7.2f}x")

        dist.barrier(group=group)

    comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion(ipc_handles, group=group)
    dist.destroy_process_group()

def main():
    num_gpus = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
    avail = torch.cuda.device_count()
    if num_gpus > avail:
        num_gpus = avail
    port = get_open_port()
    print(f"Comparing fused vs unfused: {num_gpus} GPUs, hidden={HIDDEN_DIM}, dtype={DTYPE}")
    mp.spawn(worker, args=(num_gpus, port), nprocs=num_gpus)

if __name__ == "__main__":
    main()
