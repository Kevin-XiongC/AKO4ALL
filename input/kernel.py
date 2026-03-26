"""
Baseline wrapper for trtllm_allreduce_fusion kernel.

Re-exports the flashinfer.comm APIs needed for testing/benchmarking.
The optimization target is the kARResidualRMSNorm fused pattern:
  allreduce -> residual add -> RMS normalization
"""

from typing import Optional, Union

import torch
import torch.distributed as dist

import flashinfer.comm as comm

# Re-export enums and workspace APIs
AllReduceFusionPattern = comm.AllReduceFusionPattern
QuantizationSFLayout = comm.QuantizationSFLayout

create_workspace = comm.trtllm_create_ipc_workspace_for_all_reduce_fusion
destroy_workspace = comm.trtllm_destroy_ipc_workspace_for_all_reduce_fusion


def run_allreduce_fusion(
    allreduce_in: torch.Tensor,
    world_size: int,
    world_rank: int,
    token_num: int,
    hidden_dim: int,
    workspace_ptrs: torch.Tensor,
    metadata: dict,
    pattern_code: int = AllReduceFusionPattern.kARResidualRMSNorm,
    residual_in: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    rms_gamma: Optional[torch.Tensor] = None,
    rms_eps: float = 1e-5,
    allreduce_out: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
    use_oneshot: Optional[bool] = None,
    trigger_completion_at_end: bool = True,
    fp32_acc: bool = False,
    quant_out: Optional[torch.Tensor] = None,
    scale_out: Optional[torch.Tensor] = None,
    scale_factor: Optional[Union[torch.Tensor, float]] = None,
    layout_code: Optional[int] = None,
) -> None:
    """Invoke trtllm_allreduce_fusion with the specified parameters."""
    comm.trtllm_allreduce_fusion(
        allreduce_in=allreduce_in,
        world_size=world_size,
        world_rank=world_rank,
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=workspace_ptrs,
        launch_with_pdl=launch_with_pdl,
        use_oneshot=use_oneshot,
        trigger_completion_at_end=trigger_completion_at_end,
        fp32_acc=fp32_acc,
        pattern_code=pattern_code,
        allreduce_out=allreduce_out,
        residual_in=residual_in,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        scale_factor=scale_factor,
        layout_code=layout_code,
        metadata=metadata,
    )
