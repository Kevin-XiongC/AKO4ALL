"""
PyTorch reference implementations for allreduce fusion correctness validation.

These are pure-PyTorch equivalents of the fused CUDA kernel patterns.
"""

import torch
import torch.distributed as dist


def ref_allreduce(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Reference all-reduce (sum)."""
    out = tensor.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
    return out


def ref_rms_norm(
    hidden_states: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reference RMS normalization in float32.

    Args:
        hidden_states: [token_num, hidden_dim]
        gamma: [hidden_dim]
        eps: epsilon for numerical stability
    Returns:
        [token_num, hidden_dim] in float32
    """
    h = hidden_states.float()
    variance = h.pow(2).mean(dim=-1, keepdim=True)
    normed = h * torch.rsqrt(variance + eps)
    return gamma.float() * normed


def ref_ar_residual_rmsnorm(
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    token_num: int,
    hidden_dim: int,
    group=None,
) -> tuple:
    """Full fused reference: allreduce -> residual_add -> rms_norm.

    Args:
        allreduce_in: [token_num * hidden_dim] flat tensor
        residual_in: [token_num * hidden_dim] flat tensor
        gamma: [hidden_dim]
        eps: RMS norm epsilon
        token_num, hidden_dim: shape info
        group: process group

    Returns:
        (ref_allreduce_out, ref_residual_out, ref_norm_out) all in float32,
        shaped [token_num, hidden_dim].
    """
    ar_clone = allreduce_in.clone()
    dist.all_reduce(ar_clone, op=dist.ReduceOp.SUM, group=group)
    ref_ar = ar_clone.view(token_num, hidden_dim).float()

    ref_residual = ref_ar + residual_in.view(token_num, hidden_dim).float()

    ref_norm = ref_rms_norm(ref_residual, gamma, eps)

    return ref_ar, ref_residual, ref_norm
