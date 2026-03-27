"""
Baseline reference: sgl_kernel.fused_qk_norm_rope + PyTorch FP8 E4M3 cast + scatter store.

This is the unfused production path used as the golden reference for both
correctness checking and performance comparison.

Outputs: q_output (FP8 E4M3 uint8), k_cache (FP8), v_cache (FP8).
"""

import torch
from sgl_kernel import fused_qk_norm_rope as sgl_fused_qk_norm_rope


def fused_qk_norm_rope_store(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    base: float,
    is_neox: bool,
    position_ids: torch.Tensor,
    factor: float,
    low: float,
    high: float,
    attention_factor: float,
    rotary_dim: int,
    q_output: torch.Tensor,
    q_scale: float,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out_loc: torch.Tensor,
    k_scale: float,
    v_scale: float,
):
    """
    Unfused baseline: norm+rope in-place on qkv, then separate FP8 cast + store.

    Outputs:
      - q_output: [num_tokens, num_heads_q * head_dim] uint8 (FP8 E4M3)
      - k_cache:  scatter-written at out_loc positions, uint8 (FP8 E4M3)
      - v_cache:  scatter-written at out_loc positions, uint8 (FP8 E4M3)
    """
    # Step 1: QK norm + RoPE in-place on qkv
    sgl_fused_qk_norm_rope(
        qkv, num_heads_q, num_heads_k, num_heads_v, head_dim,
        eps, q_weight, k_weight, base, is_neox, position_ids,
        factor, low, high, attention_factor, rotary_dim,
    )

    # Step 2: Extract Q, K, V slices
    q_size = num_heads_q * head_dim
    kv_size = num_heads_k * head_dim
    q = qkv[:, :q_size]
    k = qkv[:, q_size:q_size + kv_size]
    v = qkv[:, q_size + kv_size:]

    # Step 3: FP8 E4M3 cast (div by scale, then cast)
    q_fp8 = q.float().div(q_scale).to(torch.float8_e4m3fn).view(torch.uint8).view(-1, q_size)
    k_fp8 = k.float().div(k_scale).to(torch.float8_e4m3fn).view(torch.uint8).view(-1, kv_size)
    v_fp8 = v.float().div(v_scale).to(torch.float8_e4m3fn).view(torch.uint8).view(-1, kv_size)

    # Step 4: Write outputs
    q_output.copy_(q_fp8)
    k_cache[out_loc.long()] = k_fp8
    v_cache[out_loc.long()] = v_fp8
