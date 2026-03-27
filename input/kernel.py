"""
Fused QK-Norm + RoPE + FP8-Cast + KV-Store CUDA kernel.

JIT-compiles the CUDA kernel from fused_qknorm_rope_store_kernel.cu
located alongside this file.

Outputs: q_output (FP8 E4M3 uint8), k_cache (FP8), v_cache (FP8).
"""

import os
import torch
from torch.utils.cpp_extension import load_inline

_KERNEL_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_kernel():
    with open(os.path.join(_KERNEL_DIR, "fused_qknorm_rope_store_kernel.cu"), "r") as f:
        cuda_source = f.read()

    cpp_source = """
void fused_qk_norm_rope_store(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    double base,
    bool is_neox,
    torch::Tensor& position_ids,
    double factor,
    double low,
    double high,
    double attention_factor,
    int64_t rotary_dim,
    torch::Tensor& q_output,
    double q_scale,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor& out_loc,
    double k_scale,
    double v_scale);
"""

    module = load_inline(
        name="fused_qknorm_rope_store",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=["fused_qk_norm_rope_store"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return module


_module = None


def get_module():
    global _module
    if _module is None:
        _module = _load_kernel()
    return _module


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
    Fused QK-Norm + RoPE + FP8-Cast + Store.

    Outputs:
      - q_output: [num_tokens, num_heads_q * head_dim] uint8 (FP8 E4M3)
      - k_cache:  scatter-written at out_loc positions, uint8 (FP8 E4M3)
      - v_cache:  scatter-written at out_loc positions, uint8 (FP8 E4M3)
    """
    mod = get_module()
    mod.fused_qk_norm_rope_store(
        qkv, num_heads_q, num_heads_k, num_heads_v, head_dim,
        eps, q_weight, k_weight, base, is_neox, position_ids,
        factor, low, high, attention_factor, rotary_dim,
        q_output, q_scale,
        k_cache, v_cache, out_loc, k_scale, v_scale,
    )
