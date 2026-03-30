"""
Python wrapper for the fused QK-Norm + RoPE + FP8-Cast + KV-Store CUDA kernel.

Uses torch.utils.cpp_extension.load_inline to JIT-compile the kernel.
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
    torch::Tensor& q_scale,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor& out_loc,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale);
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
    qkv: torch.Tensor,          # [num_tokens, (nq+nk+nv)*head_dim]  BF16
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,     # [head_dim]
    k_weight: torch.Tensor,     # [head_dim]
    base: float,
    is_neox: bool,
    position_ids: torch.Tensor, # [num_tokens]  INT32
    factor: float,
    low: float,
    high: float,
    attention_factor: float,
    rotary_dim: int,
    q_output: torch.Tensor,     # [num_tokens, num_heads_q * head_dim]  UINT8
    q_scale: torch.Tensor = None,  # [1] float32
    k_cache: torch.Tensor = None,
    v_cache: torch.Tensor = None,
    out_loc: torch.Tensor = None,
    k_scale: torch.Tensor = None,  # [1] float32
    v_scale: torch.Tensor = None,  # [1] float32
):
    """
    Fused QK-Norm + RoPE + FP8-Cast + KV-Store.

    Outputs:
      - q_output: [num_tokens, num_heads_q * head_dim] uint8 (FP8 E4M3)
      - k_cache:  scatter-written at out_loc positions, uint8 (FP8 E4M3)
      - v_cache:  scatter-written at out_loc positions, uint8 (FP8 E4M3)

    Scales are float32 single-element tensors, default torch.ones(1).
    """
    _default = lambda s: s if s is not None else torch.ones(1, dtype=torch.float32, device=qkv.device)
    q_scale = _default(q_scale)
    k_scale = _default(k_scale)
    v_scale = _default(v_scale)
    mod = get_module()
    mod.fused_qk_norm_rope_store(
        qkv, num_heads_q, num_heads_k, num_heads_v, head_dim,
        eps, q_weight, k_weight, base, is_neox, position_ids,
        factor, low, high, attention_factor, rotary_dim,
        q_output, q_scale,
        k_cache, v_cache, out_loc, k_scale, v_scale,
    )
