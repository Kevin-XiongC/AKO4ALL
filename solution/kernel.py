"""
Fused QK-Norm + RoPE + FP8-Cast + KV-Store CUDA kernel.
Uses precomputed cos_sin_cache for RoPE.
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
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v,
    int64_t head_dim, double eps,
    torch::Tensor& q_weight, torch::Tensor& k_weight,
    bool is_neox, torch::Tensor& position_ids,
    int64_t rotary_dim,
    torch::Tensor& cos_sin_cache,
    torch::Tensor& q_output, torch::Tensor& q_scale,
    torch::Tensor& k_cache, torch::Tensor& v_cache,
    torch::Tensor& out_loc, torch::Tensor& k_scale, torch::Tensor& v_scale);

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


_cos_sin_cache = {}
_MAX_POS = 131072


def _get_cos_sin_cache(base, rotary_dim, factor, low, high, attention_factor, device):
    """Generate cos_sin_cache matching sglang RotaryEmbedding._compute_cos_sin_cache."""
    key = (base, rotary_dim, factor, low, high, attention_factor, device)
    if key not in _cos_sin_cache:
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        if factor != 1.0:
            half_rd = rotary_dim // 2
            for i in range(half_rd):
                freq = inv_freq[i].item()
                interp = freq / factor
                high_adj = high if abs(low - high) > 1e-6 else high + 0.001
                linear = (i - low) / (high_adj - low)
                ramp = min(max(linear, 0.0), 1.0)
                ext_factor = 1.0 - ramp
                inv_freq[i] = interp * (1.0 - ext_factor) + freq * ext_factor

        t = torch.arange(_MAX_POS, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * attention_factor
        sin = freqs.sin() * attention_factor
        cache = torch.cat((cos, sin), dim=-1).to(device).to(torch.bfloat16)  # [max_pos, rotary_dim], BF16
        _cos_sin_cache[key] = cache
    return _cos_sin_cache[key]


def fused_qk_norm_rope_store(
    qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
    q_weight, k_weight, base, is_neox, position_ids,
    factor, low, high, attention_factor, rotary_dim,
    q_output, q_scale=None, k_cache=None, v_cache=None, out_loc=None,
    k_scale=None, v_scale=None,
):
    _default = lambda s: s if s is not None else torch.ones(1, dtype=torch.float32, device=qkv.device)
    q_scale = _default(q_scale)
    k_scale = _default(k_scale)
    v_scale = _default(v_scale)
    cos_sin_cache = _get_cos_sin_cache(base, rotary_dim, factor, low, high, attention_factor, qkv.device)
    mod = get_module()
    mod.fused_qk_norm_rope_store(
        qkv, num_heads_q, num_heads_k, num_heads_v, head_dim,
        eps, q_weight, k_weight, is_neox, position_ids,
        rotary_dim, cos_sin_cache,
        q_output, q_scale, k_cache, v_cache, out_loc, k_scale, v_scale,
    )
