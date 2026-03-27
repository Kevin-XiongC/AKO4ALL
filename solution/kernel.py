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
    torch::Tensor& q_output, double q_scale,
    torch::Tensor& k_cache, torch::Tensor& v_cache,
    torch::Tensor& out_loc, double k_scale, double v_scale);

void compute_cos_sin_cache(
    torch::Tensor& cache, int64_t max_pos, int64_t rotary_dim,
    double base, double factor, double low, double high);
"""

    module = load_inline(
        name="fused_qknorm_rope_store",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=["fused_qk_norm_rope_store", "compute_cos_sin_cache"],
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
    """Compute cos_sin_cache using CUDA powf+sincosf, pre-baked with attention_factor."""
    key = (base, rotary_dim, factor, low, high, attention_factor, device)
    if key not in _cos_sin_cache:
        mod = get_module()
        cache = torch.empty(_MAX_POS, rotary_dim, dtype=torch.float32, device=device)
        mod.compute_cos_sin_cache(cache, _MAX_POS, rotary_dim, base, factor, low, high)
        if attention_factor != 1.0:
            cache.mul_(attention_factor)
        torch.cuda.synchronize()
        _cos_sin_cache[key] = cache
    return _cos_sin_cache[key]


def fused_qk_norm_rope_store(
    qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
    q_weight, k_weight, base, is_neox, position_ids,
    factor, low, high, attention_factor, rotary_dim,
    q_output, q_scale, k_cache, v_cache, out_loc, k_scale, v_scale,
):
    cos_sin_cache = _get_cos_sin_cache(base, rotary_dim, factor, low, high, attention_factor, qkv.device)
    mod = get_module()
    mod.fused_qk_norm_rope_store(
        qkv, num_heads_q, num_heads_k, num_heads_v, head_dim,
        eps, q_weight, k_weight, is_neox, position_ids,
        rotary_dim, cos_sin_cache,
        q_output, q_scale, k_cache, v_cache, out_loc, k_scale, v_scale,
    )
