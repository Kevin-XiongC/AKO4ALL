"""
Pure PyTorch reference: RMSNorm + RoPE + FP8-Cast + Scatter Store.

All Q/K computation in FP32 to match the fused CUDA kernel's precision.
cos_sin_cache stored as BF16, loaded as FP32 for computation.

Outputs: q_output (FP8 E4M3 uint8), k_cache (FP8), v_cache (FP8).
"""

import torch

_cos_sin_cache = {}
_MAX_POS = 131072


def _get_cos_sin_cache(base, rotary_dim, factor, low, high, attention_factor, device):
    key = (base, rotary_dim, factor, low, high, attention_factor, device)
    if key not in _cos_sin_cache:
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        if factor != 1.0:
            for i in range(rotary_dim // 2):
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
        cache = torch.cat((cos, sin), dim=-1).to(device).to(torch.bfloat16)
        _cos_sin_cache[key] = cache
    return _cos_sin_cache[key]


def _apply_rotary_emb(x, cos, sin, is_neox_style):
    """
    x: [num_tokens, num_heads, head_size]  (FP32)
    cos: [num_tokens, 1, head_size // 2]   (FP32)
    sin: [num_tokens, 1, head_size // 2]   (FP32)
    """
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def fused_qk_norm_rope_store(
    qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
    q_weight, k_weight, base, is_neox, position_ids,
    factor, low, high, attention_factor, rotary_dim,
    q_output, q_scale, k_cache, v_cache, out_loc, k_scale, v_scale,
):
    num_tokens = qkv.shape[0]
    q_size = num_heads_q * head_dim
    kv_size = num_heads_k * head_dim

    # Extract Q, K, V → FP32
    q = qkv[:, :q_size].float().view(num_tokens, num_heads_q, head_dim)
    k = qkv[:, q_size:q_size + kv_size].float().view(num_tokens, num_heads_k, head_dim)
    v = qkv[:, q_size + kv_size:]

    # RMSNorm in FP32
    rms_q = torch.sqrt(q.pow(2).mean(dim=-1, keepdim=True) + eps)
    q = q / rms_q * q_weight.float()

    rms_k = torch.sqrt(k.pow(2).mean(dim=-1, keepdim=True) + eps)
    k = k / rms_k * k_weight.float()

    # RoPE in FP32 (cos_sin_cache is BF16, load as FP32)
    cos_sin_cache = _get_cos_sin_cache(
        base, rotary_dim, factor, low, high, attention_factor, qkv.device)
    cos_sin = cos_sin_cache[position_ids.long()].float()
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(1)  # [num_tokens, 1, rotary_dim/2]
    sin = sin.unsqueeze(1)

    q_rot = _apply_rotary_emb(q[..., :rotary_dim], cos, sin, is_neox)
    q = torch.cat((q_rot, q[..., rotary_dim:]), dim=-1)

    k_rot = _apply_rotary_emb(k[..., :rotary_dim], cos, sin, is_neox)
    k = torch.cat((k_rot, k[..., rotary_dim:]), dim=-1)

    # Flatten
    q = q.view(num_tokens, q_size)
    k = k.view(num_tokens, kv_size)

    # Write back to qkv as BF16 (for benchmark verification)
    qkv[:, :q_size] = q.to(torch.bfloat16)
    qkv[:, q_size:q_size + kv_size] = k.to(torch.bfloat16)

    # FP8 cast from FP32 + store
    q_fp8 = q.div(q_scale).to(torch.float8_e4m3fn).view(torch.uint8).view(-1, q_size)
    k_fp8 = k.div(k_scale).to(torch.float8_e4m3fn).view(torch.uint8).view(-1, kv_size)
    v_fp8 = v.float().div(v_scale).to(torch.float8_e4m3fn).view(torch.uint8).view(-1, kv_size)

    q_output.copy_(q_fp8)
    k_cache[out_loc.long()] = k_fp8
    v_cache[out_loc.long()] = v_fp8
