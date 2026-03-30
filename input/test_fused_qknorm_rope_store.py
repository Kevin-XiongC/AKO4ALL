"""
Correctness tests for the fused QK-Norm + RoPE + FP8-Cast + KV-Store kernel.

Reference: use the existing sgl_kernel.fused_qk_norm_rope for Q/K norm+rope,
then do FP8 cast + scatter write in PyTorch. This ensures the norm+rope math
matches exactly (same CUDA float32 intrinsics).
"""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_qknorm_rope_store import fused_qk_norm_rope_store

# Import the existing unfused kernel for reference
from sgl_kernel import fused_qk_norm_rope as sgl_fused_qk_norm_rope


def reference_unfused(
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
    q_scale: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out_loc: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
):
    """
    Reference: existing fused_qk_norm_rope (in-place on qkv) + separate FP8 cast + store.
    This mirrors the actual unfused production path.
    """
    # Step 1: Apply QK norm + RoPE in-place on qkv (uses the existing sgl_kernel CUDA kernel)
    sgl_fused_qk_norm_rope(
        qkv, num_heads_q, num_heads_k, num_heads_v, head_dim,
        eps, q_weight, k_weight, base, is_neox, position_ids,
        factor, low, high, attention_factor, rotary_dim,
    )

    # Step 2: Split out Q, K and V
    q_size = num_heads_q * head_dim
    kv_size = num_heads_k * head_dim
    q = qkv[:, :q_size].clone()
    k = qkv[:, q_size:q_size + kv_size].clone()
    v = qkv[:, q_size + kv_size:].clone()

    # Step 3: FP8 cast (matches memory_pool.py: div by scale tensor, then .to(fp8))
    q_fp8 = q.float().div(q_scale).to(torch.float8_e4m3fn)
    k_fp8 = k.float().div(k_scale).to(torch.float8_e4m3fn)
    v_fp8 = v.float().div(v_scale).to(torch.float8_e4m3fn)

    # Step 4: Reinterpret as uint8 and scatter write
    q_u8 = q_fp8.view(torch.uint8).view(-1, q_size)
    k_u8 = k_fp8.view(torch.uint8).view(-1, kv_size)
    v_u8 = v_fp8.view(torch.uint8).view(-1, kv_size)

    q_output.copy_(q_u8)
    num_tokens = qkv.shape[0]
    for i in range(num_tokens):
        slot = out_loc[i].item()
        k_cache[slot] = k_u8[i]
        v_cache[slot] = v_u8[i]


# ============================================================================
# Parametrized test
# ============================================================================

head_dims = [64, 128]
num_heads_groups = [
    (12, 1, 1),   # GLM4.6 TP8
    (32, 8, 8),   # Qwen3-like
]
num_tokens_list = [1, 4, 32, 256]
is_neox_list = [True]
k_scale_list = [1.0, 0.8]
CACHE_SIZE = 4096


@pytest.mark.parametrize("head_dim", head_dims)
@pytest.mark.parametrize("num_heads_group", num_heads_groups)
@pytest.mark.parametrize("num_tokens", num_tokens_list)
@pytest.mark.parametrize("is_neox", is_neox_list)
@pytest.mark.parametrize("k_scale", k_scale_list)
def test_fused_qknorm_rope_store(
    head_dim, num_heads_group, num_tokens, is_neox, k_scale,
):
    """
    Compare the fused kernel against existing sgl_kernel.fused_qk_norm_rope + PyTorch FP8 cast + store.

    Checks:
      1. Q output (BF16) matches exactly (both kernels use same norm+rope math)
      2. K cache (FP8 as uint8) matches within 1 ULP
      3. V cache (FP8 as uint8) matches within 1 ULP
    """
    device = "cuda"
    torch.manual_seed(42)

    num_heads_q, num_heads_k, num_heads_v = num_heads_group
    hidden_size = (num_heads_q + num_heads_k + num_heads_v) * head_dim
    q_dim = num_heads_q * head_dim
    kv_dim = num_heads_k * head_dim

    q_scale = torch.tensor([k_scale], dtype=torch.float32, device=device)
    k_scale_t = torch.tensor([k_scale], dtype=torch.float32, device=device)
    v_scale_t = torch.tensor([k_scale], dtype=torch.float32, device=device)

    # Input tensors
    qkv_fused = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    qkv_ref = qkv_fused.clone()

    position_ids = (torch.arange(num_tokens, device=device) + 100).to(torch.int32)
    q_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 2.0
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 2.0

    # Q output buffers
    q_output_fused = torch.zeros(num_tokens, q_dim, dtype=torch.uint8, device=device)
    q_output_ref = torch.zeros(num_tokens, q_dim, dtype=torch.uint8, device=device)

    # KV cache buffers
    k_cache_fused = torch.zeros(CACHE_SIZE, kv_dim, dtype=torch.uint8, device=device)
    v_cache_fused = torch.zeros(CACHE_SIZE, kv_dim, dtype=torch.uint8, device=device)
    k_cache_ref = torch.zeros(CACHE_SIZE, kv_dim, dtype=torch.uint8, device=device)
    v_cache_ref = torch.zeros(CACHE_SIZE, kv_dim, dtype=torch.uint8, device=device)

    out_loc = torch.randperm(CACHE_SIZE, device=device)[:num_tokens].to(torch.int32)

    eps = 1e-5
    base = 10000.0
    factor, low_f, high_f, attn_factor = 1.0, 0.0, 0.0, 1.0
    rotary_dim = head_dim

    # ---- Run fused kernel ----
    fused_qk_norm_rope_store(
        qkv_fused, num_heads_q, num_heads_k, num_heads_v, head_dim,
        eps, q_weight, k_weight, base, is_neox, position_ids,
        factor, low_f, high_f, attn_factor, rotary_dim,
        q_output_fused, q_scale,
        k_cache_fused, v_cache_fused, out_loc,
        k_scale_t, v_scale_t,
    )

    # ---- Run reference (existing kernel + PyTorch FP8 cast + store) ----
    reference_unfused(
        qkv_ref, num_heads_q, num_heads_k, num_heads_v, head_dim,
        eps, q_weight, k_weight, base, is_neox, position_ids,
        factor, low_f, high_f, attn_factor, rotary_dim,
        q_output_ref, q_scale,
        k_cache_ref, v_cache_ref, out_loc,
        k_scale_t, v_scale_t,
    )

    # ---- Check Q output (FP8 as uint8) ----
    for i in range(num_tokens):
        diff = (q_output_fused[i].int() - q_output_ref[i].int()).abs()
        max_diff = diff.max().item()
        assert max_diff <= 1, (
            f"Q output mismatch at token {i}: max uint8 diff = {max_diff}"
        )

    # ---- Check K cache ----
    for i in range(num_tokens):
        slot = out_loc[i].item()
        diff = (k_cache_fused[slot].int() - k_cache_ref[slot].int()).abs()
        max_diff = diff.max().item()
        assert max_diff <= 1, (
            f"K cache mismatch at token {i}, slot {slot}: max uint8 diff = {max_diff}"
        )

    # ---- Check V cache ----
    for i in range(num_tokens):
        slot = out_loc[i].item()
        diff = (v_cache_fused[slot].int() - v_cache_ref[slot].int()).abs()
        max_diff = diff.max().item()
        assert max_diff <= 1, (
            f"V cache mismatch at token {i}, slot {slot}: max uint8 diff = {max_diff}"
        )

    print(f"PASSED: hd={head_dim}, heads=({num_heads_q},{num_heads_k},{num_heads_v}), "
          f"tok={num_tokens}, neox={is_neox}, k_scale={k_scale}")


def test_v_store_only():
    """
    Verify V values (no norm/rope) are correctly FP8-cast and stored.
    """
    device = "cuda"
    torch.manual_seed(0)

    num_heads_q, num_heads_k, num_heads_v = 4, 1, 1
    head_dim = 64
    num_tokens = 2
    hidden_size = (num_heads_q + num_heads_k + num_heads_v) * head_dim
    q_dim = num_heads_q * head_dim
    kv_dim = num_heads_k * head_dim

    qkv = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    v_section = qkv[:, (num_heads_q + num_heads_k) * head_dim:].clone()

    q_output = torch.zeros(num_tokens, q_dim, dtype=torch.uint8, device=device)
    k_cache = torch.zeros(128, kv_dim, dtype=torch.uint8, device=device)
    v_cache = torch.zeros(128, kv_dim, dtype=torch.uint8, device=device)
    out_loc = torch.tensor([10, 50], dtype=torch.int32, device=device)

    position_ids = torch.tensor([0, 1], dtype=torch.int32, device=device)
    q_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    k_weight = torch.ones(head_dim, dtype=torch.bfloat16, device=device)
    v_scale = torch.ones(1, dtype=torch.float32, device=device)

    fused_qk_norm_rope_store(
        qkv, num_heads_q, num_heads_k, num_heads_v, head_dim,
        1e-5, q_weight, k_weight, 10000.0, True, position_ids,
        1.0, 0.0, 0.0, 1.0, head_dim,
        q_output, None,
        k_cache, v_cache, out_loc, None, v_scale,
    )

    # V path: bf16_value → float / v_scale → fp8
    v_fp8_expected = (v_section.float() / v_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    v_fp8_expected = v_fp8_expected.view(num_tokens, kv_dim)

    for i in range(num_tokens):
        slot = out_loc[i].item()
        diff = (v_cache[slot].int() - v_fp8_expected[i].int()).abs()
        max_diff = diff.max().item()
        assert max_diff <= 1, f"V store mismatch at token {i}: max diff = {max_diff}"

    print("PASSED: test_v_store_only")


def test_large_batch():
    """Test with a larger batch to exercise the warp scheduling."""
    device = "cuda"
    torch.manual_seed(123)

    num_heads_q, num_heads_k, num_heads_v = 12, 1, 1
    head_dim = 128
    num_tokens = 512
    hidden_size = (num_heads_q + num_heads_k + num_heads_v) * head_dim
    q_dim = num_heads_q * head_dim
    kv_dim = num_heads_k * head_dim
    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    v_scale = torch.ones(1, dtype=torch.float32, device=device)
    q_scale = torch.ones(1, dtype=torch.float32, device=device)

    qkv_fused = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    qkv_ref = qkv_fused.clone()

    position_ids = torch.arange(num_tokens, device=device, dtype=torch.int32)
    q_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)

    q_output_fused = torch.zeros(num_tokens, q_dim, dtype=torch.uint8, device=device)
    q_output_ref = torch.zeros(num_tokens, q_dim, dtype=torch.uint8, device=device)
    k_cache_fused = torch.zeros(8192, kv_dim, dtype=torch.uint8, device=device)
    v_cache_fused = torch.zeros(8192, kv_dim, dtype=torch.uint8, device=device)
    k_cache_ref = torch.zeros(8192, kv_dim, dtype=torch.uint8, device=device)
    v_cache_ref = torch.zeros(8192, kv_dim, dtype=torch.uint8, device=device)

    out_loc = torch.randperm(8192, device=device)[:num_tokens].to(torch.int32)

    fused_qk_norm_rope_store(
        qkv_fused, num_heads_q, num_heads_k, num_heads_v, head_dim,
        1e-5, q_weight, k_weight, 10000.0, True, position_ids,
        1.0, 0.0, 0.0, 1.0, head_dim,
        q_output_fused, q_scale,
        k_cache_fused, v_cache_fused, out_loc, k_scale, v_scale,
    )

    reference_unfused(
        qkv_ref, num_heads_q, num_heads_k, num_heads_v, head_dim,
        1e-5, q_weight, k_weight, 10000.0, True, position_ids,
        1.0, 0.0, 0.0, 1.0, head_dim,
        q_output_ref, q_scale,
        k_cache_ref, v_cache_ref, out_loc, k_scale, v_scale,
    )

    # Q check (FP8 as uint8)
    for i in range(num_tokens):
        q_diff = (q_output_fused[i].int() - q_output_ref[i].int()).abs().max().item()
        assert q_diff <= 1, f"Q mismatch token {i}: diff={q_diff}"

    # K/V check
    for i in range(num_tokens):
        slot = out_loc[i].item()
        k_diff = (k_cache_fused[slot].int() - k_cache_ref[slot].int()).abs().max().item()
        v_diff = (v_cache_fused[slot].int() - v_cache_ref[slot].int()).abs().max().item()
        assert k_diff <= 1, f"K mismatch token {i}: diff={k_diff}"
        assert v_diff <= 1, f"V mismatch token {i}: diff={v_diff}"

    print("PASSED: test_large_batch (512 tokens)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
