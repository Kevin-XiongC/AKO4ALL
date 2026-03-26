"""
Triton kernels for MoE token scatter/gather with DeepGEMM m-offset layout.

Scatter (moe_align_and_scatter):
    Reorders hidden_states by expert assignment into 128-aligned m-offset
    layout for deep_gemm.m_grouped_fp8_gemm_nt_moffset.

Gather (moe_gather):
    Reverses the m-offset layout after GEMM, weighting by topk_weights and
    accumulating back to original token positions.

Usage:
    sorted_hidden, packed_layout, output_index = moe_align_and_scatter(
        hidden_states, topk_ids, num_groups, start_expert, max_total_M
    )
    # ... DeepGEMM GEMM on sorted_hidden → gemm_output (m-offset layout) ...
    output = moe_gather(gemm_output, topk_weights, output_index)
"""

import math
import torch
import triton
import triton.language as tl

ALIGNMENT = 128


# ---------------------------------------------------------------------------
# Kernel 1 — fused histogram + prefix-sum  (single program, grid=1)
# ---------------------------------------------------------------------------

@triton.jit
def _count_and_compute_layout_kernel(
    topk_ids_ptr,
    packed_layout_ptr,
    num_elements,
    start_expert,
    num_groups,
    BLOCK_G: tl.constexpr,
    ALIGNMENT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    g_offs = tl.arange(0, BLOCK_G)
    g_mask = g_offs < num_groups
    counts = tl.zeros([BLOCK_G], dtype=tl.int32)

    for start in range(NUM_ITERS):
        offs = start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_elements
        expert_ids = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
        local_ids = expert_ids - start_expert
        valid = mask & (local_ids >= 0) & (local_ids < num_groups)
        safe_ids = tl.where(valid, local_ids, 0)
        counts += tl.histogram(safe_ids, BLOCK_G, mask=valid)

    aligned = ((counts + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    offsets = tl.cumsum(aligned, axis=0) - aligned

    tl.store(packed_layout_ptr + g_offs, offsets, mask=g_mask)
    tl.store(packed_layout_ptr + num_groups + g_offs, counts, mask=g_mask)


# ---------------------------------------------------------------------------
# Kernel 2 — scatter tokens (ep_scatter style: per-token, full-row copy)
# ---------------------------------------------------------------------------

@triton.jit
def _scatter_tokens_kernel(
    hidden_states_ptr,
    sorted_hidden_ptr,
    topk_ids_ptr,
    packed_layout_ptr,
    write_counters_ptr,
    output_index_ptr,
    num_tokens,
    topk: tl.constexpr,
    start_expert,
    num_groups,
    stride_hs_m,
    stride_sh_m,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token = tl.program_id(0)
    grid_size = tl.num_programs(0)

    h_offs = tl.arange(0, HIDDEN_SIZE_PAD)
    h_mask = h_offs < HIDDEN_SIZE

    for token_idx_i32 in range(start_token, num_tokens, grid_size):
        token_idx = token_idx_i32.to(tl.int64)

        in_data = tl.load(
            hidden_states_ptr + token_idx * stride_hs_m + h_offs,
            mask=h_mask,
        )

        topk_base = token_idx_i32 * topk
        for k in range(topk):
            expert_id = tl.load(topk_ids_ptr + topk_base + k)
            local_id = expert_id - start_expert

            if local_id >= 0 and local_id < num_groups:
                pos = tl.atomic_add(write_counters_ptr + local_id, 1)
                m_offset = tl.load(packed_layout_ptr + local_id)
                dst_row = (m_offset + pos).to(tl.int64)

                tl.store(output_index_ptr + topk_base + k, (m_offset + pos))
                tl.store(
                    sorted_hidden_ptr + dst_row * stride_sh_m + h_offs,
                    in_data,
                    mask=h_mask,
                )
            else:
                tl.store(output_index_ptr + topk_base + k, -1)


# ---------------------------------------------------------------------------
# Kernel 3 — gather (output_index-only, no topk_ids dependency)
# ---------------------------------------------------------------------------

@triton.jit
def _gather_tokens_kernel(
    gemm_output_ptr,
    topk_weights_ptr,
    output_index_ptr,
    output_ptr,
    num_tokens,
    topk: tl.constexpr,
    stride_gemm_m,
    stride_out_m,
    BLOCK_D: tl.constexpr,
):
    block_idx = tl.program_id(0).to(tl.int64)
    start_token = tl.program_id(1)
    grid_tokens = tl.num_programs(1)

    d_offs = tl.arange(0, BLOCK_D)

    for token_idx_i32 in range(start_token, num_tokens, grid_tokens):
        token_idx = token_idx_i32.to(tl.int64)
        topk_base = token_idx_i32 * topk

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for k in tl.static_range(topk):
            src_row_i32 = tl.load(output_index_ptr + topk_base + k)

            if src_row_i32 >= 0:
                src_row = src_row_i32.to(tl.int64)
                weight = tl.load(topk_weights_ptr + topk_base + k)

                val = tl.load(
                    gemm_output_ptr
                    + src_row * stride_gemm_m
                    + block_idx * BLOCK_D
                    + d_offs,
                )
                acc += val.to(tl.float32) * weight

        tl.store(
            output_ptr
            + token_idx * stride_out_m
            + block_idx * BLOCK_D
            + d_offs,
            acc.to(output_ptr.dtype.element_ty),
        )


@triton.jit
def _gather_tokens_tiled_kernel(
    gemm_output_ptr,
    topk_weights_ptr,
    output_index_ptr,
    output_ptr,
    num_tokens,
    topk: tl.constexpr,
    stride_gemm_m,
    stride_out_m,
    OUT_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    start_token = tl.program_id(0)
    grid_tokens = tl.num_programs(0)

    for token_idx_i32 in range(start_token, num_tokens, grid_tokens):
        token_idx = token_idx_i32.to(tl.int64)
        topk_base = token_idx_i32 * topk

        for d_block in tl.static_range(NUM_D_BLOCKS):
            d_offs = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
            acc = tl.zeros([BLOCK_D], dtype=tl.float32)

            for k in tl.static_range(topk):
                src_row_i32 = tl.load(output_index_ptr + topk_base + k)

                if src_row_i32 >= 0:
                    src_row = src_row_i32.to(tl.int64)
                    weight = tl.load(topk_weights_ptr + topk_base + k)

                    val = tl.load(
                        gemm_output_ptr
                        + src_row * stride_gemm_m
                        + d_offs,
                    )
                    acc += val.to(tl.float32) * weight

            tl.store(
                output_ptr
                + token_idx * stride_out_m
                + d_offs,
                acc.to(output_ptr.dtype.element_ty),
            )


# ---------------------------------------------------------------------------
# Python Wrappers
# ---------------------------------------------------------------------------

def moe_align_and_scatter(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_groups: int,
    start_expert: int,
    max_total_M: int | None = None,
    alignment: int = ALIGNMENT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reorder tokens by expert assignment for DeepGEMM m-offset layout.

    Args:
        hidden_states: [bs, hidden_dim] input activations (bf16)
        topk_ids:      [bs, topk]       expert indices per token (int32)
        num_groups:    number of local experts
        start_expert:  global id of the first local expert
        max_total_M:   pre-allocated M dimension (computed if None)
        alignment:     row alignment (default 128)

    Returns:
        sorted_hidden: [max_total_M, hidden_dim]  reordered activations
        packed_layout: [2 * num_groups]            [m_offsets | m_counts]
        output_index:  [bs * topk]                 reverse mapping for gather
                       (-1 for non-local experts)
    """
    bs, hidden_dim = hidden_states.shape
    topk = topk_ids.shape[1]
    num_elements = bs * topk
    device = hidden_states.device

    if max_total_M is None:
        max_total_M = num_elements + num_groups * (alignment - 1)

    flat_topk_ids = topk_ids.reshape(-1).contiguous()

    packed_layout = torch.empty(2 * num_groups, dtype=torch.int32, device=device)
    BLOCK_SIZE = 1024
    BLOCK_G = triton.next_power_of_2(num_groups)
    NUM_ITERS = math.ceil(num_elements / BLOCK_SIZE)
    _count_and_compute_layout_kernel[(1,)](
        flat_topk_ids, packed_layout,
        num_elements, start_expert, num_groups,
        BLOCK_G=BLOCK_G, ALIGNMENT=alignment,
        BLOCK_SIZE=BLOCK_SIZE, NUM_ITERS=NUM_ITERS,
    )

    sorted_hidden = torch.empty(
        max_total_M, hidden_dim, device=device, dtype=hidden_states.dtype,
    )
    write_counters = torch.zeros(num_groups, dtype=torch.int32, device=device)
    output_index = torch.full((bs * topk,), -1, dtype=torch.int32, device=device)

    HIDDEN_SIZE_PAD = triton.next_power_of_2(hidden_dim)
    grid_size = min(bs, 1024 * 8)
    _scatter_tokens_kernel[(grid_size,)](
        hidden_states, sorted_hidden, flat_topk_ids,
        packed_layout, write_counters, output_index,
        bs, topk, start_expert, num_groups,
        hidden_states.stride(0), sorted_hidden.stride(0),
        HIDDEN_SIZE=hidden_dim, HIDDEN_SIZE_PAD=HIDDEN_SIZE_PAD,
        num_warps=8,
    )

    return sorted_hidden, packed_layout, output_index


def moe_gather(
    gemm_output: torch.Tensor,
    topk_weights: torch.Tensor,
    output_index: torch.Tensor,
) -> torch.Tensor:
    """
    Gather GEMM results back to original token order with gating weights.

    Args:
        gemm_output:   [max_total_M, out_dim]  GEMM output in m-offset layout
        topk_weights:  [bs, topk]              gating weights (float32)
        output_index:  [bs * topk]             from moe_align_and_scatter

    Returns:
        output: [bs, out_dim]  weighted sum over local experts per token
    """
    bs = topk_weights.shape[0]
    topk = topk_weights.shape[1]
    out_dim = gemm_output.shape[1]

    flat_weights = topk_weights.reshape(-1).contiguous()
    flat_index = output_index.reshape(-1).contiguous()

    output = torch.zeros(bs, out_dim, device=gemm_output.device, dtype=gemm_output.dtype)

    # Try different BLOCK_D values for best performance
    if out_dim % 512 == 0:
        BLOCK_D = 512
        num_warps = 4
    elif out_dim % 256 == 0:
        BLOCK_D = 256
        num_warps = 4
    elif out_dim % 128 == 0:
        BLOCK_D = 128
        num_warps = 2
    else:
        BLOCK_D = 64
        num_warps = 2
    assert out_dim % BLOCK_D == 0, f"out_dim={out_dim} must be divisible by BLOCK_D={BLOCK_D}"

    grid = (out_dim // BLOCK_D, min(bs, 1024))
    _gather_tokens_kernel[grid](
        gemm_output, flat_weights, flat_index,
        output,
        bs, topk,
        gemm_output.stride(0), output.stride(0),
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )

    return output
