"""
Triton kernels for MoE token scatter/gather with DeepGEMM m-offset layout.

Scatter outputs FP8 (float8_e4m3fn) quantized activations + per-row scales,
fusing the per-token quantization into the scatter to avoid a second data pass
and halve write bandwidth.
"""

import math
import torch
import triton
import triton.language as tl

ALIGNMENT = 128
FP8_E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# Kernel 1 — fused histogram + prefix-sum  (single program, grid=1)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[triton.Config({}, num_warps=8, num_stages=1)],
    key=['BLOCK_SIZE'],
)
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

    TOTAL: tl.constexpr = BLOCK_SIZE * NUM_ITERS
    if TOTAL <= 16384:
        # Single-pass: load all elements at once (avoids loop overhead)
        all_offs = tl.arange(0, TOTAL)
        mask = all_offs < num_elements
        expert_ids = tl.load(topk_ids_ptr + all_offs, mask=mask, other=-1)
        local_ids = expert_ids - start_expert
        valid = mask & (local_ids >= 0) & (local_ids < num_groups)
        safe_ids = tl.where(valid, local_ids, 0)
        counts = tl.histogram(safe_ids, BLOCK_G, mask=valid)
    else:
        # Multi-pass for large batch sizes
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
# Kernel 2a — position allocation only (lightweight, 1D grid)
# ---------------------------------------------------------------------------

@triton.jit
def _scatter_positions_kernel(
    topk_ids_ptr,
    packed_layout_ptr,
    write_counters_ptr,
    output_index_ptr,
    num_tokens,
    topk: tl.constexpr,
    start_expert,
    num_groups,
):
    start_token = tl.program_id(0)
    grid_size = tl.num_programs(0)

    for token_idx_i32 in range(start_token, num_tokens, grid_size):
        topk_base = token_idx_i32 * topk

        for k in range(topk):
            expert_id = tl.load(topk_ids_ptr + topk_base + k)
            local_id = expert_id - start_expert

            if local_id >= 0 and local_id < num_groups:
                pos = tl.atomic_add(write_counters_ptr + local_id, 1)
                m_offset = tl.load(packed_layout_ptr + local_id)
                tl.store(output_index_ptr + topk_base + k,
                         (m_offset + pos))
            else:
                tl.store(output_index_ptr + topk_base + k, -1)


# ---------------------------------------------------------------------------
# Kernel 2b — fused FP8 quantize + scatter data (1D grid, full row)
# ---------------------------------------------------------------------------

@triton.jit
def _scatter_quantize_kernel(
    hidden_states_ptr,
    sorted_hidden_ptr,
    sorted_scales_ptr,
    output_index_ptr,
    num_tokens,
    topk: tl.constexpr,
    stride_hs_m,
    stride_sh_m,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    FP8_MAX: tl.constexpr,
    GROUP_SIZE: tl.constexpr = 0,
    NUM_GROUPS_PER_ROW: tl.constexpr = 1,
    NUM_GROUPS_PER_ROW_PAD: tl.constexpr = 1,
    STRIDE_SC_M: tl.constexpr = 1,
):
    start_token = tl.program_id(0)
    grid_size = tl.num_programs(0)

    h_offs = tl.arange(0, HIDDEN_SIZE_PAD)
    h_mask = h_offs < HIDDEN_SIZE

    for token_idx_i32 in range(start_token, num_tokens, grid_size):
        token_idx = token_idx_i32.to(tl.int64)
        topk_base = token_idx_i32 * topk

        # Check if any local expert
        any_local: tl.int1 = False
        for kk in tl.static_range(topk):
            idx = tl.load(output_index_ptr + topk_base + kk)
            any_local |= (idx >= 0)

        if any_local:
            # Load bf16 hidden row
            in_data = tl.load(
                hidden_states_ptr + token_idx * stride_hs_m + h_offs,
                mask=h_mask,
                other=0.0,
            )

            in_f32 = in_data.to(tl.float32)

            if GROUP_SIZE > 0:
                # Per-group FP8 quantization
                in_2d = tl.reshape(in_f32, (NUM_GROUPS_PER_ROW_PAD, GROUP_SIZE))
                abs_2d = tl.abs(in_2d)
                max_per_group = tl.max(abs_2d, axis=1)
                scales = max_per_group / FP8_MAX
                scale_invs = tl.where(scales > 0.0, 1.0 / scales, 0.0)
                quantized_2d = in_2d * tl.reshape(scale_invs, (NUM_GROUPS_PER_ROW_PAD, 1))
                fp8_2d = quantized_2d.to(tl.float8e4nv)
                fp8_data = tl.reshape(fp8_2d, (HIDDEN_SIZE_PAD,))

                g_offs = tl.arange(0, NUM_GROUPS_PER_ROW_PAD)
                g_mask = g_offs < NUM_GROUPS_PER_ROW

                for k in tl.static_range(topk):
                    dst_row_i32 = tl.load(output_index_ptr + topk_base + k)
                    if dst_row_i32 >= 0:
                        dst_row = dst_row_i32.to(tl.int64)
                        tl.store(
                            sorted_hidden_ptr + dst_row * stride_sh_m + h_offs,
                            fp8_data,
                            mask=h_mask,
                        )
                        tl.store(
                            sorted_scales_ptr + dst_row * STRIDE_SC_M + g_offs,
                            scales,
                            mask=g_mask,
                        )
            else:
                # Per-token FP8 quantization
                max_val = tl.max(tl.where(h_mask, tl.abs(in_f32), 0.0))
                scale = max_val / FP8_MAX
                scale_inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
                fp8_data = (in_f32 * scale_inv).to(tl.float8e4nv)

                for k in tl.static_range(topk):
                    dst_row_i32 = tl.load(output_index_ptr + topk_base + k)
                    if dst_row_i32 >= 0:
                        dst_row = dst_row_i32.to(tl.int64)
                        tl.store(
                            sorted_hidden_ptr + dst_row * stride_sh_m + h_offs,
                            fp8_data,
                            mask=h_mask,
                        )
                        tl.store(sorted_scales_ptr + dst_row, scale)


# ---------------------------------------------------------------------------
# Kernel 2 — scatter tokens (original API, now with FP8 quantization)
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
    sorted_scales_ptr=None,
    FP8_MAX: tl.constexpr = 448,
    GROUP_SIZE: tl.constexpr = 0,
    NUM_GROUPS_PER_ROW: tl.constexpr = 1,
    NUM_GROUPS_PER_ROW_PAD: tl.constexpr = 1,
    STRIDE_SC_M: tl.constexpr = 1,
):
    start_token = tl.program_id(0)
    grid_size = tl.num_programs(0)

    h_offs = tl.arange(0, HIDDEN_SIZE_PAD)
    h_mask = h_offs < HIDDEN_SIZE

    for token_idx_i32 in range(start_token, num_tokens, grid_size):
        token_idx = token_idx_i32.to(tl.int64)
        topk_base = token_idx_i32 * topk

        # Pre-check: does this token have any local experts?
        any_local: tl.int32 = 0
        for kk in tl.static_range(topk):
            eid = tl.load(topk_ids_ptr + topk_base + kk)
            lid = eid - start_expert
            any_local |= ((lid >= 0) & (lid < num_groups)).to(tl.int32)

        if any_local != 0:
            in_data = tl.load(
                hidden_states_ptr + token_idx * stride_hs_m + h_offs,
                mask=h_mask,
                other=0.0,
                eviction_policy="evict_last",
            )

            in_f32 = in_data.to(tl.float32)

            if GROUP_SIZE > 0:
                # Per-group FP8 quantization
                in_2d = tl.reshape(in_f32, (NUM_GROUPS_PER_ROW_PAD, GROUP_SIZE))
                abs_2d = tl.abs(in_2d)
                max_per_group = tl.max(abs_2d, axis=1)
                scales = max_per_group / FP8_MAX
                scale_invs = tl.where(scales > 0.0, 1.0 / scales, 0.0)
                quantized_2d = in_2d * tl.reshape(scale_invs, (NUM_GROUPS_PER_ROW_PAD, 1))
                fp8_2d = quantized_2d.to(tl.float8e4nv)
                fp8_data = tl.reshape(fp8_2d, (HIDDEN_SIZE_PAD,))

                g_offs = tl.arange(0, NUM_GROUPS_PER_ROW_PAD)
                g_mask = g_offs < NUM_GROUPS_PER_ROW
            else:
                # Per-token FP8 quantization
                max_val = tl.max(tl.where(h_mask, tl.abs(in_f32), 0.0))
                scale = max_val / FP8_MAX
                scale_inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
                fp8_data = (in_f32 * scale_inv).to(tl.float8e4nv)

            for k in range(topk):
                expert_id = tl.load(topk_ids_ptr + topk_base + k)
                local_id = expert_id - start_expert

                if local_id >= 0 and local_id < num_groups:
                    m_offset = tl.load(packed_layout_ptr + local_id)
                    pos = tl.atomic_add(write_counters_ptr + local_id, 1)
                    dst_row = (m_offset + pos).to(tl.int64)

                    tl.store(output_index_ptr + topk_base + k,
                             (m_offset + pos))
                    tl.store(
                        sorted_hidden_ptr + dst_row * stride_sh_m + h_offs,
                        fp8_data,
                        mask=h_mask,
                        eviction_policy="evict_first",
                    )
                    if GROUP_SIZE > 0:
                        tl.store(
                            sorted_scales_ptr + dst_row * STRIDE_SC_M + g_offs,
                            scales,
                            mask=g_mask,
                            eviction_policy="evict_first",
                        )
                    else:
                        tl.store(sorted_scales_ptr + dst_row, scale,
                                 eviction_policy="evict_first")
                else:
                    tl.store(output_index_ptr + topk_base + k, -1)
        else:
            for k in tl.static_range(topk):
                tl.store(output_index_ptr + topk_base + k, -1)


# ---------------------------------------------------------------------------
# Kernel 3 — gather (unchanged — reads from gemm_output, not sorted_hidden)
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
    base_addr = block_idx * BLOCK_D

    for token_idx_i32 in range(start_token, num_tokens, grid_tokens):
        token_idx = token_idx_i32.to(tl.int64)
        topk_base = token_idx_i32 * topk

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for k in tl.static_range(topk):
            src_row_i32 = tl.load(output_index_ptr + topk_base + k,
                                  eviction_policy="evict_last")

            if src_row_i32 >= 0:
                src_row = src_row_i32.to(tl.int64)
                weight = tl.load(topk_weights_ptr + topk_base + k,
                                 eviction_policy="evict_last")

                val = tl.load(
                    gemm_output_ptr
                    + src_row * stride_gemm_m
                    + base_addr
                    + d_offs,
                    eviction_policy="evict_last",
                )
                acc += val.to(tl.float32) * weight

        tl.store(
            output_ptr
            + token_idx * stride_out_m
            + base_addr
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
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reorder tokens by expert assignment for DeepGEMM m-offset layout,
    with fused FP8 quantization (per-token or per-group).

    Args:
        group_size: If None, per-token quantization. Otherwise per-group.

    Returns:
        sorted_hidden: [max_total_M, hidden_dim]  FP8 e4m3fn quantized activations
        packed_layout: [2 * num_groups]            [m_offsets | m_counts]
        output_index:  [bs * topk]                 reverse mapping (-1 for non-local)
        sorted_scales: per-token: [max_total_M]  |  per-group: [max_total_M, hidden_dim // group_size]
    """
    bs, hidden_dim = hidden_states.shape
    topk = topk_ids.shape[1]
    num_elements = bs * topk
    device = hidden_states.device

    use_per_group = group_size is not None and group_size < hidden_dim
    if use_per_group:
        assert hidden_dim % group_size == 0
        num_scale_cols = hidden_dim // group_size

    if max_total_M is None:
        max_total_M = num_elements + num_groups * (alignment - 1)

    flat_topk_ids = topk_ids.view(-1)

    # Kernel 1: count + layout
    packed_layout = torch.empty(2 * num_groups, dtype=torch.int32, device=device)
    BLOCK_SIZE = min(4096, triton.next_power_of_2(num_elements))
    BLOCK_G = triton.next_power_of_2(num_groups)
    NUM_ITERS = math.ceil(num_elements / BLOCK_SIZE)
    _count_and_compute_layout_kernel[(1,)](
        flat_topk_ids, packed_layout,
        num_elements, start_expert, num_groups,
        BLOCK_G=BLOCK_G, ALIGNMENT=alignment,
        BLOCK_SIZE=BLOCK_SIZE, NUM_ITERS=NUM_ITERS,
    )

    # Allocate FP8 output + scales
    sorted_hidden = torch.empty(
        max_total_M, hidden_dim, device=device, dtype=torch.float8_e4m3fn,
    )
    if use_per_group:
        sorted_scales = torch.zeros(max_total_M, num_scale_cols, device=device, dtype=torch.float32)
    else:
        sorted_scales = torch.zeros(max_total_M, device=device, dtype=torch.float32)
    write_counters = torch.zeros(num_groups, dtype=torch.int32, device=device)
    output_index = torch.empty(bs * topk, dtype=torch.int32, device=device)

    # Kernel 2a: Position allocation
    pos_grid = min(bs, 1024)
    _scatter_positions_kernel[(pos_grid,)](
        flat_topk_ids, packed_layout, write_counters, output_index,
        bs, topk, start_expert, num_groups,
        num_warps=4,
    )

    # Kernel 2b: Fused FP8 quantize + data copy (1D, full row)
    HIDDEN_SIZE_PAD = triton.next_power_of_2(hidden_dim)
    grid_size = min(bs, 1024)

    if use_per_group:
        gs = group_size
        NGR = num_scale_cols
        NGR_PAD = HIDDEN_SIZE_PAD // gs
    else:
        gs = 0
        NGR = 1
        NGR_PAD = 1

    _scatter_quantize_kernel[(grid_size,)](
        hidden_states, sorted_hidden, sorted_scales, output_index,
        bs, topk,
        hidden_states.stride(0), sorted_hidden.stride(0),
        HIDDEN_SIZE=hidden_dim, HIDDEN_SIZE_PAD=HIDDEN_SIZE_PAD,
        FP8_MAX=FP8_E4M3_MAX,
        GROUP_SIZE=gs,
        NUM_GROUPS_PER_ROW=NGR,
        NUM_GROUPS_PER_ROW_PAD=NGR_PAD,
        STRIDE_SC_M=sorted_scales.stride(0) if use_per_group else 1,
        num_warps=8,
    )

    return sorted_hidden, packed_layout, output_index, sorted_scales


def moe_gather(
    gemm_output: torch.Tensor,
    topk_weights: torch.Tensor,
    output_index: torch.Tensor,
) -> torch.Tensor:
    """
    Gather GEMM results back to original token order with gating weights.
    Unchanged — reads from gemm_output, not sorted_hidden.
    """
    bs = topk_weights.shape[0]
    topk = topk_weights.shape[1]
    out_dim = gemm_output.shape[1]

    flat_weights = topk_weights.view(-1)
    flat_index = output_index.view(-1)

    output = torch.empty(bs, out_dim, device=gemm_output.device, dtype=gemm_output.dtype)

    if out_dim % 1024 == 0:
        BLOCK_D = 1024
        num_warps = 4
    elif out_dim % 512 == 0:
        BLOCK_D = 512
        num_warps = 4
    elif out_dim % 128 == 0:
        BLOCK_D = 128
        num_warps = 2
    else:
        BLOCK_D = 64
        num_warps = 2
    assert out_dim % BLOCK_D == 0

    grid = (out_dim // BLOCK_D, min(bs, 1024))
    _gather_tokens_kernel[grid](
        gemm_output, flat_weights, flat_index,
        output,
        bs, topk,
        gemm_output.stride(0), output.stride(0),
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=2,
    )

    return output
