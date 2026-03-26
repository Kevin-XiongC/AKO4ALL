"""
PyTorch reference implementations for MoE scatter/gather.

Slow, loop-based implementations used as correctness golden for the
Triton kernels in kernel.py.
"""

import torch

ALIGNMENT = 128


def ref_moe_align_and_scatter(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    num_groups: int,
    start_expert: int,
    max_total_M: int | None = None,
    alignment: int = ALIGNMENT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reference scatter: loop over all (token, topk) pairs, write to sorted buffer.

    Returns:
        sorted_hidden: [max_total_M, hidden_dim]
        packed_layout: [2 * num_groups]  — [m_offsets | m_counts]
        output_index:  [bs * topk]       — reverse mapping (-1 for non-local)
    """
    bs, hidden_dim = hidden_states.shape
    topk = topk_ids.shape[1]
    device = hidden_states.device

    flat_ids = topk_ids.reshape(-1)

    m_counts = torch.zeros(num_groups, dtype=torch.int32, device=device)
    for g in range(num_groups):
        m_counts[g] = ((flat_ids - start_expert) == g).sum().int()

    aligned_counts = ((m_counts + alignment - 1) // alignment * alignment).int()
    m_offsets = torch.zeros(num_groups, dtype=torch.int32, device=device)
    if num_groups > 1:
        m_offsets[1:] = torch.cumsum(aligned_counts[:-1], dim=0)

    packed_layout = torch.cat([m_offsets, m_counts])

    if max_total_M is None:
        max_total_M = bs * topk + num_groups * (alignment - 1)

    sorted_hidden = torch.zeros(
        max_total_M, hidden_dim, device=device, dtype=hidden_states.dtype,
    )
    output_index = torch.full((bs * topk,), -1, dtype=torch.int32, device=device)

    write_pos = torch.zeros(num_groups, dtype=torch.int32, device=device)
    for idx in range(bs * topk):
        token_idx = idx // topk
        expert_id = flat_ids[idx].item()
        local_id = expert_id - start_expert
        if 0 <= local_id < num_groups:
            offset = m_offsets[local_id].item()
            pos = write_pos[local_id].item()
            sorted_hidden[offset + pos] = hidden_states[token_idx]
            output_index[idx] = offset + pos
            write_pos[local_id] += 1

    return sorted_hidden, packed_layout, output_index


def ref_moe_gather(
    gemm_output: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output_index: torch.Tensor,
    start_expert: int,
    num_groups: int,
) -> torch.Tensor:
    """
    Reference gather: loop over all (token, topk) pairs, accumulate weighted rows.

    Returns:
        output: [bs, out_dim]
    """
    bs, topk = topk_ids.shape
    out_dim = gemm_output.shape[1]
    device = gemm_output.device

    output = torch.zeros(bs, out_dim, device=device, dtype=gemm_output.dtype)
    flat_ids = topk_ids.reshape(-1)
    flat_weights = topk_weights.reshape(-1)
    flat_index = output_index.reshape(-1)

    for i in range(bs * topk):
        expert_id = flat_ids[i].item()
        local_id = expert_id - start_expert
        if 0 <= local_id < num_groups:
            src_row = flat_index[i].item()
            token_idx = i // topk
            w = flat_weights[i].item()
            output[token_idx] += w * gemm_output[src_row].float()

    return output
