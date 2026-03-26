"""Test FP8 fused scatter correctness."""
import torch
import importlib.util
import sys

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sol = load_module("solution/kernel.py", "sol")
ref = load_module("input/reference.py", "ref")

configs = [
    (128, 2, 64, 8, 4096),
    (64, 8, 128, 16, 3072),
    (128, 6, 160, 20, 5120),
    (1024, 8, 160, 20, 5120),
]

ALIGNMENT = 128

for bs, topk, num_experts, num_groups, hidden_dim in configs:
    start_expert = (num_experts - num_groups) // 2
    torch.manual_seed(42)
    hidden_states = torch.randn(bs, hidden_dim, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.randint(0, num_experts, (bs, topk), device="cuda", dtype=torch.int32)
    max_total_M = bs * topk + num_groups * (ALIGNMENT - 1)

    # Run solution
    sh_sol, pl_sol, oi_sol, sc_sol = sol.moe_align_and_scatter(
        hidden_states, topk_ids, num_groups, start_expert, max_total_M
    )

    # Run reference
    sh_ref, pl_ref, oi_ref, sc_ref = ref.ref_moe_align_and_scatter(
        hidden_states, topk_ids, num_groups, start_expert, max_total_M
    )

    # Check types
    assert sh_sol.dtype == torch.float8_e4m3fn, f"sorted_hidden dtype: {sh_sol.dtype}"
    assert sc_sol.dtype == torch.float32, f"sorted_scales dtype: {sc_sol.dtype}"
    assert sh_sol.shape == (max_total_M, hidden_dim), f"sorted_hidden shape: {sh_sol.shape}"
    assert sc_sol.shape == (max_total_M,), f"sorted_scales shape: {sc_sol.shape}"

    # Check packed_layout matches
    assert torch.equal(pl_sol, pl_ref), f"packed_layout mismatch at bs={bs}"

    # Check per-group counts match
    for g in range(num_groups):
        offset = pl_ref[g].item()
        count = pl_ref[num_groups + g].item()
        if count == 0:
            continue
        # Check that the same number of rows are filled
        sol_scales = sc_sol[offset:offset+count]
        ref_scales = sc_ref[offset:offset+count]
        sol_nonzero = (sol_scales > 0).sum().item()
        ref_nonzero = (ref_scales > 0).sum().item()
        assert sol_nonzero == ref_nonzero, \
            f"Group {g}: sol has {sol_nonzero} rows, ref has {ref_nonzero}"

    # Check each output_index entry: dequant should be close to source
    flat_ids = topk_ids.reshape(-1)
    max_rel_err = 0.0
    n_checked = 0
    for i in range(bs * topk):
        eid = flat_ids[i].item()
        lid = eid - start_expert
        sol_idx = oi_sol[i].item()
        if 0 <= lid < num_groups:
            assert sol_idx >= 0, f"output_index[{i}] should be >= 0, got {sol_idx}"
            token_idx = i // topk
            src_row = hidden_states[token_idx].float()
            dst_fp8 = sh_sol[sol_idx].float()
            dst_scale = sc_sol[sol_idx].item()
            dst_dequant = dst_fp8 * dst_scale
            rel = (dst_dequant - src_row).abs().max().item() / (src_row.abs().max().item() + 1e-8)
            max_rel_err = max(max_rel_err, rel)
            n_checked += 1
            assert rel < 0.1, f"output_index[{i}] dequant error too large: {rel:.4f}"
        else:
            assert sol_idx == -1, f"output_index[{i}] should be -1, got {sol_idx}"

    print(f"  OK  bs={bs}, topk={topk}, groups={num_groups}, hidden={hidden_dim} "
          f"| checked={n_checked}, max_rel_err={max_rel_err:.4f}")

print("\nAll FP8 scatter tests PASSED!")
