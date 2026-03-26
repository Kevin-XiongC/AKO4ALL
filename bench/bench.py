#!/usr/bin/env python3
"""
Custom benchmark evaluator for MoE scatter/gather kernels.

Tests correctness against the PyTorch reference, then measures GPU-side
performance using triton.testing.do_bench (CUDA events).

Usage:
    python bench/bench.py --solution solution/kernel.py --ref input/reference.py [options]

Output (structured, one per line):
    COMPILED: True/False
    CORRECT: True/False
    SCATTER_RUNTIME: <ms>       (kernel-only, pre-allocated buffers)
    GATHER_RUNTIME: <ms>
    TOTAL_RUNTIME: <ms>         (scatter + gather)
    SCATTER_FULL_RUNTIME: <ms>  (including allocation)
"""

import argparse
import importlib.util
import math
import os
import sys
import traceback

import torch
import triton


# ---------------------------------------------------------------------------
# Config: DeepSeek-V3 style — EP=8, 160 experts, hidden=5120, topk=8
# ---------------------------------------------------------------------------
NUM_EXPERTS = 160
EP_SIZE = 8
LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE  # 20
HIDDEN_SIZE = 5120
TOPK = 8
START_EXPERT = 0
PRIMARY_BS = 1024

CORRECTNESS_CONFIGS = [
    # (bs, topk, num_experts, num_groups, hidden_dim)
    (128,  2,  64,  8, 4096),
    (512,  4, 256, 32, 7168),
    (1024, 2,  64,  8, 4096),
    (64,   8, 128, 16, 3072),
    (256,  2, 256, 32, 4096),
    (128,  6, 160, 20, 5120),
]

BENCH_BS_VALS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sort_rows(rows, proj):
    keys = rows.float() @ proj
    _, order = keys.sort(stable=True)
    return rows[order]


# ---------------------------------------------------------------------------
# Correctness checks
# ---------------------------------------------------------------------------

def check_scatter_correctness(sol_mod, ref_mod, verbose=False):
    """Returns True if all scatter tests pass."""
    ALIGNMENT = getattr(sol_mod, "ALIGNMENT", 128)

    for bs, topk, num_experts, num_groups, hidden_dim in CORRECTNESS_CONFIGS:
        start_expert = (num_experts - num_groups) // 2
        torch.manual_seed(42)
        hidden_states = torch.randn(bs, hidden_dim, device="cuda", dtype=torch.bfloat16)
        topk_ids = torch.randint(0, num_experts, (bs, topk), device="cuda", dtype=torch.int32)
        max_total_M = bs * topk + num_groups * (ALIGNMENT - 1)

        s_tri, p_tri, i_tri = sol_mod.moe_align_and_scatter(
            hidden_states, topk_ids, num_groups, start_expert, max_total_M,
        )
        s_ref, p_ref, i_ref = ref_mod.ref_moe_align_and_scatter(
            hidden_states, topk_ids, num_groups, start_expert, max_total_M,
        )

        if not torch.equal(p_tri, p_ref):
            if verbose:
                print(f"  FAIL scatter packed_layout bs={bs}: "
                      f"got {p_tri.tolist()}, expected {p_ref.tolist()}")
            return False

        gen = torch.Generator(device="cuda").manual_seed(12345)
        proj = torch.randn(hidden_dim, device="cuda", dtype=torch.float32, generator=gen)
        for g in range(num_groups):
            offset = p_ref[g].item()
            count = p_ref[num_groups + g].item()
            if count == 0:
                continue
            tri_sorted = _sort_rows(s_tri[offset:offset + count], proj)
            ref_sorted = _sort_rows(s_ref[offset:offset + count], proj)
            if not torch.equal(tri_sorted, ref_sorted):
                if verbose:
                    print(f"  FAIL scatter group {g} data mismatch bs={bs}")
                return False

        flat_ids = topk_ids.reshape(-1)
        for i in range(bs * topk):
            eid = flat_ids[i].item()
            lid = eid - start_expert
            row = i_tri[i].item()
            if 0 <= lid < num_groups:
                if row < 0:
                    if verbose:
                        print(f"  FAIL scatter output_index[{i}] should be >= 0")
                    return False
                tok = i // topk
                if not torch.equal(s_tri[row], hidden_states[tok]):
                    if verbose:
                        print(f"  FAIL scatter output_index[{i}] points to wrong data")
                    return False
            else:
                if row != -1:
                    if verbose:
                        print(f"  FAIL scatter output_index[{i}] should be -1")
                    return False

        if verbose:
            print(f"  scatter OK  bs={bs}, topk={topk}, experts={num_experts}, "
                  f"groups={num_groups}, hidden={hidden_dim}")

    return True


def check_gather_correctness(sol_mod, ref_mod, verbose=False):
    """Returns True if all gather tests pass."""
    ALIGNMENT = getattr(sol_mod, "ALIGNMENT", 128)

    for bs, topk, num_experts, num_groups, hidden_dim in CORRECTNESS_CONFIGS:
        start_expert = (num_experts - num_groups) // 2
        out_dim = hidden_dim

        torch.manual_seed(42)
        hidden_states = torch.randn(bs, hidden_dim, device="cuda", dtype=torch.bfloat16)
        topk_ids = torch.randint(0, num_experts, (bs, topk), device="cuda", dtype=torch.int32)
        topk_weights = torch.softmax(
            torch.randn(bs, topk, device="cuda", dtype=torch.float32), dim=-1,
        )
        max_total_M = bs * topk + num_groups * (ALIGNMENT - 1)

        _, _, output_index = sol_mod.moe_align_and_scatter(
            hidden_states, topk_ids, num_groups, start_expert, max_total_M,
        )
        torch.manual_seed(123)
        gemm_output = torch.randn(max_total_M, out_dim, device="cuda", dtype=torch.bfloat16)

        out_tri = sol_mod.moe_gather(gemm_output, topk_weights, output_index)
        out_ref = ref_mod.ref_moe_gather(
            gemm_output, topk_ids, topk_weights, output_index,
            start_expert, num_groups,
        )

        rel_err = (out_tri.float() - out_ref.float()).abs().max().item() / (
            out_ref.float().abs().max().item() + 1e-8
        )
        if rel_err > 1e-2:
            if verbose:
                print(f"  FAIL gather bs={bs} rel_err={rel_err:.6f}")
            return False

        if verbose:
            print(f"  gather OK  bs={bs}, topk={topk}, experts={num_experts}, "
                  f"groups={num_groups}, out_dim={out_dim}, rel_err={rel_err:.2e}")

    return True


def check_roundtrip(sol_mod, verbose=False):
    """Scatter → identity GEMM → gather roundtrip."""
    ALIGNMENT = getattr(sol_mod, "ALIGNMENT", 128)
    bs, topk, num_experts, num_groups, hidden_dim = 1024, 8, 160, 20, 5120
    start_expert = (num_experts - num_groups) // 2

    torch.manual_seed(42)
    hidden_states = torch.randn(bs, hidden_dim, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.randint(0, num_experts, (bs, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(
        torch.randn(bs, topk, device="cuda", dtype=torch.float32), dim=-1,
    )
    max_total_M = bs * topk + num_groups * (ALIGNMENT - 1)

    sorted_hidden, _, output_index = sol_mod.moe_align_and_scatter(
        hidden_states, topk_ids, num_groups, start_expert, max_total_M,
    )
    out_tri = sol_mod.moe_gather(sorted_hidden, topk_weights, output_index)

    out_ref = torch.zeros_like(out_tri)
    for i in range(bs):
        for k in range(topk):
            eid = topk_ids[i, k].item()
            lid = eid - start_expert
            if 0 <= lid < num_groups:
                out_ref[i] += topk_weights[i, k].item() * hidden_states[i].float()

    rel_err = (out_tri.float() - out_ref.float()).abs().max().item() / (
        out_ref.float().abs().max().item() + 1e-8
    )
    ok = rel_err < 1e-2
    if verbose:
        status = "OK" if ok else "FAIL"
        print(f"  roundtrip {status}  rel_err={rel_err:.2e}")
    return ok


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------

def bench_scatter_kernel_only(sol_mod, bs, topk, local_experts, hidden_size, start_expert):
    """Pre-allocated buffers, measure only the Triton kernels."""
    ALIGNMENT = getattr(sol_mod, "ALIGNMENT", 128)
    hidden_states = torch.randn(bs, hidden_size, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.randint(0, NUM_EXPERTS, (bs, topk), device="cuda", dtype=torch.int32)
    max_total_M = bs * topk + local_experts * (ALIGNMENT - 1)
    num_elements = bs * topk
    flat_topk_ids = topk_ids.reshape(-1).contiguous()

    packed_layout = torch.empty(2 * local_experts, dtype=torch.int32, device="cuda")
    sorted_hidden = torch.empty(max_total_M, hidden_size, device="cuda", dtype=torch.bfloat16)
    write_counters = torch.zeros(local_experts, dtype=torch.int32, device="cuda")
    output_index = torch.full((num_elements,), -1, dtype=torch.int32, device="cuda")

    BLOCK_SIZE = 1024
    BLOCK_G = triton.next_power_of_2(local_experts)
    NUM_ITERS = math.ceil(num_elements / BLOCK_SIZE)
    HIDDEN_SIZE_PAD = triton.next_power_of_2(hidden_size)
    grid_size = min(bs, 1024 * 8)

    # JIT warmup
    sol_mod._count_and_compute_layout_kernel[(1,)](
        flat_topk_ids, packed_layout, num_elements, start_expert, local_experts,
        BLOCK_G=BLOCK_G, ALIGNMENT=ALIGNMENT, BLOCK_SIZE=BLOCK_SIZE, NUM_ITERS=NUM_ITERS,
    )
    sol_mod._scatter_tokens_kernel[(grid_size,)](
        hidden_states, sorted_hidden, flat_topk_ids,
        packed_layout, write_counters, output_index,
        bs, topk, start_expert, local_experts,
        hidden_states.stride(0), sorted_hidden.stride(0),
        HIDDEN_SIZE=hidden_size, HIDDEN_SIZE_PAD=HIDDEN_SIZE_PAD, num_warps=8,
    )
    torch.cuda.synchronize()

    def _run():
        write_counters.zero_()
        sol_mod._count_and_compute_layout_kernel[(1,)](
            flat_topk_ids, packed_layout, num_elements, start_expert, local_experts,
            BLOCK_G=BLOCK_G, ALIGNMENT=ALIGNMENT, BLOCK_SIZE=BLOCK_SIZE, NUM_ITERS=NUM_ITERS,
        )
        sol_mod._scatter_tokens_kernel[(grid_size,)](
            hidden_states, sorted_hidden, flat_topk_ids,
            packed_layout, write_counters, output_index,
            bs, topk, start_expert, local_experts,
            hidden_states.stride(0), sorted_hidden.stride(0),
            HIDDEN_SIZE=hidden_size, HIDDEN_SIZE_PAD=HIDDEN_SIZE_PAD, num_warps=8,
        )
    return _run


def bench_scatter_full(sol_mod, bs, topk, local_experts, hidden_size, start_expert):
    ALIGNMENT = getattr(sol_mod, "ALIGNMENT", 128)
    hidden_states = torch.randn(bs, hidden_size, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.randint(0, NUM_EXPERTS, (bs, topk), device="cuda", dtype=torch.int32)
    max_total_M = bs * topk + local_experts * (ALIGNMENT - 1)

    def _run():
        sol_mod.moe_align_and_scatter(hidden_states, topk_ids, local_experts, start_expert, max_total_M)
    return _run


def bench_gather(sol_mod, bs, topk, local_experts, hidden_size, start_expert):
    ALIGNMENT = getattr(sol_mod, "ALIGNMENT", 128)
    hidden_states = torch.randn(bs, hidden_size, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.randint(0, NUM_EXPERTS, (bs, topk), device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(
        torch.randn(bs, topk, device="cuda", dtype=torch.float32), dim=-1,
    )
    max_total_M = bs * topk + local_experts * (ALIGNMENT - 1)
    _, _, output_index = sol_mod.moe_align_and_scatter(
        hidden_states, topk_ids, local_experts, start_expert, max_total_M,
    )
    gemm_output = torch.randn(max_total_M, hidden_size, device="cuda", dtype=torch.bfloat16)

    def _run():
        sol_mod.moe_gather(gemm_output, topk_weights, output_index)
    return _run


def run_benchmark(sol_mod, verbose=False):
    if verbose:
        print(f"\n--- Perf sweep (E={NUM_EXPERTS}, EP={EP_SIZE}, "
              f"local={LOCAL_EXPERTS}, hidden={HIDDEN_SIZE}, topk={TOPK}) ---\n")
        header = (f"{'bs':>6}  {'Scatter(full)':>14} {'Scatter(kern)':>14} "
                  f"{'Gather':>10} {'Kern+Gather':>12}")
        print(header)
        print("-" * len(header))

    primary_scatter = None
    primary_gather = None
    primary_total = None

    for bs in BENCH_BS_VALS:
        fn_full = bench_scatter_full(sol_mod, bs, TOPK, LOCAL_EXPERTS, HIDDEN_SIZE, START_EXPERT)
        fn_kern = bench_scatter_kernel_only(sol_mod, bs, TOPK, LOCAL_EXPERTS, HIDDEN_SIZE, START_EXPERT)
        fn_gath = bench_gather(sol_mod, bs, TOPK, LOCAL_EXPERTS, HIDDEN_SIZE, START_EXPERT)

        ms_full = triton.testing.do_bench(fn_full, warmup=50, rep=200)
        ms_kern = triton.testing.do_bench(fn_kern, warmup=50, rep=200)
        ms_gath = triton.testing.do_bench(fn_gath, warmup=50, rep=200)

        if verbose:
            print(f"{bs:>6}  {ms_full:>13.4f}ms {ms_kern:>13.4f}ms "
                  f"{ms_gath:>9.4f}ms {ms_kern+ms_gath:>11.4f}ms")

        if bs == PRIMARY_BS:
            primary_scatter = ms_kern
            primary_gather = ms_gath
            primary_total = ms_kern + ms_gath

    return primary_scatter, primary_gather, primary_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MoE scatter/gather benchmark")
    parser.add_argument("--solution", required=True, help="Path to solution kernel.py")
    parser.add_argument("--ref", required=True, help="Path to reference.py")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-perf", action="store_true", help="Only check correctness")
    args = parser.parse_args()

    # Load modules
    compiled = True
    try:
        sol_mod = _load_module(os.path.abspath(args.solution), "solution_kernel")
        ref_mod = _load_module(os.path.abspath(args.ref), "reference")
    except Exception as e:
        compiled = False
        print(f"COMPILED: False")
        print(f"CORRECT: False")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

    print(f"COMPILED: True")

    # Correctness
    try:
        scatter_ok = check_scatter_correctness(sol_mod, ref_mod, verbose=args.verbose)
        gather_ok = check_gather_correctness(sol_mod, ref_mod, verbose=args.verbose)
        roundtrip_ok = check_roundtrip(sol_mod, verbose=args.verbose)
        correct = scatter_ok and gather_ok and roundtrip_ok
    except Exception as e:
        correct = False
        if args.verbose:
            traceback.print_exc()

    print(f"CORRECT: {correct}")

    if not correct:
        sys.exit(1)

    if args.skip_perf:
        sys.exit(0)

    # Performance
    try:
        scatter_ms, gather_ms, total_ms = run_benchmark(sol_mod, verbose=args.verbose)
        print(f"SCATTER_RUNTIME: {scatter_ms:.4f}")
        print(f"GATHER_RUNTIME: {gather_ms:.4f}")
        print(f"TOTAL_RUNTIME: {total_ms:.4f}")
    except Exception as e:
        print(f"PERF_ERROR: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
