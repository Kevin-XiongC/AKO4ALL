#!/usr/bin/env python3
"""
Custom benchmark for fused QK-Norm + RoPE + FP8-Cast + KV-Store kernel.

Target shape: GLM4 TP8
  head_dim=128, q_heads=12, kv_heads=1
  partial_rotary_factor=0.5 (rotary_dim=64), NeoX style

All outputs are FP8 E4M3: q_output, k_cache, v_cache.

Loss = negative sum of speedup ratios across all batch sizes.

Usage:
    python bench/bench.py --solution solution/kernel.py --ref input/reference.py [--verbose]

Output:
    COMPILED: True/False
    CORRECT: True/False
    SUM_SPEEDUP: <sum of speedups across all batch sizes>
    SPEEDUP: <mean speedup across all batch sizes>
"""

import argparse
import importlib.util
import os
import statistics
import sys
import traceback

import torch

# ---------------------------------------------------------------------------
# Target shape config
# ---------------------------------------------------------------------------
HEAD_DIM = 128
NUM_HEADS_Q = 12
NUM_HEADS_K = 1
NUM_HEADS_V = 1
EPS = 1e-5
BASE = 10000.0
IS_NEOX = True
PARTIAL_ROTARY_FACTOR = 0.5
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)  # 64
FACTOR = 1.0
LOW = 0.0
HIGH = 0.0
ATTENTION_FACTOR = 1.0
Q_SCALE = 1.0
K_SCALE = 1.0
V_SCALE = 1.0

CACHE_SIZE = 32768

NUM_CORRECT_TRIALS = 5
NUM_PERF_TRIALS = 100
NUM_WARMUP = 10

CORRECTNESS_TOKEN_CONFIGS = [1, 128, 1024, 4096, 16384]
PERF_SWEEP_TOKENS = list(range(128, 16384 + 1, 512))


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------
def _load_module(path, name):
    """Load a Python module from file path using importlib."""
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------
def make_inputs(num_tokens, device="cuda", seed=42):
    """Generate fresh inputs for a single trial."""
    torch.manual_seed(seed)
    hidden = (NUM_HEADS_Q + NUM_HEADS_K + NUM_HEADS_V) * HEAD_DIM
    q_dim = NUM_HEADS_Q * HEAD_DIM
    kv_dim = NUM_HEADS_K * HEAD_DIM

    qkv = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    position_ids = (torch.arange(num_tokens, device=device) + 100).to(torch.int64)
    q_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device) * 2.0
    k_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device) * 2.0
    q_output = torch.zeros(num_tokens, q_dim, dtype=torch.uint8, device=device)
    k_cache = torch.zeros(CACHE_SIZE, kv_dim, dtype=torch.uint8, device=device)
    v_cache = torch.zeros(CACHE_SIZE, kv_dim, dtype=torch.uint8, device=device)
    out_loc = torch.randperm(CACHE_SIZE, device=device)[:num_tokens].to(torch.int32)

    return qkv, position_ids, q_weight, k_weight, q_output, k_cache, v_cache, out_loc


def run_kernel(mod, qkv, position_ids, q_weight, k_weight,
               q_output, k_cache, v_cache, out_loc):
    """Call the kernel module's fused_qk_norm_rope_store function."""
    mod.fused_qk_norm_rope_store(
        qkv, NUM_HEADS_Q, NUM_HEADS_K, NUM_HEADS_V, HEAD_DIM,
        EPS, q_weight, k_weight, BASE, IS_NEOX, position_ids,
        FACTOR, LOW, HIGH, ATTENTION_FACTOR, ROTARY_DIM,
        q_output, Q_SCALE,
        k_cache, v_cache, out_loc, K_SCALE, V_SCALE,
    )


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
# FP8 E4M3: 3-bit mantissa → 1 ULP relative tolerance = 2^-3 = 0.125
# Subnormal minimum step = 2^-9 ≈ 0.002
FP8_ATOL = 0.002
FP8_RTOL = 0.125


def _check_fp8_close(ref_u8, sol_u8):
    """Compare FP8 E4M3 values in float domain with 1-ULP tolerance (vLLM style).

    Converting uint8 → float8_e4m3fn → float32 means ±0 both become 0.0,
    so signed-zero differences are naturally handled.
    """
    ref_f = ref_u8.view(torch.float8_e4m3fn).float()
    sol_f = sol_u8.view(torch.float8_e4m3fn).float()
    diff = (ref_f - sol_f).abs()
    tol = FP8_ATOL + FP8_RTOL * torch.max(ref_f.abs(), sol_f.abs())
    ok = (diff <= tol).all().item()
    max_diff = diff.max().item()
    return ok, max_diff


def check_correctness(ref_mod, sol_mod, verbose=False):
    """
    Check correctness across multiple token counts and trials.
    FP8 outputs compared as float values with atol/rtol (matching vLLM style).
    """
    all_pass = True

    for num_tokens in CORRECTNESS_TOKEN_CONFIGS:
        for trial in range(NUM_CORRECT_TRIALS):
            seed = 42 + trial * 1000 + num_tokens

            (qkv_ref, pos, qw, kw, qo_ref, kc_ref, vc_ref, out_loc
             ) = make_inputs(num_tokens, seed=seed)
            qkv_sol = qkv_ref.clone()
            qo_sol = qo_ref.clone()
            kc_sol = kc_ref.clone()
            vc_sol = vc_ref.clone()

            run_kernel(ref_mod, qkv_ref, pos, qw, kw,
                       qo_ref, kc_ref, vc_ref, out_loc)
            run_kernel(sol_mod, qkv_sol, pos, qw, kw,
                       qo_sol, kc_sol, vc_sol, out_loc)

            q_ok, q_diff = _check_fp8_close(qo_ref, qo_sol)

            k_ok = True
            v_ok = True
            k_max_diff = 0.0
            v_max_diff = 0.0
            for i in range(num_tokens):
                slot = out_loc[i].item()
                ki_ok, kd = _check_fp8_close(kc_ref[slot], kc_sol[slot])
                vi_ok, vd = _check_fp8_close(vc_ref[slot], vc_sol[slot])
                k_max_diff = max(k_max_diff, kd)
                v_max_diff = max(v_max_diff, vd)
                if not ki_ok:
                    k_ok = False
                if not vi_ok:
                    v_ok = False

            ok = q_ok and k_ok and v_ok
            if not ok:
                all_pass = False

            if verbose:
                status = "PASS" if ok else "FAIL"
                print(
                    f"  [{status}] tokens={num_tokens:4d} trial={trial} "
                    f"Q={'ok' if q_ok else 'FAIL'}(maxdiff={q_diff:.4e}) "
                    f"K={'ok' if k_ok else 'FAIL'}(maxdiff={k_max_diff:.4e}) "
                    f"V={'ok' if v_ok else 'FAIL'}(maxdiff={v_max_diff:.4e})"
                )

    return all_pass


# ---------------------------------------------------------------------------
# L2 cache clearing (from KernelBench)
# ---------------------------------------------------------------------------
def clear_l2_cache(device="cuda"):
    """Thrash L2 with ~256 MB allocation."""
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    dummy.fill_(42)
    del dummy


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def bench_kernel(mod, num_tokens, num_trials=NUM_PERF_TRIALS,
                 warmup=NUM_WARMUP, device="cuda"):
    """
    Time the kernel using CUDA events with L2 cache clearing.
    Returns list of elapsed times in milliseconds.
    """
    q_dim = NUM_HEADS_Q * HEAD_DIM
    kv_dim = NUM_HEADS_K * HEAD_DIM

    # Pre-create base inputs (cloned per trial for in-place safety)
    qkv_base, pos, qw, kw, _, _, _, out_loc = make_inputs(num_tokens, seed=42)

    # Warmup
    for _ in range(warmup):
        qkv = qkv_base.clone()
        qo = torch.zeros(num_tokens, q_dim, dtype=torch.uint8, device=device)
        kc = torch.zeros(CACHE_SIZE, kv_dim, dtype=torch.uint8, device=device)
        vc = torch.zeros(CACHE_SIZE, kv_dim, dtype=torch.uint8, device=device)
        run_kernel(mod, qkv, pos, qw, kw, qo, kc, vc, out_loc)
    torch.cuda.synchronize()

    # Timed trials
    times = []
    for _ in range(num_trials):
        qkv = qkv_base.clone()
        qo = torch.zeros(num_tokens, q_dim, dtype=torch.uint8, device=device)
        kc = torch.zeros(CACHE_SIZE, kv_dim, dtype=torch.uint8, device=device)
        vc = torch.zeros(CACHE_SIZE, kv_dim, dtype=torch.uint8, device=device)

        clear_l2_cache(device)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        run_kernel(mod, qkv, pos, qw, kw, qo, kc, vc, out_loc)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return times


def get_stats(times):
    """Compute mean/std/min/max from a list of times."""
    return {
        "mean": statistics.mean(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fused QK-Norm+RoPE+FP8+Store kernel"
    )
    parser.add_argument("--solution", required=True, help="Path to solution kernel.py")
    parser.add_argument("--ref", required=True, help="Path to reference.py")
    parser.add_argument("--verbose", action="store_true", help="Detailed output")
    parser.add_argument("--skip-perf", action="store_true", help="Only check correctness")
    parser.add_argument(
        "--num-perf-trials", type=int, default=NUM_PERF_TRIALS,
        help=f"Number of performance trials (default: {NUM_PERF_TRIALS})"
    )
    args = parser.parse_args()

    if args.verbose:
        print(f"Target shape: head_dim={HEAD_DIM}, q={NUM_HEADS_Q}, "
              f"kv={NUM_HEADS_K}, rotary_dim={ROTARY_DIM}, NeoX={IS_NEOX}")
        print(f"Sweep: {len(PERF_SWEEP_TOKENS)} batch sizes from "
              f"{PERF_SWEEP_TOKENS[0]} to {PERF_SWEEP_TOKENS[-1]}")
        print(f"Hardware: {torch.cuda.get_device_name()}")
        print()

    # ---- Load modules ----
    try:
        ref_mod = _load_module(args.ref, "reference")
        sol_mod = _load_module(args.solution, "solution")
    except Exception as e:
        print(f"COMPILED: False")
        print(f"CORRECT: False")
        traceback.print_exc()
        sys.exit(1)

    print("COMPILED: True")

    # ---- Correctness ----
    if args.verbose:
        print("\n--- Correctness checks ---")
    correct = check_correctness(ref_mod, sol_mod, verbose=args.verbose)
    print(f"CORRECT: {correct}")

    if not correct:
        sys.exit(1)

    if args.skip_perf:
        sys.exit(0)

    # ---- Performance sweep across all batch sizes ----
    num_trials = args.num_perf_trials

    if args.verbose:
        print(f"\n--- Batch-size sweep ({num_trials} trials each) ---")
        print(f"  {'tokens':>6s}  {'sol(ms)':>8s}  {'ref(ms)':>8s}  {'speedup':>8s}")

    speedups = []
    for nt in PERF_SWEEP_TOKENS:
        st = bench_kernel(sol_mod, nt, num_trials=num_trials, warmup=5)
        rt = bench_kernel(ref_mod, nt, num_trials=num_trials, warmup=5)
        sm = statistics.mean(st)
        rm = statistics.mean(rt)
        sp = rm / sm if sm > 0 else 0
        speedups.append(sp)

        if args.verbose:
            print(f"  {nt:6d}  {sm:8.4f}  {rm:8.4f}  {sp:7.2f}x")

    sum_speedup = sum(speedups)
    mean_speedup = sum_speedup / len(speedups) if speedups else 0

    print(f"SUM_SPEEDUP: {sum_speedup:.4f}")
    print(f"SPEEDUP: {mean_speedup:.4f}x")

    if args.verbose:
        print(f"\n  Sum of speedups: {sum_speedup:.4f} (across {len(speedups)} batch sizes)")
        print(f"  Mean speedup:    {mean_speedup:.4f}x")
        print(f"  Min speedup:     {min(speedups):.4f}x (at {PERF_SWEEP_TOKENS[speedups.index(min(speedups))]} tokens)")
        print(f"  Max speedup:     {max(speedups):.4f}x (at {PERF_SWEEP_TOKENS[speedups.index(max(speedups))]} tokens)")

    sys.exit(0)


if __name__ == "__main__":
    main()
