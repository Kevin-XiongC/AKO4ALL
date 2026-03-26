#!/usr/bin/env python3
"""
Custom benchmark for trtllm_allreduce_fusion (multi-GPU NCCL kernel).

Deploys modified CUDA sources, runs correctness + bandwidth test on 8 GPUs,
and outputs results in KernelBench-compatible format.

Usage:
  python bench/bench.py [--solution X --ref Y --verbose]
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_FILE = os.path.join(ROOT, "bench", "baseline.json")


def run_cmd(cmd, cwd=None, timeout=300):
    """Run a command, return (returncode, stdout)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=cwd or ROOT, timeout=timeout,
    )
    return result.returncode, result.stdout + result.stderr


def deploy():
    """Deploy modified CUDA sources to flashinfer."""
    rc, out = run_cmd("bash scripts/deploy.sh")
    if rc != 0:
        print(f"Deploy failed:\n{out}", file=sys.stderr)
        return False
    return True


def run_test():
    """Run the multi-GPU allreduce fusion test. Returns (success, output)."""
    env = os.environ.copy()
    # Ensure no debug build
    env.pop("FLASHINFER_JIT_VERBOSE", None)
    env["FLASHINFER_JIT_DEBUG"] = "0"
    result = subprocess.run(
        [sys.executable, "tests/test_allreduce_fusion.py"],
        capture_output=True, text=True, cwd=ROOT,
        env=env, timeout=600,
    )
    output = result.stdout + result.stderr
    return result.returncode == 0, output


def parse_results(output):
    """Parse test output for correctness and bandwidth."""
    # Check correctness
    correct = "ALL PASS" in output

    # Parse bandwidth lines: "  token_num=  128  avg=0.0085 ms  total=..."
    latencies = {}
    for line in output.split("\n"):
        m = re.match(r"\s*token_num=\s*(\d+)\s+avg=([\d.]+)\s*ms", line)
        if m:
            token_num = int(m.group(1))
            avg_ms = float(m.group(2))
            latencies[token_num] = avg_ms

    return correct, latencies


def geomean(values):
    """Geometric mean of a list of values."""
    if not values:
        return 0.0
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / len(values))


def save_baseline(latencies):
    """Save baseline latencies."""
    with open(BASELINE_FILE, "w") as f:
        json.dump({str(k): v for k, v in latencies.items()}, f, indent=2)


def load_baseline():
    """Load baseline latencies. Returns dict or None."""
    if not os.path.exists(BASELINE_FILE):
        return None
    with open(BASELINE_FILE) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution/kernel.py")
    parser.add_argument("--ref", default="input/reference.py")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save current results as baseline")
    args = parser.parse_args()

    # Step 1: Deploy
    print("--- Deploying CUDA sources ---")
    if not deploy():
        print("COMPILED: False")
        print("CORRECT: False")
        sys.exit(1)
    print("COMPILED: True")

    # Step 2: Run test
    print("\n--- Running multi-GPU test (8x H200) ---")
    success, output = run_test()

    if args.verbose:
        print(output)

    # Step 3: Parse results
    correct, latencies = parse_results(output)
    print(f"\nCORRECT: {correct}")

    if not latencies:
        print("ERROR: No bandwidth results parsed")
        if not args.verbose:
            print(output)
        sys.exit(1)

    # Print per-token results
    print("\n--- Per-token latencies ---")
    for tok in sorted(latencies.keys()):
        print(f"  token_num={tok:5d}  avg={latencies[tok]:.4f} ms")

    # Compute summary metric (geometric mean of all token latencies)
    runtime = geomean(list(latencies.values()))
    print(f"\nRUNTIME: {runtime:.4f}")

    # Handle baseline
    if args.save_baseline:
        save_baseline(latencies)
        print(f"REF_RUNTIME: {runtime:.4f}")
        print(f"SPEEDUP: 1.0000x")
    else:
        baseline = load_baseline()
        if baseline:
            ref_runtime = geomean([baseline[k] for k in sorted(baseline.keys())])
            speedup = ref_runtime / runtime if runtime > 0 else 0
            print(f"REF_RUNTIME: {ref_runtime:.4f}")
            print(f"SPEEDUP: {speedup:.4f}x")

            # Per-token speedups
            print("\n--- Per-token speedups ---")
            for tok in sorted(latencies.keys()):
                if tok in baseline:
                    sp = baseline[tok] / latencies[tok] if latencies[tok] > 0 else 0
                    print(f"  token_num={tok:5d}  {baseline[tok]:.4f} -> {latencies[tok]:.4f} ms  ({sp:.2f}x)")
        else:
            print(f"REF_RUNTIME: {runtime:.4f}")
            print("SPEEDUP: 1.0000x")
            print("(No baseline found. Run with --save-baseline first)")

    sys.exit(0 if correct else 1)


if __name__ == "__main__":
    main()
