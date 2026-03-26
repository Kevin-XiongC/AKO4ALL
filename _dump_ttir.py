import os
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_dump"
os.environ["MLIR_ENABLE_DUMP"] = "1"

import math, torch, triton, triton.language as tl
import importlib.util

def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sol = _load_module("solution/kernel.py", "sol")

bs = 1024
HIDDEN_SIZE = 5120
TOPK = 8
LOCAL_EXPERTS = 20
ALIGNMENT = 128
FP8_MAX = 448.0

torch.manual_seed(42)
hidden_states = torch.randn(bs, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
topk_ids = torch.randint(0, 160, (bs, TOPK), device="cuda", dtype=torch.int32)
max_total_M = bs * TOPK + LOCAL_EXPERTS * (ALIGNMENT - 1)
flat_topk_ids = topk_ids.reshape(-1).contiguous()

packed_layout = torch.empty(2 * LOCAL_EXPERTS, dtype=torch.int32, device="cuda")
sorted_hidden = torch.empty(max_total_M, HIDDEN_SIZE, device="cuda", dtype=torch.float8_e4m3fn)
sorted_scales = torch.zeros(max_total_M, device="cuda", dtype=torch.float32)
write_counters = torch.zeros(LOCAL_EXPERTS, dtype=torch.int32, device="cuda")
output_index = torch.full((bs*TOPK,), -1, dtype=torch.int32, device="cuda")

HIDDEN_SIZE_PAD = triton.next_power_of_2(HIDDEN_SIZE)
grid_size = min(bs, 1024*8)

# This will compile and cache the kernel
sol._scatter_tokens_kernel[(grid_size,)](
    hidden_states, sorted_hidden, flat_topk_ids,
    packed_layout, write_counters, output_index,
    bs, TOPK, 0, LOCAL_EXPERTS,
    hidden_states.stride(0), sorted_hidden.stride(0),
    HIDDEN_SIZE=HIDDEN_SIZE, HIDDEN_SIZE_PAD=HIDDEN_SIZE_PAD,
    sorted_scales_ptr=sorted_scales, FP8_MAX=FP8_MAX,
    num_warps=8,
)
torch.cuda.synchronize()

# Find compiled kernel in cache
import glob
cache_files = glob.glob("/tmp/triton_dump/**/*.ttgir", recursive=True)
for f in sorted(cache_files):
    if 'scatter' in f.lower() or os.path.getsize(f) > 5000:
        print(f"=== {f} ({os.path.getsize(f)} bytes) ===")
        # Just print size, don't print content
        
# Check cubin for register info
cubin_files = glob.glob("/tmp/triton_dump/**/*.cubin", recursive=True)
print(f"\nFound {len(cubin_files)} cubin files")

# Use triton's asm property
kernel = sol._scatter_tokens_kernel
print(f"\nKernel type: {type(kernel)}")
