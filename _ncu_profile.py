import math, torch, triton
import importlib.util

def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sol = _load_module("solution/kernel.py", "sol")

NUM_EXPERTS = 160
EP_SIZE = 8
LOCAL_EXPERTS = 20
HIDDEN_SIZE = 5120
TOPK = 8
START_EXPERT = 0
bs = 1024
ALIGNMENT = 128
FP8_MAX = 448.0

torch.manual_seed(42)
hidden_states = torch.randn(bs, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
topk_ids = torch.randint(0, NUM_EXPERTS, (bs, TOPK), device="cuda", dtype=torch.int32)
topk_weights = torch.softmax(torch.randn(bs, TOPK, device="cuda", dtype=torch.float32), dim=-1)
max_total_M = bs * TOPK + LOCAL_EXPERTS * (ALIGNMENT - 1)
num_elements = bs * TOPK
flat_topk_ids = topk_ids.reshape(-1).contiguous()

packed_layout = torch.empty(2 * LOCAL_EXPERTS, dtype=torch.int32, device="cuda")
sorted_hidden = torch.empty(max_total_M, HIDDEN_SIZE, device="cuda", dtype=torch.float8_e4m3fn)
sorted_scales = torch.zeros(max_total_M, device="cuda", dtype=torch.float32)
write_counters = torch.zeros(LOCAL_EXPERTS, dtype=torch.int32, device="cuda")
output_index = torch.full((num_elements,), -1, dtype=torch.int32, device="cuda")
gemm_output = torch.randn(max_total_M, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

BLOCK_SIZE = 1024
BLOCK_G = triton.next_power_of_2(LOCAL_EXPERTS)
NUM_ITERS = math.ceil(num_elements / BLOCK_SIZE)
HIDDEN_SIZE_PAD = triton.next_power_of_2(HIDDEN_SIZE)
grid_size = min(bs, 1024 * 8)

# Warmup
sol._count_and_compute_layout_kernel[(1,)](
    flat_topk_ids, packed_layout, num_elements, START_EXPERT, LOCAL_EXPERTS,
    BLOCK_G=BLOCK_G, ALIGNMENT=ALIGNMENT, BLOCK_SIZE=BLOCK_SIZE, NUM_ITERS=NUM_ITERS,
)
sol._scatter_tokens_kernel[(grid_size,)](
    hidden_states, sorted_hidden, flat_topk_ids,
    packed_layout, write_counters, output_index,
    bs, TOPK, START_EXPERT, LOCAL_EXPERTS,
    hidden_states.stride(0), sorted_hidden.stride(0),
    HIDDEN_SIZE=HIDDEN_SIZE, HIDDEN_SIZE_PAD=HIDDEN_SIZE_PAD,
    sorted_scales_ptr=sorted_scales, FP8_MAX=FP8_MAX,
    num_warps=8,
)
sol.moe_gather(gemm_output, topk_weights, output_index)
torch.cuda.synchronize()

# Profiled run
write_counters.zero_()
sol._count_and_compute_layout_kernel[(1,)](
    flat_topk_ids, packed_layout, num_elements, START_EXPERT, LOCAL_EXPERTS,
    BLOCK_G=BLOCK_G, ALIGNMENT=ALIGNMENT, BLOCK_SIZE=BLOCK_SIZE, NUM_ITERS=NUM_ITERS,
)
sol._scatter_tokens_kernel[(grid_size,)](
    hidden_states, sorted_hidden, flat_topk_ids,
    packed_layout, write_counters, output_index,
    bs, TOPK, START_EXPERT, LOCAL_EXPERTS,
    hidden_states.stride(0), sorted_hidden.stride(0),
    HIDDEN_SIZE=HIDDEN_SIZE, HIDDEN_SIZE_PAD=HIDDEN_SIZE_PAD,
    sorted_scales_ptr=sorted_scales, FP8_MAX=FP8_MAX,
    num_warps=8,
)
sol.moe_gather(gemm_output, topk_weights, output_index)
torch.cuda.synchronize()
