import math, torch, triton
import importlib.util

def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sol = _load_module("/root/AKO4ALL/solution/kernel.py", "sol")

bs, topk, num_experts, local_experts, hidden_size = 1024, 8, 160, 20, 5120
start_expert = 0
ALIGNMENT = 128

hidden_states = torch.randn(bs, hidden_size, device="cuda", dtype=torch.bfloat16)
topk_ids = torch.randint(0, num_experts, (bs, topk), device="cuda", dtype=torch.int32)
topk_weights = torch.softmax(torch.randn(bs, topk, device="cuda", dtype=torch.float32), dim=-1)
max_total_M = bs * topk + local_experts * (ALIGNMENT - 1)

# warmup
sorted_hidden, packed_layout, output_index = sol.moe_align_and_scatter(hidden_states, topk_ids, local_experts, start_expert, max_total_M)
gemm_output = torch.randn(max_total_M, hidden_size, device="cuda", dtype=torch.bfloat16)
out = sol.moe_gather(gemm_output, topk_weights, output_index)
torch.cuda.synchronize()

# profiled run
sorted_hidden, packed_layout, output_index = sol.moe_align_and_scatter(hidden_states, topk_ids, local_experts, start_expert, max_total_M)
out = sol.moe_gather(gemm_output, topk_weights, output_index)
torch.cuda.synchronize()
