import torch
import triton
import triton.language as tl

@triton.jit
def test_fp8_kernel(in_ptr, out_ptr, scale_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(in_ptr + offs)
    x_f32 = x.to(tl.float32)
    max_val = tl.max(tl.abs(x_f32))
    scale = max_val / 448.0
    scale_inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
    q = (x_f32 * scale_inv).to(tl.float8e4nv)
    tl.store(out_ptr + offs, q)
    tl.store(scale_ptr, scale)

N = 128
x = torch.randn(N, device='cuda', dtype=torch.bfloat16)
out = torch.empty(N, device='cuda', dtype=torch.float8_e4m3fn)
scale = torch.empty(1, device='cuda', dtype=torch.float32)

test_fp8_kernel[(1,)](x, out, scale, N=N)
torch.cuda.synchronize()
print('Scale:', scale.item())
print('Output dtype:', out.dtype)
dequant = out.float() * scale.item()
err = (dequant - x.float()).abs().max().item()
print('Max dequant error:', err)
print('Relative error:', err / x.float().abs().max().item())
print('SUCCESS')
