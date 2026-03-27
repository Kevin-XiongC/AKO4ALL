/*
 * Fused QK-Norm + RoPE + FP8-Cast + KV-Cache-Store kernel
 *
 * Combines the following pipeline into a single GPU kernel:
 *   1. RMS normalization on Q and K heads
 *   2. Rotary Position Embedding (RoPE) with YaRN support
 *   3. FP8 E4M3 quantization of Q, K, and V (with per-tensor scale)
 *   4. Scatter-write of quantized K/V into the paged KV cache
 *
 * Q heads are written as FP8 E4M3 to a separate q_output buffer.
 * K/V heads are written directly to the KV cache as FP8 bytes.
 *
 * Adapted from sgl-kernel/csrc/moe/fused_qknorm_rope_kernel.cu
 */

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/cuda.h>

#include <cmath>

#define CHECK_TYPE(x, st) \
  TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), ", while ", st, " is expected")
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, st) \
  CHECK_TH_CUDA(x);        \
  CHECK_CONTIGUOUS(x);     \
  CHECK_TYPE(x, st)

#define FINAL_MASK 0xffffffff

// ============================================================================
// Utility helpers
// ============================================================================

namespace fused_helpers {

template <typename T, int num>
struct packed_as;
template <> struct packed_as<uint, 1> { using type = uint;  };
template <> struct packed_as<uint, 2> { using type = uint2; };
template <> struct packed_as<uint, 4> { using type = uint4; };

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
inline __device__ __host__ T divUp(T m, T n) {
  return (m + n - 1) / n;
}

}  // namespace fused_helpers

// ============================================================================
// YaRN frequency computation (same as original)
// ============================================================================

__device__ inline float compute_freq_yarn(
    float base, int head_dim, int half_dim,
    float factor, float low, float high) {
  float freq = powf(base, -2.0f * half_dim / static_cast<float>(head_dim));

  if (factor != 1.0f) {
    float inv_freq_extrapolation = freq;
    float inv_freq_interpolation = freq / factor;

    float high_adj = high;
    if (fabsf(low - high_adj) <= 1e-6f) {
      high_adj += 0.001f;
    }

    float linear_func = (static_cast<float>(half_dim) - low) / (high_adj - low);
    float ramp_func = fminf(fmaxf(linear_func, 0.0f), 1.0f);
    float inv_freq_extrapolation_factor = 1.0f - ramp_func;

    freq = inv_freq_interpolation * (1.0f - inv_freq_extrapolation_factor) +
           inv_freq_extrapolation * inv_freq_extrapolation_factor;
  }

  return freq;
}

// ============================================================================
// Main fused kernel
// ============================================================================

// Each warp processes one (token, head) pair.
//   - Q heads: RMSNorm → RoPE → scale → FP8 cast → write to q_output
//   - K heads: RMSNorm → RoPE → scale → FP8 cast → scatter write to k_cache
//   - V heads: (no norm/rope) → scale → FP8 cast → scatter write to v_cache
//
// Template parameters:
//   head_dim   – dimension of each head (must be multiple of 64)
//   interleave – true for interleaved RoPE, false for NeoX style
template <int head_dim, bool interleave>
__global__ void fusedQKNormRopeStoreKernel(
    __nv_bfloat16 const* qkv,      // [num_tokens, (nq+nk+nv)*head_dim]  (read-only)
    int const num_heads_q,
    int const num_heads_k,
    int const num_heads_v,
    float const eps,
    __nv_bfloat16 const* q_weight,  // [head_dim]
    __nv_bfloat16 const* k_weight,  // [head_dim]
    float const base,
    int const* position_ids,        // [num_tokens]
    int const num_tokens,
    // YaRN parameters
    float factor,
    float low,
    float high,
    float attention_factor,
    int const rotary_dim,
    // Q output parameters
    __nv_fp8_e4m3* q_output,        // [num_tokens, num_heads_q * head_dim]  (FP8 bytes)
    float const q_scale_inv,        // 1.0 / q_scale  (pre-inverted for multiply)
    int const q_output_stride,      // num_heads_q * head_dim
    // KV cache store parameters
    __nv_fp8_e4m3* k_cache,         // [max_tokens, num_kv_heads * head_dim]  (as FP8 bytes)
    __nv_fp8_e4m3* v_cache,         // [max_tokens, num_kv_heads * head_dim]  (as FP8 bytes)
    int const* out_loc,             // [num_tokens]  – cache slot index per token
    float const k_scale_inv,        // 1.0 / k_scale  (pre-inverted for multiply)
    float const v_scale_inv,        // 1.0 / v_scale
    int const kv_cache_stride       // num_kv_heads * head_dim  (elements per cache row)
) {
  int const warpsPerBlock = blockDim.x / 32;
  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;

  int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

  // Total heads per token: Q + K + V
  int const total_heads = num_heads_q + num_heads_k + num_heads_v;

  int const tokenIdx = globalWarpIdx / total_heads;
  int const localHeadIdx = globalWarpIdx % total_heads;

  if (tokenIdx >= num_tokens) return;

  // Determine head type: Q, K, or V
  enum HeadType { Q_HEAD, K_HEAD, V_HEAD };
  HeadType headType;
  int headIdx;
  if (localHeadIdx < num_heads_q) {
    headType = Q_HEAD;
    headIdx = localHeadIdx;
  } else if (localHeadIdx < num_heads_q + num_heads_k) {
    headType = K_HEAD;
    headIdx = localHeadIdx - num_heads_q;
  } else {
    headType = V_HEAD;
    headIdx = localHeadIdx - num_heads_q - num_heads_k;
  }

  int const num_all_heads = num_heads_q + num_heads_k + num_heads_v;

  static_assert(
      head_dim % (32 * 2) == 0,
      "head_dim must be divisible by 64");
  constexpr int numElemsPerThread = head_dim / 32;
  float elements[numElemsPerThread];
  constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
  static_assert(elemSizeBytes % 4 == 0);
  constexpr int vecSize = elemSizeBytes / 4;
  using vec_T = typename fused_helpers::packed_as<uint, vecSize>::type;

  // Compute offset into qkv buffer for this (token, head)
  int offsetWarp;
  if (headType == Q_HEAD) {
    offsetWarp = tokenIdx * num_all_heads * head_dim + headIdx * head_dim;
  } else if (headType == K_HEAD) {
    offsetWarp = tokenIdx * num_all_heads * head_dim + num_heads_q * head_dim + headIdx * head_dim;
  } else {  // V_HEAD
    offsetWarp = tokenIdx * num_all_heads * head_dim +
                 (num_heads_q + num_heads_k) * head_dim + headIdx * head_dim;
  }
  int offsetThread = offsetWarp + laneId * numElemsPerThread;

  // ---- Load from QKV buffer ----
  float sumOfSquares = 0.0f;
  {
    vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[offsetThread]);
    for (int i = 0; i < vecSize; i++) {
      float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(
          reinterpret_cast<uint*>(&vec) + i));
      sumOfSquares += vals.x * vals.x;
      sumOfSquares += vals.y * vals.y;
      elements[2 * i] = vals.x;
      elements[2 * i + 1] = vals.y;
    }
  }

  // ---- V heads: no norm/rope, just FP8 cast + store ----
  if (headType == V_HEAD) {
    int const cacheSlot = out_loc[tokenIdx];
    int const cacheOffset = cacheSlot * kv_cache_stride + headIdx * head_dim
                            + laneId * numElemsPerThread;

    for (int i = 0; i < numElemsPerThread; i++) {
      // V values are already in BF16 precision (loaded from QKV buffer),
      // so the float→BF16 round-trip is identity here. Just scale and cast.
      float val = elements[i] * v_scale_inv;
      v_cache[cacheOffset + i] = __nv_fp8_e4m3(val);
    }
    return;
  }

  // ---- Q and K heads: RMSNorm ----
  sumOfSquares = fused_helpers::warpReduceSum(sumOfSquares);
  float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

  bool const isQ = (headType == Q_HEAD);
  for (int i = 0; i < numElemsPerThread; i++) {
    int dim = laneId * numElemsPerThread + i;
    float weight = isQ ? __bfloat162float(q_weight[dim])
                       : __bfloat162float(k_weight[dim]);
    elements[i] *= rms_rcp * weight;
  }

  // ---- Q and K heads: RoPE ----
  float elements2[numElemsPerThread];
  float cos_vals[numElemsPerThread];
  float sin_vals[numElemsPerThread];
  float pos_id = static_cast<float>(position_ids[tokenIdx]);
  int const rotary_lanes = rotary_dim / numElemsPerThread;
  bool const applyRotary = (laneId < rotary_lanes);

  if (applyRotary) {
    if constexpr (interleave) {
      for (int i = 0; i < numElemsPerThread; i++) {
        elements2[i] = (i % 2 == 0) ? -elements[i + 1] : elements[i - 1];
        int dim_idx = laneId * numElemsPerThread + i;
        int half_dim = dim_idx / 2;
        float freq = compute_freq_yarn(base, rotary_dim, half_dim, factor, low, high);
        float theta = pos_id * freq;
        __sincosf(theta, &sin_vals[i], &cos_vals[i]);
      }
    } else {
      // NeoX style
      __syncwarp();
      int const half_rotary_lanes = rotary_lanes / 2;
      unsigned int active_mask = (1u << rotary_lanes) - 1;
      for (int i = 0; i < numElemsPerThread; i++) {
        elements2[i] = __shfl_xor_sync(active_mask, elements[i], half_rotary_lanes);
        if (laneId < half_rotary_lanes) {
          elements2[i] = -elements2[i];
        }
        int dim_idx = laneId * numElemsPerThread + i;
        dim_idx = (dim_idx * 2) % rotary_dim;
        int half_dim = dim_idx / 2;
        float freq = compute_freq_yarn(base, rotary_dim, half_dim, factor, low, high);
        float theta = pos_id * freq;
        __sincosf(theta, &sin_vals[i], &cos_vals[i]);
      }
      __syncwarp();
    }

    for (int i = 0; i < numElemsPerThread; i++) {
      elements[i] = (elements[i] * cos_vals[i] + elements2[i] * sin_vals[i]) * attention_factor;
    }
  }

  // ---- Store results ----
  if (headType == Q_HEAD) {
    // Q: round to BF16 precision, then cast to FP8 E4M3 and write to q_output
    int const qOutOffset = tokenIdx * q_output_stride + headIdx * head_dim
                            + laneId * numElemsPerThread;

    for (int i = 0; i < numElemsPerThread; i++) {
      float val = __bfloat162float(__float2bfloat16(elements[i]));
      val *= q_scale_inv;
      q_output[qOutOffset + i] = __nv_fp8_e4m3(val);
    }
  } else {
    // K: cast to FP8 E4M3 and scatter-write to k_cache
    // Round through BF16 first to match unfused path precision
    // (unfused path writes BF16 to QKV buffer, then reads back for FP8 cast)
    int const cacheSlot = out_loc[tokenIdx];
    int const cacheOffset = cacheSlot * kv_cache_stride + headIdx * head_dim
                            + laneId * numElemsPerThread;

    for (int i = 0; i < numElemsPerThread; i++) {
      // Round to BF16 precision (match unfused path where K is written as BF16 then re-read)
      float val = __bfloat162float(__float2bfloat16(elements[i]));
      val *= k_scale_inv;
      k_cache[cacheOffset + i] = __nv_fp8_e4m3(val);
    }
  }
}

// ============================================================================
// Dispatch helpers
// ============================================================================

#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    const bool INTERLEAVE = true;                        \
    __VA_ARGS__                                          \
  } else {                                               \
    const bool INTERLEAVE = false;                       \
    __VA_ARGS__                                          \
  }

void launchFusedQKNormRopeStore(
    void const* qkv,
    int const num_tokens,
    int const num_heads_q,
    int const num_heads_k,
    int const num_heads_v,
    int const head_dim,
    float const eps,
    void const* q_weight,
    void const* k_weight,
    float const base,
    bool const interleave,
    int const* position_ids,
    float factor,
    float low,
    float high,
    float attention_factor,
    int const rotary_dim,
    // Q output parameters
    void* q_output,
    float const q_scale_inv,
    int const q_output_stride,
    // KV cache store parameters
    void* k_cache,
    void* v_cache,
    int const* out_loc,
    float const k_scale_inv,
    float const v_scale_inv,
    int const kv_cache_stride,
    cudaStream_t stream) {

  constexpr int blockSize = 256;
  int const warpsPerBlock = blockSize / 32;
  // Now we process Q + K + V heads (not just Q + K)
  int const totalHeads = num_heads_q + num_heads_k + num_heads_v;
  int const totalWarps = num_tokens * totalHeads;
  int const gridSize = fused_helpers::divUp(totalWarps, warpsPerBlock);
  dim3 gridDim(gridSize);
  dim3 blockDim(blockSize);

  #define LAUNCH_KERNEL(HD)                                                \
    DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {                          \
      fusedQKNormRopeStoreKernel<HD, INTERLEAVE><<<gridDim, blockDim, 0, stream>>>(  \
          reinterpret_cast<__nv_bfloat16 const*>(qkv),                     \
          num_heads_q, num_heads_k, num_heads_v, eps,                      \
          reinterpret_cast<__nv_bfloat16 const*>(q_weight),                \
          reinterpret_cast<__nv_bfloat16 const*>(k_weight),                \
          base, position_ids, num_tokens,                                  \
          factor, low, high, attention_factor, rotary_dim,                 \
          reinterpret_cast<__nv_fp8_e4m3*>(q_output),                      \
          q_scale_inv, q_output_stride,                                    \
          reinterpret_cast<__nv_fp8_e4m3*>(k_cache),                       \
          reinterpret_cast<__nv_fp8_e4m3*>(v_cache),                       \
          out_loc, k_scale_inv, v_scale_inv, kv_cache_stride);             \
    });

  switch (head_dim) {
    case 64:  LAUNCH_KERNEL(64);  break;
    case 128: LAUNCH_KERNEL(128); break;
    case 256: LAUNCH_KERNEL(256); break;
    default:
      TORCH_CHECK(false, "Unsupported head dimension: ", head_dim);
  }
  #undef LAUNCH_KERNEL
}

// ============================================================================
// Torch C++ entry point
// ============================================================================

void fused_qk_norm_rope_store(
    torch::Tensor& qkv,           // [num_tokens, (nq+nk+nv)*head_dim]  BF16 (read-only)
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,      // [head_dim]  BF16
    torch::Tensor& k_weight,      // [head_dim]  BF16
    double base,
    bool is_neox,
    torch::Tensor& position_ids,  // [num_tokens]  INT32
    double factor,
    double low,
    double high,
    double attention_factor,
    int64_t rotary_dim,
    // Q output
    torch::Tensor& q_output,      // [num_tokens, num_heads_q * head_dim]  UINT8 (FP8 E4M3)
    double q_scale,
    // KV cache parameters
    torch::Tensor& k_cache,       // [max_tokens, num_kv_heads * head_dim]  UINT8 (FP8 E4M3 bitcast)
    torch::Tensor& v_cache,       // [max_tokens, num_kv_heads * head_dim]  UINT8
    torch::Tensor& out_loc,       // [num_tokens]  INT32
    double k_scale,
    double v_scale) {

  // Input validation
  TORCH_CHECK(qkv.dim() == 2, "QKV must be 2D");
  TORCH_CHECK(position_ids.dim() == 1, "position_ids must be 1D");
  TORCH_CHECK(q_weight.dim() == 1 && q_weight.size(0) == head_dim);
  TORCH_CHECK(k_weight.dim() == 1 && k_weight.size(0) == head_dim);
  TORCH_CHECK(q_output.dim() == 2, "q_output must be 2D");
  TORCH_CHECK(k_cache.dim() == 2, "k_cache must be 2D");
  TORCH_CHECK(v_cache.dim() == 2, "v_cache must be 2D");
  TORCH_CHECK(out_loc.dim() == 1, "out_loc must be 1D");
  CHECK_INPUT(qkv, torch::kBFloat16);
  CHECK_INPUT(position_ids, torch::kInt32);
  CHECK_INPUT(q_weight, torch::kBFloat16);
  CHECK_INPUT(k_weight, torch::kBFloat16);
  CHECK_INPUT(out_loc, torch::kInt32);
  CHECK_TH_CUDA(q_output);
  CHECK_CONTIGUOUS(q_output);
  CHECK_TH_CUDA(k_cache);
  CHECK_CONTIGUOUS(k_cache);
  CHECK_TH_CUDA(v_cache);
  CHECK_CONTIGUOUS(v_cache);

  int64_t num_tokens = qkv.size(0);
  TORCH_CHECK(position_ids.size(0) == num_tokens);
  TORCH_CHECK(out_loc.size(0) == num_tokens);

  int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(qkv.size(1) == total_heads * head_dim);

  int64_t q_output_stride = num_heads_q * head_dim;
  TORCH_CHECK(q_output.size(0) == num_tokens, "q_output rows must match num_tokens");
  TORCH_CHECK(q_output.size(1) == q_output_stride, "q_output width must be num_heads_q * head_dim");

  int64_t kv_cache_stride = num_heads_k * head_dim;
  TORCH_CHECK(k_cache.size(1) == kv_cache_stride, "k_cache width must be num_kv_heads * head_dim");
  TORCH_CHECK(v_cache.size(1) == kv_cache_stride, "v_cache width must be num_kv_heads * head_dim");

  TORCH_CHECK(q_scale > 0, "q_scale must be positive");
  TORCH_CHECK(k_scale > 0, "k_scale must be positive");
  TORCH_CHECK(v_scale > 0, "v_scale must be positive");

  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  launchFusedQKNormRopeStore(
      qkv.data_ptr(),
      static_cast<int>(num_tokens),
      static_cast<int>(num_heads_q),
      static_cast<int>(num_heads_k),
      static_cast<int>(num_heads_v),
      static_cast<int>(head_dim),
      static_cast<float>(eps),
      q_weight.data_ptr(),
      k_weight.data_ptr(),
      static_cast<float>(base),
      !is_neox,  // interleave = !is_neox
      reinterpret_cast<int const*>(position_ids.data_ptr()),
      static_cast<float>(factor),
      static_cast<float>(low),
      static_cast<float>(high),
      static_cast<float>(attention_factor),
      static_cast<int>(rotary_dim),
      q_output.data_ptr(),
      static_cast<float>(1.0 / q_scale),
      static_cast<int>(q_output_stride),
      k_cache.data_ptr(),
      v_cache.data_ptr(),
      reinterpret_cast<int const*>(out_loc.data_ptr()),
      static_cast<float>(1.0 / k_scale),
      static_cast<float>(1.0 / v_scale),
      static_cast<int>(kv_cache_stride),
      stream);
}
