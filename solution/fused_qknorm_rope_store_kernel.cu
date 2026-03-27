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
__global__ void __launch_bounds__(256, 8) fusedQKNormRopeStoreKernel(
    __nv_bfloat16 const* __restrict__ qkv,
    int const num_heads_q,
    int const num_heads_k,
    int const num_heads_v,
    float const eps,
    __nv_bfloat16 const* __restrict__ q_weight,
    __nv_bfloat16 const* __restrict__ k_weight,
    int const* __restrict__ position_ids,
    int const num_tokens,
    float attention_factor,
    int const rotary_dim,
    float const* __restrict__ cos_sin_cache,  // [max_pos, rotary_dim] FP32
    __nv_fp8_e4m3* q_output,
    float const q_scale_inv,
    int const q_output_stride,
    __nv_fp8_e4m3* k_cache,
    __nv_fp8_e4m3* v_cache,
    int const* __restrict__ out_loc,
    float const k_scale_inv,
    float const v_scale_inv,
    int const kv_cache_stride
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

    // Pack FP8 bytes into uint32 for coalesced store
    uint32_t packed = 0;
    #pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(elements[i] * v_scale_inv);
      packed |= (static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8)) << (i * 8));
    }
    *reinterpret_cast<uint32_t*>(&v_cache[cacheOffset]) = packed;
    return;
  }

  // ---- Q and K heads: RMSNorm ----
  sumOfSquares = fused_helpers::warpReduceSum(sumOfSquares);
  float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

  // Vectorized weight load
  bool const isQ = (headType == Q_HEAD);
  {
    __nv_bfloat16 const* wptr = isQ ? q_weight : k_weight;
    vec_T wvec = *reinterpret_cast<vec_T const*>(&wptr[laneId * numElemsPerThread]);
    #pragma unroll
    for (int i = 0; i < vecSize; i++) {
      float2 wvals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(
          reinterpret_cast<uint const*>(&wvec) + i));
      elements[2 * i] *= rms_rcp * wvals.x;
      elements[2 * i + 1] *= rms_rcp * wvals.y;
    }
  }

  // ---- Q and K heads: RoPE (from precomputed cos_sin_cache) ----
  float elements2[numElemsPerThread];
  int const rotary_lanes = rotary_dim / numElemsPerThread;
  bool const applyRotary = (laneId < rotary_lanes);

  if (applyRotary) {
    int const pos = position_ids[tokenIdx];
    int const half_rotary = rotary_dim / 2;
    float const* cache_row = cos_sin_cache + pos * rotary_dim;

    if constexpr (interleave) {
      #pragma unroll
      for (int i = 0; i < numElemsPerThread; i++) {
        elements2[i] = (i % 2 == 0) ? -elements[i + 1] : elements[i - 1];
        int half_dim = (laneId * numElemsPerThread + i) / 2;
        float cos_val = cache_row[half_dim];
        float sin_val = cache_row[half_rotary + half_dim];
        elements[i] = (elements[i] * cos_val + elements2[i] * sin_val) * attention_factor;
      }
    } else {
      // NeoX style
      __syncwarp();
      int const half_rotary_lanes = rotary_lanes / 2;
      unsigned int active_mask = (1u << rotary_lanes) - 1;
      #pragma unroll
      for (int i = 0; i < numElemsPerThread; i++) {
        elements2[i] = __shfl_xor_sync(active_mask, elements[i], half_rotary_lanes);
        if (laneId < half_rotary_lanes) {
          elements2[i] = -elements2[i];
        }
        int dim_idx = laneId * numElemsPerThread + i;
        dim_idx = (dim_idx * 2) % rotary_dim;
        int half_dim = dim_idx / 2;
        float cos_val = cache_row[half_dim];
        float sin_val = cache_row[half_rotary + half_dim];
        elements[i] = (elements[i] * cos_val + elements2[i] * sin_val) * attention_factor;
      }
      __syncwarp();
    }
  }

  // ---- Store results (vectorized uint32 writes) ----
  if (headType == Q_HEAD) {
    int const qOutOffset = tokenIdx * q_output_stride + headIdx * head_dim
                            + laneId * numElemsPerThread;

    uint32_t packed = 0;
    #pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(elements[i] * q_scale_inv);
      packed |= (static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8)) << (i * 8));
    }
    *reinterpret_cast<uint32_t*>(&q_output[qOutOffset]) = packed;
  } else {
    int const cacheSlot = out_loc[tokenIdx];
    int const cacheOffset = cacheSlot * kv_cache_stride + headIdx * head_dim
                            + laneId * numElemsPerThread;

    uint32_t packed = 0;
    #pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(elements[i] * k_scale_inv);
      packed |= (static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8)) << (i * 8));
    }
    *reinterpret_cast<uint32_t*>(&k_cache[cacheOffset]) = packed;
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
    void const* qkv, int const num_tokens,
    int const num_heads_q, int const num_heads_k, int const num_heads_v,
    int const head_dim, float const eps,
    void const* q_weight, void const* k_weight,
    bool const interleave, int const* position_ids,
    float attention_factor, int const rotary_dim,
    float const* cos_sin_cache,
    void* q_output, float const q_scale_inv, int const q_output_stride,
    void* k_cache, void* v_cache, int const* out_loc,
    float const k_scale_inv, float const v_scale_inv,
    int const kv_cache_stride, cudaStream_t stream) {

  constexpr int blockSize = 256;
  int const warpsPerBlock = blockSize / 32;
  int const totalHeads = num_heads_q + num_heads_k + num_heads_v;
  int const totalWarps = num_tokens * totalHeads;
  int const gridSize = fused_helpers::divUp(totalWarps, warpsPerBlock);

  #define LAUNCH_KERNEL(HD)                                                \
    DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {                          \
      fusedQKNormRopeStoreKernel<HD, INTERLEAVE><<<gridSize, blockSize, 0, stream>>>(  \
          reinterpret_cast<__nv_bfloat16 const*>(qkv),                     \
          num_heads_q, num_heads_k, num_heads_v, eps,                      \
          reinterpret_cast<__nv_bfloat16 const*>(q_weight),                \
          reinterpret_cast<__nv_bfloat16 const*>(k_weight),                \
          position_ids, num_tokens,                                        \
          attention_factor, rotary_dim, cos_sin_cache,                     \
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
    torch::Tensor& qkv,
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v,
    int64_t head_dim, double eps,
    torch::Tensor& q_weight, torch::Tensor& k_weight,
    bool is_neox, torch::Tensor& position_ids,
    double attention_factor, int64_t rotary_dim,
    torch::Tensor& cos_sin_cache,
    torch::Tensor& q_output, double q_scale,
    torch::Tensor& k_cache, torch::Tensor& v_cache,
    torch::Tensor& out_loc, double k_scale, double v_scale) {

  CHECK_INPUT(qkv, torch::kBFloat16);
  CHECK_INPUT(position_ids, torch::kInt32);
  CHECK_INPUT(q_weight, torch::kBFloat16);
  CHECK_INPUT(k_weight, torch::kBFloat16);
  CHECK_INPUT(out_loc, torch::kInt32);
  CHECK_INPUT(cos_sin_cache, torch::kFloat32);
  CHECK_TH_CUDA(q_output); CHECK_CONTIGUOUS(q_output);
  CHECK_TH_CUDA(k_cache); CHECK_CONTIGUOUS(k_cache);
  CHECK_TH_CUDA(v_cache); CHECK_CONTIGUOUS(v_cache);

  int64_t num_tokens = qkv.size(0);
  int64_t q_output_stride = num_heads_q * head_dim;
  int64_t kv_cache_stride = num_heads_k * head_dim;
  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  launchFusedQKNormRopeStore(
      qkv.data_ptr(), static_cast<int>(num_tokens),
      static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
      static_cast<int>(num_heads_v), static_cast<int>(head_dim),
      static_cast<float>(eps), q_weight.data_ptr(), k_weight.data_ptr(),
      !is_neox, reinterpret_cast<int const*>(position_ids.data_ptr()),
      static_cast<float>(attention_factor), static_cast<int>(rotary_dim),
      reinterpret_cast<float const*>(cos_sin_cache.data_ptr()),
      q_output.data_ptr(), static_cast<float>(1.0 / q_scale),
      static_cast<int>(q_output_stride),
      k_cache.data_ptr(), v_cache.data_ptr(),
      reinterpret_cast<int const*>(out_loc.data_ptr()),
      static_cast<float>(1.0 / k_scale), static_cast<float>(1.0 / v_scale),
      static_cast<int>(kv_cache_stride), stream);
}

// ============================================================================
// cos_sin_cache precompute (uses exact same powf+sincosf as original kernel)
// ============================================================================

__global__ void computeCosSinCacheKernel(
    float* __restrict__ cache, int total_elems, int half_rd, int rotary_dim,
    float base, float factor, float low, float high) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elems) return;
  int pos = idx / half_rd;
  int half_dim = idx % half_rd;
  float freq = powf(base, -2.0f * half_dim / static_cast<float>(rotary_dim));
  if (factor != 1.0f) {
    float interp = freq / factor;
    float ha = high;
    if (fabsf(low - ha) <= 1e-6f) ha += 0.001f;
    float lf = (static_cast<float>(half_dim) - low) / (ha - low);
    float rf = fminf(fmaxf(lf, 0.0f), 1.0f);
    float ef = 1.0f - rf;
    freq = interp * (1.0f - ef) + freq * ef;
  }
  float theta = static_cast<float>(pos) * freq;
  float sv, cv;
  sincosf(theta, &sv, &cv);
  cache[pos * rotary_dim + half_dim] = cv;
  cache[pos * rotary_dim + half_rd + half_dim] = sv;
}

void compute_cos_sin_cache(
    torch::Tensor& cache, int64_t max_pos, int64_t rotary_dim,
    double base, double factor, double low, double high) {
  CHECK_INPUT(cache, torch::kFloat32);
  auto stream = at::cuda::getCurrentCUDAStream(cache.get_device());
  int half_rd = static_cast<int>(rotary_dim / 2);
  int total = static_cast<int>(max_pos) * half_rd;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  computeCosSinCacheKernel<<<blocks, threads, 0, stream>>>(
      cache.data_ptr<float>(), total, half_rd, static_cast<int>(rotary_dim),
      static_cast<float>(base), static_cast<float>(factor),
      static_cast<float>(low), static_cast<float>(high));
}
