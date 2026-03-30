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
__global__ void __launch_bounds__(128, 16) fusedQKNormRopeStoreKernel(
    __nv_bfloat16 const* __restrict__ qkv,
    int const num_heads_q,
    int const num_heads_k,
    int const num_heads_v,
    float const eps,
    __nv_bfloat16 const* __restrict__ q_weight,
    __nv_bfloat16 const* __restrict__ k_weight,
    int64_t const* __restrict__ position_ids,
    int const num_tokens,
    int const rotary_dim,
    __nv_bfloat16 const* __restrict__ cos_sin_cache,  // [max_pos, rotary_dim] BF16
    __nv_fp8_e4m3* q_output,
    float const* __restrict__ q_scale_ptr,
    int const q_output_stride,
    __nv_fp8_e4m3* k_cache,
    __nv_fp8_e4m3* v_cache,
    int const* __restrict__ out_loc,
    float const* __restrict__ k_scale_ptr,
    float const* __restrict__ v_scale_ptr,
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
  // Use int64_t to avoid overflow for large models (e.g. 128 heads × 256 dim × 64K tokens)
  int64_t offsetWarp;
  if (headType == Q_HEAD) {
    offsetWarp = static_cast<int64_t>(tokenIdx) * num_all_heads * head_dim + headIdx * head_dim;
  } else if (headType == K_HEAD) {
    offsetWarp = static_cast<int64_t>(tokenIdx) * num_all_heads * head_dim + num_heads_q * head_dim + headIdx * head_dim;
  } else {  // V_HEAD
    offsetWarp = static_cast<int64_t>(tokenIdx) * num_all_heads * head_dim +
                 (num_heads_q + num_heads_k) * head_dim + headIdx * head_dim;
  }
  int64_t offsetThread = offsetWarp + laneId * numElemsPerThread;

  // ---- Load from QKV buffer ----
  {
    vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[offsetThread]);
    #pragma unroll
    for (int i = 0; i < vecSize; i++) {
      float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(
          reinterpret_cast<uint*>(&vec) + i));
      elements[2 * i] = vals.x;
      elements[2 * i + 1] = vals.y;
    }
  }

  // ---- V heads: no norm/rope, just FP8 cast + store ----
  if (headType == V_HEAD) {
    int const cacheSlot = out_loc[tokenIdx];
    int64_t const cacheOffset = static_cast<int64_t>(cacheSlot) * kv_cache_stride
                                + headIdx * head_dim + laneId * numElemsPerThread;

    // Pack FP8 bytes into uint32 for coalesced store
    uint32_t packed = 0;
    #pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(elements[i] / *v_scale_ptr);
      packed |= (static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8)) << (i * 8));
    }
    *reinterpret_cast<uint32_t*>(&v_cache[cacheOffset]) = packed;
    return;
  }

  // ---- Q and K heads: RMSNorm ----
  float sumOfSquares = 0.0f;
  #pragma unroll
  for (int i = 0; i < numElemsPerThread; i++) {
    sumOfSquares += elements[i] * elements[i];
  }
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

  // ---- Q and K heads: RoPE (from precomputed cos_sin_cache) + FP8 store ----
  int const rotary_lanes = rotary_dim / numElemsPerThread;
  bool const applyRotary = (laneId < rotary_lanes);

  // Determine output location
  float const* scale_ptr;
  __nv_fp8_e4m3* out_ptr;
  int64_t out_offset;
  if (headType == Q_HEAD) {
    scale_ptr = q_scale_ptr;
    out_ptr = q_output;
    out_offset = static_cast<int64_t>(tokenIdx) * q_output_stride + headIdx * head_dim + laneId * numElemsPerThread;
  } else {
    scale_ptr = k_scale_ptr;
    out_ptr = k_cache;
    int const cacheSlot = out_loc[tokenIdx];
    out_offset = static_cast<int64_t>(cacheSlot) * kv_cache_stride + headIdx * head_dim + laneId * numElemsPerThread;
  }

  if (applyRotary) {
    int64_t const pos = position_ids[tokenIdx];
    int const half_rotary = rotary_dim / 2;
    __nv_bfloat16 const* cache_row = cos_sin_cache + static_cast<int64_t>(pos) * rotary_dim;

    if constexpr (interleave) {
      #pragma unroll
      for (int i = 0; i < numElemsPerThread; i++) {
        float e2 = (i % 2 == 0) ? -elements[i + 1] : elements[i - 1];
        int half_dim = (laneId * numElemsPerThread + i) / 2;
        float cos_val = __bfloat162float(cache_row[half_dim]);
        float sin_val = __bfloat162float(cache_row[half_rotary + half_dim]);
        elements[i] = (elements[i] * cos_val + e2 * sin_val) ;
      }
    } else {
      // NeoX style — vectorized cos/sin cache loads
      __syncwarp();
      int const half_rotary_lanes = rotary_lanes / 2;
      unsigned int active_mask = (1u << rotary_lanes) - 1;
      int base_half = (laneId * numElemsPerThread) % half_rotary;

      // Load 4 BF16 cos values and convert to FP32
      __nv_bfloat162 cos_p0 = *reinterpret_cast<__nv_bfloat162 const*>(&cache_row[base_half]);
      __nv_bfloat162 cos_p1 = *reinterpret_cast<__nv_bfloat162 const*>(&cache_row[base_half + 2]);
      float2 cf0 = __bfloat1622float2(cos_p0);
      float2 cf1 = __bfloat1622float2(cos_p1);
      float cos_arr[4] = {cf0.x, cf0.y, cf1.x, cf1.y};

      // Load 4 BF16 sin values and convert to FP32
      __nv_bfloat162 sin_p0 = *reinterpret_cast<__nv_bfloat162 const*>(&cache_row[half_rotary + base_half]);
      __nv_bfloat162 sin_p1 = *reinterpret_cast<__nv_bfloat162 const*>(&cache_row[half_rotary + base_half + 2]);
      float2 sf0 = __bfloat1622float2(sin_p0);
      float2 sf1 = __bfloat1622float2(sin_p1);
      float sin_arr[4] = {sf0.x, sf0.y, sf1.x, sf1.y};

      #pragma unroll
      for (int i = 0; i < numElemsPerThread; i++) {
        float e2 = __shfl_xor_sync(active_mask, elements[i], half_rotary_lanes);
        if (laneId < half_rotary_lanes) {
          e2 = -e2;
        }
        elements[i] = (elements[i] * cos_arr[i] + e2 * sin_arr[i]) ;
      }
      __syncwarp();
    }
  }

  // Vectorized uint32 FP8 store
  uint32_t packed = 0;
  #pragma unroll
  for (int i = 0; i < numElemsPerThread; i++) {
    __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(elements[i] / *scale_ptr);
    packed |= (static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8)) << (i * 8));
  }
  *reinterpret_cast<uint32_t*>(&out_ptr[out_offset]) = packed;
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
    bool const interleave, int64_t const* position_ids,
    int const rotary_dim, __nv_bfloat16 const* cos_sin_cache,
    void* q_output, float const* q_scale, int const q_output_stride,
    void* k_cache, void* v_cache, int const* out_loc,
    float const* k_scale, float const* v_scale,
    int const kv_cache_stride, cudaStream_t stream) {

  constexpr int blockSize = 128;
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
          position_ids, num_tokens, rotary_dim, cos_sin_cache,              \
          reinterpret_cast<__nv_fp8_e4m3*>(q_output),                      \
          q_scale, q_output_stride,                                        \
          reinterpret_cast<__nv_fp8_e4m3*>(k_cache),                       \
          reinterpret_cast<__nv_fp8_e4m3*>(v_cache),                       \
          out_loc, k_scale, v_scale, kv_cache_stride);                     \
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
    int64_t rotary_dim,
    torch::Tensor& cos_sin_cache,
    torch::Tensor& q_output, torch::Tensor& q_scale,
    torch::Tensor& k_cache, torch::Tensor& v_cache,
    torch::Tensor& out_loc, torch::Tensor& k_scale, torch::Tensor& v_scale) {

  CHECK_INPUT(qkv, torch::kBFloat16);
  CHECK_INPUT(position_ids, torch::kInt64);
  CHECK_INPUT(q_weight, torch::kBFloat16);
  CHECK_INPUT(k_weight, torch::kBFloat16);
  CHECK_INPUT(out_loc, torch::kInt32);
  CHECK_INPUT(cos_sin_cache, torch::kBFloat16);
  CHECK_TH_CUDA(q_output); CHECK_CONTIGUOUS(q_output);
  CHECK_TH_CUDA(k_cache); CHECK_CONTIGUOUS(k_cache);
  CHECK_TH_CUDA(v_cache); CHECK_CONTIGUOUS(v_cache);

  TORCH_CHECK(q_scale.numel() == 1, "q_scale must be a single-element tensor");
  TORCH_CHECK(k_scale.numel() == 1, "k_scale must be a single-element tensor");
  TORCH_CHECK(v_scale.numel() == 1, "v_scale must be a single-element tensor");
  CHECK_INPUT(q_scale, torch::kFloat32);
  CHECK_INPUT(k_scale, torch::kFloat32);
  CHECK_INPUT(v_scale, torch::kFloat32);

  int64_t num_tokens = qkv.size(0);
  int64_t q_output_stride = num_heads_q * head_dim;
  int64_t kv_cache_stride = num_heads_k * head_dim;
  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  launchFusedQKNormRopeStore(
      qkv.data_ptr(), static_cast<int>(num_tokens),
      static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
      static_cast<int>(num_heads_v), static_cast<int>(head_dim),
      static_cast<float>(eps), q_weight.data_ptr(), k_weight.data_ptr(),
      !is_neox, reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
      static_cast<int>(rotary_dim),
      reinterpret_cast<__nv_bfloat16 const*>(cos_sin_cache.data_ptr()),
      q_output.data_ptr(),
      reinterpret_cast<float const*>(q_scale.data_ptr()),
      static_cast<int>(q_output_stride),
      k_cache.data_ptr(), v_cache.data_ptr(),
      reinterpret_cast<int const*>(out_loc.data_ptr()),
      reinterpret_cast<float const*>(k_scale.data_ptr()),
      reinterpret_cast<float const*>(v_scale.data_ptr()),
      static_cast<int>(kv_cache_stride), stream);
}
