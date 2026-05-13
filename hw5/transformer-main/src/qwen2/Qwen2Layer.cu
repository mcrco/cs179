#include <cuda_bf16.h>

#include "Qwen2Config.h"
#include "Qwen2Layer.cuh"
#include "../CudaBuffer.cuh"
#include <memory>

#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "../gpu_ops/LayerNorm.cuh"
#include "../ErrorCheck.h"
#include "../gpu_ops/RoPE.cuh"
#include "../gpu_ops/GroupQueryAttention.cuh"
#include "../gpu_ops/SiLUMult.cuh"

const int RES_ADD_THREADS = 128;
const int RES_ADD_MAX_BLOCKS = 1024;

__device__ inline float normalize_float(float x) {
    return x;
}

__device__ inline float normalize_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template<typename residual_float_t>
__global__ void residualAddKernel(__nv_bfloat16 *hidden_state, residual_float_t *residual, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        float hval = __bfloat162float(hidden_state[i]);
        float rval = normalize_float(residual[i]);
        hidden_state[idx] = __float2bfloat16(hval + rval);
    }
}

template<Qwen2Size QWEN2_SIZE>
Qwen2Layer<QWEN2_SIZE>::Qwen2Layer(uint32_t layer_num, uint32_t max_seq_len):
    layer_num(layer_num), input_layernorm(Qwen2Config::hidden_size()), post_attention_layernorm(Qwen2Config::hidden_size()) {
    queries = std::make_shared<CudaBuffer>(Qwen2Config::queries_size() * sizeof(float));
    attention_output = std::make_shared<CudaBuffer>(Qwen2Config::hidden_size() * sizeof(float));
    gate_proj = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(float));
    up_proj = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(float));
    down_proj = std::make_shared<CudaBuffer>(Qwen2Config::intermediate_size() * sizeof(float));
}

template<Qwen2Size QWEN2_SIZE>
void Qwen2Layer<QWEN2_SIZE>::forward(const std::shared_ptr<CudaBuffer>& k_cache, const std::shared_ptr<CudaBuffer> &v_cache, const std::shared_ptr<CudaBuffer> &hidden_state, int32_t seq_len, cudaStream_t stream) {
    __nv_bfloat16 *q_proj_weight_ptr = static_cast<__nv_bfloat16 *>(q_proj_weight->data);
    __nv_bfloat16 *q_proj_bias_ptr = static_cast<__nv_bfloat16 *>(q_proj_bias->data);
    __nv_bfloat16 *k_proj_weight_ptr = static_cast<__nv_bfloat16 *>(k_proj_weight->data);
    __nv_bfloat16 *k_proj_bias_ptr = static_cast<__nv_bfloat16 *>(k_proj_bias->data);
    __nv_bfloat16 *v_proj_weight_ptr = static_cast<__nv_bfloat16 *>(v_proj_weight->data);
    __nv_bfloat16 *v_proj_bias_ptr = static_cast<__nv_bfloat16 *>(v_proj_bias->data);
    __nv_bfloat16 *gate_proj_weight_ptr = static_cast<__nv_bfloat16 *>(gate_proj_weight->data);
    __nv_bfloat16 *up_proj_weight_ptr = static_cast<__nv_bfloat16 *>(up_proj_weight->data);
    __nv_bfloat16 *down_proj_weight_ptr = static_cast<__nv_bfloat16 *>(down_proj_weight->data);

    __nv_bfloat16 *k_cache_ptr = static_cast<__nv_bfloat16 *>(k_cache->data);
    __nv_bfloat16 *v_cache_ptr = static_cast<__nv_bfloat16 *>(v_cache->data);
    __nv_bfloat16 *hidden_state_ptr = static_cast<__nv_bfloat16 *>(hidden_state->data);
    __nv_bfloat16 *queries_ptr = static_cast<__nv_bfloat16 *>(attention_output->data);

    float *attention_output_ptr = static_cast<float *>(attention_output->data);
    __nv_bfloat16 *gate_proj_ptr = static_cast<__nv_bfloat16 *>(gate_proj->data);
    __nv_bfloat16 *up_proj_ptr = static_cast<__nv_bfloat16 *>(up_proj->data);
    __nv_bfloat16 *down_proj_ptr = static_cast<__nv_bfloat16 *>(down_proj->data);

    input_layernorm.normalize_hidden_state(hidden_state, hidden_state, stream);

    MatrixVectorMultiply::bf16_matmul(Qwen2Config::hidden_size(), Qwen2Config::hidden_size(), q_proj_weight_ptr, q_proj_bias_ptr, hidden_state_ptr, queries_ptr, stream);
    __nv_bfloat16 *key_ptr = k_cache_ptr + seq_len * Qwen2Config::num_layers() * Qwen2Config::keys_size();
    __nv_bfloat16 *val_ptr = v_cache_ptr + seq_len * Qwen2Config::num_layers() * Qwen2Config::values_size();
    MatrixVectorMultiply::bf16_matmul(Qwen2Config::hidden_size(), Qwen2Config::hidden_size(), k_proj_weight_ptr, k_proj_bias_ptr, hidden_state_ptr, key_ptr, stream);
    MatrixVectorMultiply::bf16_matmul(Qwen2Config::hidden_size(), Qwen2Config::hidden_size(), v_proj_weight_ptr, v_proj_bias_ptr, hidden_state_ptr, val_ptr, stream);

    GroupQueryAttention<QWEN2_SIZE>::sdpa(queries_ptr, k_cache_ptr, v_cache_ptr, attention_output_ptr, layer_num, seq_len, stream);
    int threads = RES_ADD_THREADS;
    int blocks = min(RES_ADD_MAX_BLOCKS, ((int)Qwen2Config::hidden_size() + threads - 1) / threads);
    residualAddKernel<float><<<blocks, threads, 0, stream>>>(hidden_state_ptr, attention_output_ptr, Qwen2Config::hidden_size());
    checkCuda(cudaGetLastError());

    post_attention_layernorm.normalize_hidden_state(hidden_state, hidden_state, stream);

    MatrixVectorMultiply::bf16_matmul(Qwen2Config::hidden_size(), Qwen2Config::intermediate_size(), gate_proj_weight_ptr, nullptr, hidden_state_ptr, gate_proj_ptr, stream);
    MatrixVectorMultiply::bf16_matmul(Qwen2Config::hidden_size(), Qwen2Config::intermediate_size(), up_proj_weight_ptr, nullptr, hidden_state_ptr, up_proj_ptr, stream);
    SiLUMult::silu_mult_in_place(gate_proj, up_proj, stream);
    MatrixVectorMultiply::bf16_matmul(Qwen2Config::intermediate_size(), Qwen2Config::hidden_size(), down_proj_weight_ptr, nullptr, gate_proj_ptr, down_proj_ptr, stream);
    residualAddKernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(hidden_state_ptr, down_proj_ptr, Qwen2Config::hidden_size());
}

template class Qwen2Layer<QWEN2_0_5B>;
