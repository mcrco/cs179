#include "ArgMax.cuh"
#include <cfloat>
#include <cuda_bf16.h>
#include <memory>
#include <algorithm>
#include "../ErrorCheck.h"
#include <float.h>

const int ARGMAX_THREADS = 128;
const int ARGMAX_MAX_BLOCKS = 1024;

struct ValueIndexPair {
    float val;
    int idx;
};

__device__ inline ValueIndexPair pair_argmax(ValueIndexPair a, ValueIndexPair b) {
    if (a.val == b.val) {
        if (a.idx < b.idx) {
            return a;
        }
        return b;
    }
    if (a.val > b.val) {
        return a;
    }
    return b;
}

ArgMax::ArgMax(int32_t len) {
    temp_space = std::make_shared<CudaBuffer>(len * sizeof(ValueIndexPair));
    result_space = std::make_shared<CudaBuffer>(sizeof(ValueIndexPair));
}

/** atomicMax from hw3:
 __device__ static float atomicMax(float* address, float val)
 {
     int* address_as_i = (int*) address;
     int old = *address_as_i, assumed;
     do {
         assumed = old;
         old = ::atomicCAS(address_as_i, assumed,
             __float_as_int(::fmaxf(val, __int_as_float(assumed))));
     } while (assumed != old);
     return __int_as_float(old);
 }
 */

__device__ static void atomicArgMax(ValueIndexPair *pair, ValueIndexPair other) {
    // ValuePairIndex is 32 bit float and 32 bit integer for a total of 64 bits,
    // so use 64 bit atomic CAS to achieve atomic arg max.
    unsigned long long* address = (unsigned long long*) pair;
    unsigned long long old = *address, assumed;
    do {
        assumed = old;
        ValueIndexPair best = pair_argmax(other, *(ValueIndexPair*)&assumed);
        old = ::atomicCAS(address, assumed, *(unsigned long long*)&best);
    } while (assumed != old);
}

__global__ void argmaxKernel(ValueIndexPair *data, ValueIndexPair *result, int n) {
    extern __shared__ ValueIndexPair partial_argmaxes[];

    int tidx = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    ValueIndexPair local_argmax = {-FLT_MAX, -1};
    for (int i = idx; i < n; i += stride) {
        ValueIndexPair input = data[i];
        local_argmax = pair_argmax(local_argmax, input);
    }
    partial_argmaxes[tidx] = local_argmax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tidx < s) {
            partial_argmaxes[tidx] = pair_argmax(partial_argmaxes[tidx], partial_argmaxes[tidx + s]);
        }
        __syncthreads();
    }

    if (tidx == 0) {
        atomicArgMax(result, partial_argmaxes[tidx]);
    }
}

int32_t *ArgMax::bf16_argmax(const std::shared_ptr<CudaBuffer> &bf16_data, cudaStream_t stream) {
    int n = bf16_data->size / sizeof(__nv_bfloat16);

    // Copy and cast bf16 data to temp.
    __nv_bfloat16* bf16_input = static_cast<__nv_bfloat16*>(bf16_data->data);
    ValueIndexPair* pairs = static_cast<ValueIndexPair*>(temp_space->data);
    for (int i = 0; i < n; i++) {
        pairs[i].val = __bfloat162float(bf16_input[i]);
        pairs[i].idx = i;
    }

    ValueIndexPair* result = static_cast<ValueIndexPair*>(result_space->data);
    result->val = -FLT_MAX;
    result->idx = -1;

    // Run kernel.
    int threads = ARGMAX_THREADS;
    int blocks = min((n + threads - 1) / threads, ARGMAX_MAX_BLOCKS);
    int shared_mem_size = threads * sizeof(ValueIndexPair);
    argmaxKernel<<<blocks, threads, shared_mem_size, stream>>>(pairs, result, n);
    checkCuda(cudaGetLastError());

    return &result->idx;
}
