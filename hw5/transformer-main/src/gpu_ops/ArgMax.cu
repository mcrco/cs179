#include "ArgMax.cuh"
#include <cuda_bf16.h>
#include "../CudaBuffer.cuh"
#include <memory>
#include "../ErrorCheck.h"
#include <float.h>

struct ValueIndexPair {
    float val;
    int idx;
};

ArgMax::ArgMax(int32_t len) {
    temp_space = std::make_shared<CudaBuffer>(len * sizeof(ValueIndexPair));
}

__device__ static void atomicArgMax(ValueIndexPair *address, ValueIndexPair target) {
    // ValuePairIndex is 32 bit float and 32 bit integer for a total of 64 bits,
    // so use 64 bit atomic CAS to achieve atomic arg max.

}

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
    // TODO
    return nullptr;
}
