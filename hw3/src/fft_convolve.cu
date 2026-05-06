/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve.cuh"


/*
Atomic-max function. You may find it useful for normalization.

Source:
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
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



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v,
    cufftComplex *out_data,
    int padded_length) {

    // Multiply all values by scaling factor since cufft returns values * padded_length.
    float scale = 1.0f / (float) padded_length;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Need stride since we might not have num_threads >= padded_length.
    int stride = blockDim.x * gridDim.x;
    for (; idx < padded_length; idx += stride) {
        if (idx < padded_length) {
            cufftComplex prod = cuCmulf(raw_data[idx], impulse_v[idx]);
            prod.x *= scale;
            prod.y *= scale;
            out_data[idx] = prod;
        }
    }
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    // Use shared accumulation per block since doing maxes among shared memory
    // is much faster than many reads/writes from global memory.
    extern __shared__ float partial_maxes[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Go through array and get max magnitudes for each thread with coalesced read.
    int stride = blockDim.x * gridDim.x;
    float max_abs_val_idx = 0.0f;
    for (int i = idx; i < padded_length; i += stride) {
        if (i < padded_length) {
            cufftComplex val = out_data[i];
            max_abs_val_idx = fmaxf(max_abs_val_idx, fabs(val.x));
        }
    }
    int tidx = threadIdx.x;
    partial_maxes[tidx] = max_abs_val_idx;

    // Make sure all threads have collected their maximum values.
    __syncthreads();

    // Perform the reduction.
    for (int upperbound = blockDim.x / 2; upperbound > 0; upperbound >>= 1) {
        if (tidx < upperbound) {
            partial_maxes[tidx] = fmaxf(partial_maxes[tidx], partial_maxes[tidx + upperbound]);
        }
        // Need sync here since bound check will diverge.
        __syncthreads();
    }

    // All "first threads" of each block should contain max for block.
    if (tidx == 0) {
        atomicMax(max_abs_val, partial_maxes[0]);
    }
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float max_allowed_amp = 0.99999; // match cpu implementation.
    // Go through array and get max magnitudes for each thread with coalesced read.
    // No point in doing anything else since this is as simple as it gets.
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < padded_length; i += stride) {
        if (i < padded_length) {
            cufftComplex val = out_data[i];
            val.x = val.x / *max_abs_val * max_allowed_amp;
            val.y = val.y / *max_abs_val * max_allowed_amp;
            out_data[i] = val;
        }
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v, out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
    cudaMaximumKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(out_data, max_abs_val, padded_length);
}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val, padded_length);
}
