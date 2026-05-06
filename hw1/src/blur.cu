#include "blur.cuh"
#include "ErrorCheck.cuh"
#include <cstdio>
#include <cuda_runtime.h>


__device__
void cuda_blur_kernel_convolution(uint raw_data_index, const float* gpu_raw_data,
                                  const float* gpu_blur_v, float* gpu_out_data,
                                  const unsigned int n_frames,
                                  const unsigned int blur_v_size) {
    gpu_out_data[raw_data_index] = 0;

    for (uint i = 0; i < blur_v_size; i++) {
        if (i <= raw_data_index) {
            gpu_out_data[raw_data_index] += gpu_raw_data[raw_data_index - i] * gpu_blur_v[i];
        }
    }
}

__global__
void cuda_blur_kernel(const float *gpu_raw_data, const float *gpu_blur_v,
                      float *gpu_out_data, int n_frames, int blur_v_size) {
    uint raw_data_index = blockIdx.x * blockDim.x + threadIdx.x;
    while (raw_data_index < n_frames) {
        cuda_blur_kernel_convolution(raw_data_index, gpu_raw_data,
                                     gpu_blur_v, gpu_out_data,
                                     n_frames, blur_v_size);
        raw_data_index += blockDim.x * gridDim.x;
    }
}


float cuda_call_blur_kernel(const unsigned int blocks,
                            const unsigned int threads_per_block,
                            const float *raw_data,
                            const float *blur_v,
                            float *out_data,
                            const unsigned int n_frames,
                            const unsigned int blur_v_size) {
    // Use the CUDA machinery for recording time
    cudaEvent_t start_gpu, stop_gpu;
    float time_milli = -1;
    checkCuda(cudaEventCreate(&start_gpu));
    checkCuda(cudaEventCreate(&stop_gpu));
    checkCuda(cudaEventRecord(start_gpu));

    float* gpu_raw_data;
    size_t data_size = n_frames * sizeof (float);
    checkCuda(cudaMalloc((void**) &gpu_raw_data, data_size));
    checkCuda(cudaMemcpy(gpu_raw_data, raw_data, data_size, cudaMemcpyHostToDevice));

    float* gpu_blur_v;
    size_t blur_size = blur_v_size * sizeof (float);
    checkCuda(cudaMalloc((void**) &gpu_blur_v, blur_size));
    checkCuda(cudaMemcpy(gpu_blur_v, blur_v, blur_size, cudaMemcpyHostToDevice));

    float* gpu_out_data;
    checkCuda(cudaMalloc((void**) &gpu_out_data, data_size));
    
    cuda_blur_kernel<<<blocks, threads_per_block>>>(gpu_raw_data, gpu_blur_v, gpu_out_data, n_frames, blur_v_size);

    // Check for errors on kernel call
    // Always include an error check after every kernel call
    checkCuda(cudaGetLastError());

    checkCuda(cudaMemcpy(out_data, gpu_out_data, data_size, cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(gpu_raw_data));
    checkCuda(cudaFree(gpu_out_data));
    checkCuda(cudaFree(gpu_blur_v));

    // Stop the recording timer and return the computation time
    checkCuda(cudaEventRecord(stop_gpu));
    checkCuda(cudaEventSynchronize(stop_gpu));
    checkCuda(cudaEventElapsedTime(&time_milli, start_gpu, stop_gpu));
    return time_milli;
}
