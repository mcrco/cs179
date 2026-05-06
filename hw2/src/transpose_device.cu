#include <cassert>
#include <cuda_runtime.h>
#include <stdexcept>
#include "transpose_device.cuh"
#include "ErrorCheck.cuh"

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */


/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304    matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    // Output write is not coalesced because consecutive threads write
    // with a stride of n. Since n is a multiple of 64, there are at
    // least 64 floats between strides => 256 bytes between strides > 128
    // between strides, so each warp accesses 32 cache lines.
    for (; j < end_j; j++)
        output[j + n * i] = input[i + n * j];
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    // Create shared memory for 64 x 64 patch, but pad columns to prevent 
    // bank conflict (pointed out later).
    __shared__ float patch[64][65];

    // Get input indices.
    int ix = blockIdx.x * 64 + threadIdx.x;
    int iy = blockIdx.y * 64 + 4 * threadIdx.y; // 4 * threadIdx.y because we load 4 floats per thread.

    // For each of input[iy + i][ix], load into the shared memory.
    for (int i = 0; i < 4; i++) {
        // This is a coalesced read because at each step, ix for adjacent 
        // threads differs by 1, and iy is the same.
        patch[4 * threadIdx.y + i][threadIdx.x] = input[ix + (iy + i) * n];
    }

    // Wait for block to load entire patch.
    __syncthreads();

    // Now, patch should contain the original input in the same orientation 
    // but in just a 64 x 64 array in shared memory.

    // Transpose patch and write to output by inverting blockIdx, and then
    // inverting our coordinates for the patch.
    int ox = blockIdx.y * 64 + threadIdx.x;
    int oy = blockIdx.x * 64 + 4 * threadIdx.y;
    for (int i = 0; i < 4; i++) {
        // This is where the bank conflict would have happened if we didn't
        // pad the patch cols to 65 because thread t + 1 would have accessed
        // thread t's flattened shared memory index + 64, which is the same mod 
        // 32. By making patch have 65 columns, thread t + 1 accesses thread t's
        // flattend index + 65 = index + 1 mod 32.
        output[ox + (oy + i) * n] = patch[threadIdx.x][4 * threadIdx.y + i];
    }

    // As far as I can tell, no suboptimal accesses besides the loop, which
    // we can unroll? All of the input/output writes are caolesced, and we prevent
    // a 32-way bank conflict by padding patch.
}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    __shared__ float patch[64][65];

    int ix = blockIdx.x * 64 + threadIdx.x;
    int iy = blockIdx.y * 64 + 4 * threadIdx.y;

    // Unrolled loop + parallelized global memory access.
    float p0 = input[ix + (iy + 0) * n];
    float p1 = input[ix + (iy + 1) * n];
    float p2 = input[ix + (iy + 2) * n];
    float p3 = input[ix + (iy + 3) * n];
    patch[4 * threadIdx.y + 0][threadIdx.x] = p0;
    patch[4 * threadIdx.y + 1][threadIdx.x] = p1;
    patch[4 * threadIdx.y + 2][threadIdx.x] = p2;
    patch[4 * threadIdx.y + 3][threadIdx.x] = p3;

    __syncthreads();

    int ox = blockIdx.y * 64 + threadIdx.x;
    int oy = blockIdx.x * 64 + 4 * threadIdx.y;

    // Unrolled loop
    output[ox + (oy + 0) * n] = patch[threadIdx.x][4 * threadIdx.y + 0];
    output[ox + (oy + 1) * n] = patch[threadIdx.x][4 * threadIdx.y + 1];
    output[ox + (oy + 2) * n] = patch[threadIdx.x][4 * threadIdx.y + 2];
    output[ox + (oy + 3) * n] = patch[threadIdx.x][4 * threadIdx.y + 3];
}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else {
        throw std::runtime_error("unknown transpose type");
    }
    checkCuda(cudaGetLastError());
}
