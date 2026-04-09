// conv2d.cu
// Base 2D Convolution Kernel
// One thread per output pixel, zero-padded borders
// Pass different kernel matrices for edge detection, gaussian smoothing, etc.

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

// Base 2D convolution kernel
// input:  m x n matrix (row-major)
// ker:    k x k convolution kernel (row-major), k should be odd
// output: m x n matrix (row-major)
// Zero-padding: out-of-bounds input pixels are treated as 0
__global__ void BaseConv2D(float *output, float *input, int m, int n, float *ker, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        float sum = 0.0f;
        int half_k = k / 2;

        for (int ki = 0; ki < k; ki++) {
            for (int kj = 0; kj < k; kj++) {
                int in_row = row - half_k + ki;
                int in_col = col - half_k + kj;

                // Zero-padding: skip out-of-bounds pixels
                if (in_row >= 0 && in_row < m && in_col >= 0 && in_col < n) {
                    sum += input[in_row * n + in_col] * ker[ki * k + kj];
                }
            }
        }

        output[row * n + col] = sum;
    }
}

// Host wrapper: allocates device memory, copies data, launches kernel, copies back
void launchConv2D(float *input, float *kernel, float *output, int m, int n, int k) {
    float *d_input, *d_kernel, *d_output;
    size_t mat_size = m * n * sizeof(float);
    size_t ker_size = k * k * sizeof(float);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_input, mat_size));
    CUDA_CHECK(cudaMalloc((void **)&d_kernel, ker_size));
    CUDA_CHECK(cudaMalloc((void **)&d_output, mat_size));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_input, input, mat_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel, ker_size, cudaMemcpyHostToDevice));

    // Launch with 2D grid: 16x16 threads per block
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                  (m + blockSize.y - 1) / blockSize.y);

    BaseConv2D<<<gridSize, blockSize>>>(d_output, d_input, m, n, d_kernel, k);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output, d_output, mat_size, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}