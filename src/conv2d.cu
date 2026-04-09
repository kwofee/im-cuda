// conv2d.cu
// Optimized 2D Convolution Kernel
// — Tiled execution with shared memory for input data reuse
// — Constant memory for the convolution kernel (broadcast to all threads)

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

#define TILE_SIZE 16
#define MAX_KERNEL_SIZE 7

// Convolution kernel in constant memory — cached & broadcast to all threads in a warp
__constant__ float c_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

// Each block computes a TILE_SIZE x TILE_SIZE output tile.
// Shared memory holds the input tile + halo: (TILE_SIZE + k-1)^2 elements.
// Threads cooperatively load the shared tile, then each thread computes one output pixel.
__global__ void BaseConv2D(float *output, float *input, int m, int n, int k) {
    extern __shared__ float s_input[];

    int half_k = k / 2;
    int tile_w = TILE_SIZE + k - 1;  // shared tile width (includes halo on both sides)

    // Output pixel this thread is responsible for
    int out_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_row = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Top-left corner of the shared tile in global input coordinates
    int in_start_col = (int)(blockIdx.x * TILE_SIZE) - half_k;
    int in_start_row = (int)(blockIdx.y * TILE_SIZE) - half_k;

    // ---- Cooperative load: tile + halo into shared memory ----
    // tile_w x tile_w elements loaded by TILE_SIZE x TILE_SIZE threads
    // Some threads load multiple elements via striding
    for (int i = threadIdx.y; i < tile_w; i += TILE_SIZE) {
        for (int j = threadIdx.x; j < tile_w; j += TILE_SIZE) {
            int gr = in_start_row + i;
            int gc = in_start_col + j;

            // Zero-padding for out-of-bounds pixels
            if (gr >= 0 && gr < m && gc >= 0 && gc < n) {
                s_input[i * tile_w + j] = input[gr * n + gc];
            } else {
                s_input[i * tile_w + j] = 0.0f;
            }
        }
    }

    __syncthreads();

    // ---- Compute convolution from shared memory ----
    if (out_row < m && out_col < n) {
        float sum = 0.0f;

        for (int ki = 0; ki < k; ki++) {
            for (int kj = 0; kj < k; kj++) {
                // threadIdx.{x,y} offsets into the shared tile give the correct
                // position because the halo starts at index 0 and the first
                // "real" output pixel for this block is at index half_k
                sum += s_input[(threadIdx.y + ki) * tile_w + (threadIdx.x + kj)]
                     * c_kernel[ki * k + kj];
            }
        }

        output[out_row * n + out_col] = sum;
    }
}

// Host wrapper — interface unchanged (input, kernel, output as host pointers)
void launchConv2D(float *input, float *kernel, float *output, int m, int n, int k) {
    if (k > MAX_KERNEL_SIZE) {
        printf("Error: kernel size %d exceeds MAX_KERNEL_SIZE %d\n", k, MAX_KERNEL_SIZE);
        return;
    }

    float *d_input, *d_output;
    size_t mat_size = m * n * sizeof(float);
    size_t ker_size = k * k * sizeof(float);

    // Copy convolution kernel to constant memory (no device pointer needed)
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, kernel, ker_size));

    // Allocate device memory for input and output only (kernel is in constant mem)
    CUDA_CHECK(cudaMalloc((void **)&d_input, mat_size));
    CUDA_CHECK(cudaMalloc((void **)&d_output, mat_size));

    CUDA_CHECK(cudaMemcpy(d_input, input, mat_size, cudaMemcpyHostToDevice));

    // 2D grid of TILE_SIZE x TILE_SIZE blocks
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((n + TILE_SIZE - 1) / TILE_SIZE,
                  (m + TILE_SIZE - 1) / TILE_SIZE);

    // Dynamic shared memory: (TILE_SIZE + k - 1)^2 floats for the input tile
    int tile_w = TILE_SIZE + k - 1;
    size_t sharedMemSize = tile_w * tile_w * sizeof(float);

    BaseConv2D<<<gridSize, blockSize, sharedMemSize>>>(d_output, d_input, m, n, k);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, d_output, mat_size, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
}