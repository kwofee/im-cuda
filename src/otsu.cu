#include <stdio.h>
#include <cuda_runtime.h>
#include "otsu.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

#define HIST_BINS 256

__global__ void HistogramKernel(float *input, unsigned int *hist, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory histogram for faster atomic adds
    __shared__ unsigned int s_hist[HIST_BINS];
    
    if (threadIdx.x < HIST_BINS) {
        s_hist[threadIdx.x] = 0;
    }
    __syncthreads();
    
    if (idx < size) {
        int val = (int)(input[idx]);
        if (val < 0) val = 0;
        if (val >= HIST_BINS) val = HIST_BINS - 1;
        atomicAdd(&s_hist[val], 1);
    }
    
    __syncthreads();
    
    // Write shared memory histogram to global memory
    if (threadIdx.x < HIST_BINS) {
        atomicAdd(&hist[threadIdx.x], s_hist[threadIdx.x]);
    }
}

__global__ void BinarizeKernel(float *input, float *output, int size, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] >= threshold) ? 255.0f : 0.0f;
    }
}

void OtsuBinarizer::apply(float *input, float *output, int m, int n) {
    int size = m * n;
    float *d_input, *d_output;
    unsigned int *d_hist;
    size_t img_bytes = size * sizeof(float);
    size_t hist_bytes = HIST_BINS * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc((void **)&d_input, img_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_output, img_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_hist, hist_bytes));

    CUDA_CHECK(cudaMemcpy(d_input, input, img_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, hist_bytes));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    HistogramKernel<<<gridSize, blockSize>>>(d_input, d_hist, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned int h_hist[HIST_BINS];
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, hist_bytes, cudaMemcpyDeviceToHost));

    // Calculate Otsu threshold on host
    float sum = 0;
    for (int i = 0; i < HIST_BINS; ++i) sum += i * h_hist[i];

    float sumB = 0;
    int wB = 0;
    int wF = 0;
    float varMax = 0;
    float threshold = 0;

    for (int i = 0; i < HIST_BINS; ++i) {
        wB += h_hist[i];
        if (wB == 0) continue;
        wF = size - wB;
        if (wF == 0) break;

        sumB += (float)(i * h_hist[i]);
        float mB = sumB / wB;
        float mF = (sum - sumB) / wF;
        float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);

        if (varBetween > varMax) {
            varMax = varBetween;
            threshold = (float)i;
        }
    }

    // Launch binarization
    BinarizeKernel<<<gridSize, blockSize>>>(d_input, d_output, size, threshold);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, d_output, img_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_hist);
}
