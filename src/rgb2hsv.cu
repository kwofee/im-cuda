#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include "rgb2hsv.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

__global__ void RGB2HSVKernel(float *input, float *output, int pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pixels) {
        float r = input[idx * 3 + 0] / 255.0f;
        float g = input[idx * 3 + 1] / 255.0f;
        float b = input[idx * 3 + 2] / 255.0f;

        float cmax = fmaxf(r, fmaxf(g, b));
        float cmin = fminf(r, fminf(g, b));
        float delta = cmax - cmin;

        float h = 0.0f;
        if (delta == 0.0f) {
            h = 0.0f;
        } else if (cmax == r) {
            h = 60.0f * fmodf(((g - b) / delta), 6.0f);
        } else if (cmax == g) {
            h = 60.0f * (((b - r) / delta) + 2.0f);
        } else if (cmax == b) {
            h = 60.0f * (((r - g) / delta) + 4.0f);
        }

        if (h < 0.0f) {
            h += 360.0f;
        }

        float s = (cmax == 0.0f) ? 0.0f : (delta / cmax);
        float v = cmax;

        output[idx * 3 + 0] = h;
        output[idx * 3 + 1] = s;
        output[idx * 3 + 2] = v;
    }
}

void RGB2HSVConverter::apply(float *input, float *output, int m, int n) {
    int pixels = m * n;
    size_t img_bytes = pixels * 3 * sizeof(float);
    float *d_input, *d_output;

    CUDA_CHECK(cudaMalloc((void **)&d_input, img_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_output, img_bytes));

    CUDA_CHECK(cudaMemcpy(d_input, input, img_bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (pixels + blockSize - 1) / blockSize;

    RGB2HSVKernel<<<gridSize, blockSize>>>(d_input, d_output, pixels);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, d_output, img_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
}
