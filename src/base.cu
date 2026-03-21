// base.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

__global__ void basicKernel(float *A, float *B, float *C, int n){
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid < n){
        C[tid] = A[tid] + B[tid];
    }
}

void launchKernel(float *A, float *B, float *C, int n){
    float *d_A, *d_B, *d_C;
    size_t size = n*sizeof(float);

    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256, blocks = (n + threadsPerBlock - 1)/threadsPerBlock;
    basicKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}