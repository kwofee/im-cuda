// filters.cpp
// Implementations of Conv2D base class and all derived filter classes

#include "conv2d.h"
#include <cmath>
#include <stdexcept>

// ==================== Conv2D base ====================

Conv2D::Conv2D(const std::vector<float>& kernel, int k)
    : kernel_(kernel), kernel_size_(k) {
    if ((int)kernel.size() != k * k) {
        throw std::invalid_argument("Kernel vector size must equal k*k");
    }
}

void Conv2D::apply(float *input, float *output, int m, int n) {
    launchConv2D(input, kernel_.data(), output, m, n, kernel_size_);
}

// ==================== Sobel ====================
// Detects edges using first-order gradients
//
//  Sobel X:        Sobel Y:
//  -1  0  1       -1 -2 -1
//  -2  0  2        0  0  0
//  -1  0  1        1  2  1

SobelFilter::SobelFilter(Direction dir)
    : Conv2D(
        dir == X
            ? std::vector<float>{-1, 0, 1, -2, 0, 2, -1, 0, 1}
            : std::vector<float>{-1,-2,-1,  0, 0, 0,  1, 2, 1},
        3
    ) {}

// ==================== Scharr ====================
// More rotationally symmetric than Sobel
//
//  Scharr X:         Scharr Y:
//   -3   0   3       -3 -10  -3
//  -10   0  10        0   0   0
//   -3   0   3        3  10   3

ScharrFilter::ScharrFilter(Direction dir)
    : Conv2D(
        dir == X
            ? std::vector<float>{-3, 0, 3, -10, 0, 10, -3, 0, 3}
            : std::vector<float>{-3,-10,-3,  0, 0, 0,  3, 10, 3},
        3
    ) {}

// ==================== Laplacian ====================
// Second-order edge detector
//
//  4-connected:       8-connected:
//   0 -1  0           -1 -1 -1
//  -1  4 -1           -1  8 -1
//   0 -1  0           -1 -1 -1

LaplacianFilter::LaplacianFilter(int connectivity)
    : Conv2D(
        connectivity == 8
            ? std::vector<float>{-1,-1,-1, -1, 8,-1, -1,-1,-1}
            : std::vector<float>{ 0,-1, 0, -1, 4,-1,  0,-1, 0},
        3
    ) {
    if (connectivity != 4 && connectivity != 8) {
        throw std::invalid_argument("Laplacian connectivity must be 4 or 8");
    }
}

// ==================== Gaussian Blur ====================
// Generates a size x size Gaussian kernel with the given sigma
// G(x,y) = exp(-(x² + y²) / (2σ²))  then normalized to sum to 1

GaussianBlur::GaussianBlur(int size, float sigma)
    : Conv2D(std::vector<float>(size * size, 0.0f), size) {
    if (size % 2 == 0) {
        throw std::invalid_argument("Gaussian kernel size must be odd");
    }
    if (sigma <= 0) {
        throw std::invalid_argument("Sigma must be positive");
    }

    int half = size / 2;
    float sum = 0.0f;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float x = (float)(j - half);
            float y = (float)(i - half);
            float val = expf(-(x * x + y * y) / (2.0f * sigma * sigma));
            kernel_[i * size + j] = val;
            sum += val;
        }
    }

    // Normalize so all values sum to 1 (preserves brightness)
    for (int i = 0; i < size * size; i++) {
        kernel_[i] /= sum;
    }
}
