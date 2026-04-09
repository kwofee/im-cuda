#pragma once

#include <vector>

// CUDA kernel launcher (defined in conv2d.cu)
void launchConv2D(float *input, float *kernel, float *output, int m, int n, int k);

// ---- Base class: wraps a convolution kernel and calls the CUDA launcher ----
class Conv2D {
protected:
    std::vector<float> kernel_;
    int kernel_size_;

public:
    Conv2D(const std::vector<float>& kernel, int k);
    virtual ~Conv2D() = default;

    void apply(float *input, float *output, int m, int n);
    int getKernelSize() const { return kernel_size_; }
    const std::vector<float>& getKernel() const { return kernel_; }
};

// ---- Sobel filter: 3x3 first-order edge detector ----
class SobelFilter : public Conv2D {
public:
    enum Direction { X, Y };
    SobelFilter(Direction dir);
};

// ---- Scharr filter: 3x3, more rotationally accurate than Sobel ----
class ScharrFilter : public Conv2D {
public:
    enum Direction { X, Y };
    ScharrFilter(Direction dir);
};

// ---- Laplacian: second-order edge detector ----
class LaplacianFilter : public Conv2D {
public:
    // connectivity: 4 (cross pattern) or 8 (full 3x3 neighborhood)
    LaplacianFilter(int connectivity = 4);
};

// ---- Gaussian blur: variable-size smoothing kernel ----
class GaussianBlur : public Conv2D {
public:
    // size: kernel side length (must be odd, e.g. 3, 5, 7)
    // sigma: standard deviation of the Gaussian
    GaussianBlur(int size, float sigma);
};
