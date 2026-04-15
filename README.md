# CUDA 2D Convolution & Image Processing Filters

A high-performance image processing library utilizing CUDA for parallel computation. It is exposed to Python via `pybind11` under the module `my_extension`.

## Hierarchy of Classes

The library provides an object-oriented architecture. It includes a base class for generic 2D convolutions, several derived filter classes for common image operations, and specialized classes for non-convolutional operations.

### Convolution Filters
All convolution-based filters derive from the base `Conv2D` class.

*   **`Conv2D(kernel, k)`**: The base class for applying a custom $k \times k$ convolution. It takes a flattened (1D list) custom `kernel` and its dimension `k`.
*   **`SobelFilter(direction)`**: Derives from `Conv2D`. A Sobel edge detector. `direction` must be `'x'` (horizontal) or `'y'` (vertical).
*   **`ScharrFilter(direction)`**: Derives from `Conv2D`. A Scharr edge detector which offers better rotational symmetry than Sobel. `direction` must be `'x'` or `'y'`.
*   **`LaplacianFilter(connectivity)`**: Derives from `Conv2D`. A Laplacian edge detector. `connectivity` can be `4` (default) or `8`.
*   **`GaussianBlur(size, sigma)`**: Derives from `Conv2D`. A Gaussian blur filter. `size` must be an odd kernel size (e.g., 3, 5, 7) and `sigma` is the standard deviation.

### Specialized Image Processing Filters
These operate independently of the `Conv2D` class and define specific CUDA operations.

*   **`OtsuBinarizer()`**: Applies Otsu's thresholding method to automatically calculate a threshold and binarize the image.
*   **`RGB2HSVConverter()`**: Converts an RGB image into the HSV color space.

## How to Apply and Use

All classes implement an `apply` method, changing how the particular CUDA kernel is invoked while giving a consistent caller API. The method processes an input NumPy array and stores the result in a pre-allocated output array. 

### General Usage Pattern
1.  Import `numpy` and `my_extension`.
2.  Prepare an input NumPy array (`dtype=np.float32`).
3.  Pre-allocate an output NumPy array of the same shape and type.
4.  Instantiate the desired filter (e.g., `my_extension.SobelFilter('x')`).
5.  Call `.apply(input, output, m, n)` where `m` and `n` represent the image dimensions (rows and columns).

### Usage Example

```python
import numpy as np
import my_extension

# Prepare an m x n input image and an output buffer
m, n = 100, 100
input_image = np.random.rand(m, n).astype(np.float32)

# ==========================================
# 1. Using a custom Conv2D kernel
# ==========================================
k_size = 3
custom_kernel = [0, -1, 0, -1, 4, -1, 0, -1, 0] # 3x3 Flattened
conv = my_extension.Conv2D(custom_kernel, k_size)

out_custom = np.zeros((m, n), dtype=np.float32)
conv.apply(input_image, out_custom, m, n)

# ==========================================
# 2. Using Sobel Filter (Derived)
# ==========================================
sobel_x = my_extension.SobelFilter('x')

out_sobel = np.zeros((m, n), dtype=np.float32)
sobel_x.apply(input_image, out_sobel, m, n)

# ==========================================
# 3. Using Gaussian Blur (Derived)
# ==========================================
gaussian = my_extension.GaussianBlur(size=5, sigma=1.4)

out_gauss = np.zeros((m, n), dtype=np.float32)
gaussian.apply(input_image, out_gauss, m, n)

# ==========================================
# 4. Otsu's Binarization (Independent)
# ==========================================
otsu = my_extension.OtsuBinarizer()

out_otsu = np.zeros((m, n), dtype=np.float32)
otsu.apply(input_image, out_otsu, m, n)

# ==========================================
# 5. RGB to HSV Conversion (Independent)
# ==========================================
# Input should represent an m x n RGB image
rgb_image = np.random.rand(m, n, 3).astype(np.float32)
hsv_image = np.zeros((m, n, 3), dtype=np.float32)

rgb2hsv = my_extension.RGB2HSVConverter()
rgb2hsv.apply(rgb_image, hsv_image, m, n)
```

## Free Functions exposed below module
The Python module also directly exposes these standalone CUDA invocation routines:
*   `runConv2D(input, kernel, output, m, n, k)`: Runs a 2D convolution directly utilizing a raw kernel matrix without creating filter objects.
*   `runKernel(A, B, C, n)`: Runs a basic vector addition fallback CUDA test.