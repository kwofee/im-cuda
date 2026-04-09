import numpy as np
import my_extension

# ------ Test 1: Identity kernel (output should equal input) ------
m, n = 5, 5
input_mat = np.array([
    [1,  2,  3,  4,  5],
    [6,  7,  8,  9,  10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32)

identity_kernel = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=np.float32)

output = np.zeros((m, n), dtype=np.float32)
my_extension.runConv2D(input_mat, identity_kernel, output, m, n, 3)
print("=== Identity Kernel (should match input) ===")
print(output)
assert np.allclose(output, input_mat), "Identity kernel test FAILED!"
print("PASSED\n")

# ------ Test 2: Box blur (3x3 averaging) ------
box_kernel = np.ones((3, 3), dtype=np.float32) / 9.0
output_blur = np.zeros((m, n), dtype=np.float32)
my_extension.runConv2D(input_mat, box_kernel, output_blur, m, n, 3)
print("=== Box Blur (3x3 average) ===")
print(output_blur)
# Center pixel (2,2) = average of all 9 neighbors of 13 = (7+8+9+12+13+14+17+18+19)/9 = 13.0
assert abs(output_blur[2, 2] - 13.0) < 0.01, "Box blur center test FAILED!"
print("PASSED\n")

# ------ Test 3: Laplacian edge detection ------
laplacian_kernel = np.array([
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
], dtype=np.float32)

output_edge = np.zeros((m, n), dtype=np.float32)
my_extension.runConv2D(input_mat, laplacian_kernel, output_edge, m, n, 3)
print("=== Laplacian Edge Detection ===")
print(output_edge)
# Interior pixels of a linearly varying image should give 0
assert abs(output_edge[2, 2]) < 0.01, "Laplacian interior test FAILED!"
print("PASSED\n")

# ------ Test 4: Sobel X (horizontal edges) ------
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

output_sobel = np.zeros((m, n), dtype=np.float32)
my_extension.runConv2D(input_mat, sobel_x, output_sobel, m, n, 3)
print("=== Sobel X (horizontal gradient) ===")
print(output_sobel)
print("DONE — all tests passed!")
