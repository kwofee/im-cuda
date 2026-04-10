giimport numpy as np
import my_extension

# 5x5 test matrix
m, n = 5, 5
input_mat = np.array([
    [1,  2,  3,  4,  5],
    [6,  7,  8,  9,  10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32)

# ---- Test 1: Raw conv2d with identity kernel (backward compat) ----
identity = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32)
out = np.zeros((m, n), dtype=np.float32)
my_extension.runConv2D(input_mat, identity, out, m, n, 3)
assert np.allclose(out, input_mat), "Raw conv2d FAILED"
print("=== Raw runConv2D (identity): PASSED ===\n")

# ---- Test 2: Conv2D base class with custom kernel ----
conv = my_extension.Conv2D([0,0,0, 0,1,0, 0,0,0], 3)
out = np.zeros((m, n), dtype=np.float32)
conv.apply(input_mat, out, m, n)
assert np.allclose(out, input_mat), "Conv2D base class FAILED"
print("=== Conv2D base (identity): PASSED ===\n")

# ---- Test 3: SobelFilter ----
sobel_x = my_extension.SobelFilter("x")
out_sx = np.zeros((m, n), dtype=np.float32)
sobel_x.apply(input_mat, out_sx, m, n)
print(f"=== SobelFilter('x') ===\n{out_sx}\n")

sobel_y = my_extension.SobelFilter("y")
out_sy = np.zeros((m, n), dtype=np.float32)
sobel_y.apply(input_mat, out_sy, m, n)
print(f"=== SobelFilter('y') ===\n{out_sy}\n")

# ---- Test 4: ScharrFilter ----
scharr_x = my_extension.ScharrFilter("x")
out_schx = np.zeros((m, n), dtype=np.float32)
scharr_x.apply(input_mat, out_schx, m, n)
print(f"=== ScharrFilter('x') ===\n{out_schx}\n")

# ---- Test 5: LaplacianFilter ----
lap4 = my_extension.LaplacianFilter(4)
out_l4 = np.zeros((m, n), dtype=np.float32)
lap4.apply(input_mat, out_l4, m, n)
# Interior pixels of linear gradient should be 0
assert abs(out_l4[2, 2]) < 0.01, "Laplacian-4 interior FAILED"
print(f"=== LaplacianFilter(4) ===\n{out_l4}")
print("Interior zero check: PASSED\n")

lap8 = my_extension.LaplacianFilter(8)
out_l8 = np.zeros((m, n), dtype=np.float32)
lap8.apply(input_mat, out_l8, m, n)
assert abs(out_l8[2, 2]) < 0.01, "Laplacian-8 interior FAILED"
print(f"=== LaplacianFilter(8) ===\n{out_l8}")
print("Interior zero check: PASSED\n")

# ---- Test 6: GaussianBlur ----
gauss3 = my_extension.GaussianBlur(3, 1.0)
out_g3 = np.zeros((m, n), dtype=np.float32)
gauss3.apply(input_mat, out_g3, m, n)
print(f"=== GaussianBlur(3, sigma=1.0) ===\n{out_g3}\n")

gauss5 = my_extension.GaussianBlur(5, 1.4)
out_g5 = np.zeros((m, n), dtype=np.float32)
gauss5.apply(input_mat, out_g5, m, n)
print(f"=== GaussianBlur(5, sigma=1.4) ===\n{out_g5}\n")

# Gaussian output should be smoothed (center closer to average of neighbors)
assert abs(out_g3[2,2] - 13.0) < 1.0, "Gaussian center value unexpected"
print("=== All tests PASSED! ===")
