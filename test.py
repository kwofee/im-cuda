import numpy as np
import my_extension

n = 1024
A = np.ones(n, dtype=np.float32)
B = np.ones(n, dtype=np.float32) * 2
C = np.zeros(n, dtype=np.float32)  # output array

# call the kernel
my_extension.runKernel(A, B, C, n)

print(C)        # should print [3. 3. 3. ... 3.]
print(C[0]) 