

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "base.h"
#include "conv2d.h"

namespace py = pybind11;

void runKernel(py::array_t<float>& A, py::array_t<float>& B, py::array_t<float>& C, int n) {
    auto bufA = A.request();
    auto bufB = B.request();
    auto bufC = C.request();

    float* ptrA = static_cast<float*>(bufA.ptr);
    float* ptrB = static_cast<float*>(bufB.ptr);
    float* ptrC = static_cast<float*>(bufC.ptr);

    launchKernel(ptrA, ptrB, ptrC, n);
}

// 2D Convolution binding
// input:  m x n numpy array (float32)
// kernel: k x k numpy array (float32)
// output: m x n numpy array (float32), pre-allocated
void runConv2D(py::array_t<float>& input, py::array_t<float>& kernel,
               py::array_t<float>& output, int m, int n, int k) {
    auto bufInput  = input.request();
    auto bufKernel = kernel.request();
    auto bufOutput = output.request();

    float* ptrInput  = static_cast<float*>(bufInput.ptr);
    float* ptrKernel = static_cast<float*>(bufKernel.ptr);
    float* ptrOutput = static_cast<float*>(bufOutput.ptr);

    launchConv2D(ptrInput, ptrKernel, ptrOutput, m, n, k);
}

PYBIND11_MODULE(my_extension, m) {
    m.def("runKernel", &runKernel, "Runs the vector addition CUDA kernel");
    m.def("runConv2D", &runConv2D, "Runs a 2D convolution with a given kernel matrix");
}