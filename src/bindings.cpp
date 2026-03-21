

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "base.h"

namespace py = pybind11;

void runKernel(py::array_t<float>& A, py::array_t<float>& B, py::array_t<float>& C, int n) {
    auto bufA = A.request();
    auto bufB = B.request();
    auto bufC = C.request();

    float* ptrA = static_cast<float*>(bufA.ptr);
    float* ptrB = static_cast<float*>(bufB.ptr);
    float* ptrC = static_cast<float*>(bufC.ptr);

    launchKernel(ptrA, ptrB, ptrC, n);  // ← semicolon + pass the pointers
}

PYBIND11_MODULE(my_extension, m) {
    m.def("runKernel", &runKernel, "Runs the CUDA kernel");
}