
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "base.h"
#include "conv2d.h"

namespace py = pybind11;

// ---- Legacy vector addition binding ----
void runKernel(py::array_t<float>& A, py::array_t<float>& B, py::array_t<float>& C, int n) {
    auto bufA = A.request();
    auto bufB = B.request();
    auto bufC = C.request();

    float* ptrA = static_cast<float*>(bufA.ptr);
    float* ptrB = static_cast<float*>(bufB.ptr);
    float* ptrC = static_cast<float*>(bufC.ptr);

    launchKernel(ptrA, ptrB, ptrC, n);
}

// ---- Raw conv2d binding (pass your own kernel) ----
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

// ---- Helper: call .apply() from Python with numpy arrays ----
void applyFilter(Conv2D& filter, py::array_t<float>& input,
                 py::array_t<float>& output, int m, int n) {
    auto bufIn  = input.request();
    auto bufOut = output.request();
    filter.apply(static_cast<float*>(bufIn.ptr),
                 static_cast<float*>(bufOut.ptr), m, n);
}

PYBIND11_MODULE(my_extension, m) {
    m.doc() = "CUDA-accelerated 2D convolution filters";

    // Free functions
    m.def("runKernel", &runKernel, "Runs the vector addition CUDA kernel");
    m.def("runConv2D", &runConv2D, "Runs a 2D convolution with a given kernel matrix");

    // Base class
    py::class_<Conv2D>(m, "Conv2D")
        .def(py::init<const std::vector<float>&, int>(),
             py::arg("kernel"), py::arg("k"),
             "Create a Conv2D filter with a custom k x k kernel")
        .def("apply", &applyFilter,
             py::arg("input"), py::arg("output"), py::arg("m"), py::arg("n"),
             "Apply the convolution to an m x n float32 array")
        .def("get_kernel_size", &Conv2D::getKernelSize);

    // Sobel
    py::class_<SobelFilter, Conv2D>(m, "SobelFilter")
        .def(py::init([](const std::string& dir) {
            if (dir == "x" || dir == "X") return SobelFilter(SobelFilter::X);
            if (dir == "y" || dir == "Y") return SobelFilter(SobelFilter::Y);
            throw std::invalid_argument("Direction must be 'x' or 'y'");
        }), py::arg("direction"),
        "Sobel edge detector. direction: 'x' (horizontal) or 'y' (vertical)");

    // Scharr
    py::class_<ScharrFilter, Conv2D>(m, "ScharrFilter")
        .def(py::init([](const std::string& dir) {
            if (dir == "x" || dir == "X") return ScharrFilter(ScharrFilter::X);
            if (dir == "y" || dir == "Y") return ScharrFilter(ScharrFilter::Y);
            throw std::invalid_argument("Direction must be 'x' or 'y'");
        }), py::arg("direction"),
        "Scharr edge detector. direction: 'x' or 'y'");

    // Laplacian
    py::class_<LaplacianFilter, Conv2D>(m, "LaplacianFilter")
        .def(py::init<int>(), py::arg("connectivity") = 4,
             "Laplacian edge detector. connectivity: 4 (default) or 8");

    // Gaussian Blur
    py::class_<GaussianBlur, Conv2D>(m, "GaussianBlur")
        .def(py::init<int, float>(), py::arg("size"), py::arg("sigma"),
             "Gaussian blur. size: odd kernel size (3,5,7). sigma: std deviation");
}