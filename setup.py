import os
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

CUDA_HOME = "/usr"
PYTHON_HOME = "/home/laalenthika/anaconda3"
PYBIND11_INCLUDE = f"{PYTHON_HOME}/lib/python3.12/site-packages/pybind11/include"
PYTHON_INCLUDE = f"{PYTHON_HOME}/include/python3.12"


def compile_cuda(sources, build_dir):
    os.makedirs(build_dir, exist_ok=True)
    objects = []
    for src in sources:
        obj = os.path.join(build_dir, os.path.basename(src) + ".o")
        cmd = [
            f"{CUDA_HOME}/bin/nvcc",
            "-c", src,
            "-o", obj,
            "-O2",
            "--compiler-options", "-fPIC",
            "-std=c++17",
            "-I", f"{CUDA_HOME}/include",
            "-I", PYBIND11_INCLUDE,
            "-I", PYTHON_INCLUDE,
        ]
        print(f"Compiling CUDA: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        objects.append(obj)
    return objects


class CUDABuildExt(build_ext):
    def build_extension(self, ext):
        cuda_sources = [s for s in ext.sources if s.endswith(".cu")]
        ext.sources = [s for s in ext.sources if not s.endswith(".cu")]

        build_dir = os.path.join(self.build_temp, "cuda_objects")
        cuda_objects = compile_cuda(cuda_sources, build_dir)

        ext.extra_objects = getattr(ext, "extra_objects", []) + cuda_objects
        ext.library_dirs += [f"{CUDA_HOME}/lib64"]
        ext.libraries += ["cudart"]

        super().build_extension(ext)


ext = Extension(
    name="my_extension",
    sources=[
        "src/bindings.cpp",
        "src/filters.cpp",
        "src/base.cu",
        "src/conv2d.cu",
        "src/otsu.cu",
        "src/rgb2hsv.cu"
    ],
    include_dirs=[
        PYBIND11_INCLUDE,
        PYTHON_INCLUDE,
        f"{CUDA_HOME}/include",
        # "/usr/include/cuda",
        "src/",
    ],
    extra_compile_args=["-O2", "-std=c++17"],
)

setup(
    name="my_extension",
    ext_modules=[ext],
    cmdclass={"build_ext": CUDABuildExt},
)