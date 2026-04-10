import os
import shutil
import subprocess
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def find_nvcc_and_cuda_home():
    cuda_home = os.environ.get("CUDA_HOME")
    candidates = []

    if cuda_home:
        candidates.append(os.path.join(cuda_home, "bin", "nvcc"))

    nvcc_on_path = shutil.which("nvcc")
    if nvcc_on_path:
        candidates.append(nvcc_on_path)

    candidates.extend([
        "/usr/local/cuda/bin/nvcc",
        "/usr/bin/nvcc",
    ])

    for nvcc in candidates:
        if nvcc and os.path.exists(nvcc):
            return nvcc, os.path.dirname(os.path.dirname(nvcc))

    raise RuntimeError(
        "Could not find nvcc. Install CUDA toolkit and/or set CUDA_HOME to your CUDA root."
    )


def get_include_dirs(cuda_home):
    try:
        import pybind11
    except ImportError as exc:
        raise RuntimeError("pybind11 is required. Install with: pip install pybind11>=2.12") from exc

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required. Install with: pip install numpy") from exc

    python_include = sysconfig.get_paths()["include"]

    return [
        pybind11.get_include(),
        np.get_include(),
        python_include,
        os.path.join(cuda_home, "include"),
        "src",
    ]


def compile_cuda(sources, build_dir, nvcc, include_dirs):
    os.makedirs(build_dir, exist_ok=True)
    objects = []
    for src in sources:
        obj = os.path.join(build_dir, os.path.basename(src) + ".o")
        cmd = [
            nvcc,
            "-c", src,
            "-o", obj,
            "-O2",
            "--compiler-options", "-fPIC",
            "-std=c++17",
        ]

        for inc in include_dirs:
            cmd.extend(["-I", inc])

        print(f"Compiling CUDA: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        objects.append(obj)
    return objects


class CUDABuildExt(build_ext):
    def build_extension(self, ext):
        nvcc, cuda_home = find_nvcc_and_cuda_home()
        include_dirs = get_include_dirs(cuda_home)

        cuda_sources = [s for s in ext.sources if s.endswith(".cu")]
        ext.sources = [s for s in ext.sources if not s.endswith(".cu")]

        build_dir = os.path.join(self.build_temp, "cuda_objects")
        cuda_objects = compile_cuda(cuda_sources, build_dir, nvcc, include_dirs)

        ext.extra_objects = getattr(ext, "extra_objects", []) + cuda_objects
        ext.include_dirs = list(dict.fromkeys((ext.include_dirs or []) + include_dirs))

        lib_dirs = [os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "targets", "x86_64-linux", "lib")]
        ext.library_dirs += [d for d in lib_dirs if os.path.exists(d)]
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
    include_dirs=[],
    extra_compile_args=["-O2", "-std=c++17"],
)

setup(
    name="my_extension",
    ext_modules=[ext],
    cmdclass={"build_ext": CUDABuildExt},
)