from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="spatial_variable_convolution",
    version="1.0.0",
    description="CUDA implementation of spatial variable convolution",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="svc._ext",
            sources=[
                "src/cuda/svc_cuda.cpp",
                "src/cuda/svc_cuda_kernel.cu",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=1.7.0",
        "einops>=0.4.0",
    ],
)
