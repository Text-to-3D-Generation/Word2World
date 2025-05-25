

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/gaussianRasterizer.cu",
            "cuda_rasterizer/gaussianForwardPass.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "cuda_rasterizer/projection.cu",
            "cuda_rasterizer/ADRculling.cu",
            "cuda_rasterizer/pixelGaussiansFull.cu",
            "cuda_rasterizer/gaussianTiles.cu",
            "cuda_rasterizer/prepareSort.cu",
            #"cuda_rasterizer/colCal.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "GLMLibrary/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
