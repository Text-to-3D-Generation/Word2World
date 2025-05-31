

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
            "gaussianRasterizer.cu",
            "gaussianForwardPass.cu",
            "backward.cu",
            "rasterize_points.cu",
            "projection.cu",
            "ADRculling.cu",
            "pixelGaussiansFull.cu",
            "gaussianTiles.cu",
            "prepareSort.cu",
            #"colCal.cu",
            "gradientPass.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "GLMLibrary/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
