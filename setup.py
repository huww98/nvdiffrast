# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import nvdiffrast
import setuptools
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nvdiffrast",
    version=nvdiffrast.__version__,
    author="Samuli Laine",
    author_email="slaine@nvidia.com",
    description="nvdiffrast - modular primitives for high-performance differentiable rendering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NVlabs/nvdiffrast",
    packages=setuptools.find_packages(exclude=["nvdiffrast.tensorflow"]),
    ext_modules=[
        CUDAExtension(
            name='nvdiffrast_plugin',
            sources=[
                'nvdiffrast/common/common.cpp',
                'nvdiffrast/common/glutil.cpp',
                'nvdiffrast/common/rasterize_kernel.cu',
                'nvdiffrast/common/rasterize.cpp',
                'nvdiffrast/common/interpolate_kernel.cu',
                'nvdiffrast/common/texture_kernel.cu',
                'nvdiffrast/common/texture.cpp',
                'nvdiffrast/common/antialias.cu',
                'nvdiffrast/torch/torch_bindings.cpp',
                'nvdiffrast/torch/torch_rasterize.cpp',
                'nvdiffrast/torch/torch_interpolate.cpp',
                'nvdiffrast/torch/torch_texture.cpp',
                'nvdiffrast/torch/torch_antialias.cpp',
            ],
            libraries=['GL', 'EGL'],
            define_macros=[('NVDR_TORCH', None)],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['numpy'],  # note: can't require torch here as it will install torch even for a TensorFlow container
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
