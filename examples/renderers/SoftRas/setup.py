# Copyright (c) 2017 Hiroharu Kato
# Copyright (c) 2018 Nikos Kolotouros
# Copyright (c) 2019 Shichen Liu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []

ext_modules=[
    CUDAExtension('soft_renderer.cuda.load_textures', [
        'soft_renderer/cuda/load_textures_cuda.cpp',
        'soft_renderer/cuda/load_textures_cuda_kernel.cu',
        ]),
    CUDAExtension('soft_renderer.cuda.create_texture_image', [
        'soft_renderer/cuda/create_texture_image_cuda.cpp',
        'soft_renderer/cuda/create_texture_image_cuda_kernel.cu',
        ]),
    CUDAExtension('soft_renderer.cuda.soft_rasterize', [
        'soft_renderer/cuda/soft_rasterize_cuda.cpp',
        'soft_renderer/cuda/soft_rasterize_cuda_kernel.cu',
        ]),
    CUDAExtension('soft_renderer.cuda.voxelization', [
        'soft_renderer/cuda/voxelization_cuda.cpp',
        'soft_renderer/cuda/voxelization_cuda_kernel.cu',
        ]),
    ]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image', 'tqdm', 'imageio']

setup(
    description='PyTorch implementation of "Soft Rasterizer"',
    author='Shichen Liu',
    author_email='liushichen95@gmail.com',
    license='MIT License',
    version='1.0.0',
    name='soft_renderer',
    packages=['soft_renderer', 'soft_renderer.cuda', 'soft_renderer.functional'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)
