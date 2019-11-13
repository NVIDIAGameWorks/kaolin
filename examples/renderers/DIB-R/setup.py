# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:

#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.

#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.


from setuptools import setup, find_packages

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'torchvision', 'scikit-image', 'tqdm', 'imageio', 'opencv-python']

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ext_modules=[
        CUDAExtension('dr_batch_nsoft32', [
            'graphics/cuda/dr.cpp',
            'graphics/cuda/dr_cuda_for.cu',
            'graphics/cuda/dr_cuda_back.cu'
        ])
    ]

if __name__ == '__main__':
	print(find_packages(exclude=( 'test')))
	setup(
	    description='PyTorch implementation of "Dib-renderer"',
	    version='1.0.0',
	    name='graphics',
	    packages=find_packages(exclude=( 'test')),
	    zip_safe=True,
	    install_requires=INSTALL_REQUIREMENTS,
   		ext_modules=ext_modules,
        cmdclass = {'build_ext': BuildExtension}   
	)
