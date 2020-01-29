import os
import io
import logging
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import numpy as np


cwd = os.path.dirname(os.path.abspath(__file__))

if not torch.cuda.is_available():
    # From: https://github.com/NVIDIA/apex/blob/b66ffc1d952d0b20d6706ada783ae5b23e4ee734/setup.py
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    print('\nWarning: Torch did not find available GPUs on this system.\n',
          'If your intention is to cross-compile, this is not an error.\n'
          'By default, Kaolin will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n'
          'Volta (compute capability 7.0), and Turing (compute capability 7.5).\n'
          'If you wish to cross-compile for a single specific architecture,\n'
          'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n')
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"

PACKAGE_NAME = 'kaolin'
VERSION = '0.1.0'
DESCRIPTION = 'Kaolin: A PyTorch library for accelerating 3D deep learning research'
URL = 'https://github.com/NVIDIAGameWorks/kaolin'
AUTHOR = 'NVIDIA'
LICENSE = 'Apache License 2.0'
DOWNLOAD_URL = ''
LONG_DESCRIPTION = """
Kaolin is a PyTorch library aiming to accelerate 3D deep learning research. Kaolin provides efficient implementations
of differentiable 3D modules for use in deep learning systems. With functionality to load and preprocess several popular
3D datasets, and native functions to manipulate meshes, pointclouds, signed distance functions, and voxel grids, Kaolin
mitigates the need to write wasteful boilerplate code. Kaolin packages together several differentiable graphics modules
including rendering, lighting, shading, and view warping. Kaolin also supports an array of loss functions and evaluation
metrics for seamless evaluation and provides visualization functionality to render the 3D results. Importantly, we curate
a comprehensive model zoo comprising many state-of-the-art 3D deep learning architectures, to serve as a starting point
for future research endeavours.
"""


logger = logging.getLogger()
logging.basicConfig(format='%(levelname)s - %(message)s')


# Check that PyTorch version installed meets minimum requirements
if torch.__version__ < '1.2.0':
    logger.warning(f'Kaolin is tested with PyTorch >= 1.2.0. Found version {torch.__version__} instead.')


# Get version number from version.py
version = {}
with open("kaolin/version.py") as fp:
    exec(fp.read(), version)


def build_deps():
    print('Building nv-usd...')
    os.system('./buildusd.sh')


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get('encoding', 'utf8')
    ) as fp:
        return fp.read()


def get_extensions():
    use_cython = os.getenv('USE_CYTHON')
    ext = '.pyx' if use_cython else '.cpp'
    cython_extensions = [
        CppExtension(
            'kaolin.triangle_hash',
            sources=[
                f'kaolin/cython/triangle_hash{ext}'
            ],
        ),
        CppExtension(
            'kaolin.triangle_hash',
            sources=[
                f'kaolin/cython/triangle_hash{ext}'
            ],
        ),
        CppExtension(
            'kaolin.mise',
            sources=[
                f'kaolin/cython/mise{ext}'
            ],
        ),
        CppExtension(
            'kaolin.mcubes',
            sources=[
                f'kaolin/cython/mcubes{ext}',
                'kaolin/cython/pywrapper.cpp',
                'kaolin/cython/marchingcubes.cpp'
            ],
            extra_compile_args=['-std=c++11'],
        ),
        CppExtension(
            'kaolin.nnsearch',
            sources=[
                f'kaolin/cython/nnsearch{ext}'
            ],
        ),
    ]

    # If building with readthedocs, don't compile CUDA extensions
    if os.getenv('READTHEDOCS') != 'True':
        cuda_extensions = [
            CUDAExtension('kaolin.cuda.load_textures', [
                'kaolin/cuda/load_textures_cuda.cpp',
                'kaolin/cuda/load_textures_cuda_kernel.cu',
            ]),
            CUDAExtension('kaolin.cuda.sided_distance', [
                'kaolin/cuda/sided_distance.cpp',
                'kaolin/cuda/sided_distance_cuda.cu',
            ]),
            CUDAExtension('kaolin.cuda.furthest_point_sampling', [
                'kaolin/cuda/furthest_point_sampling.cpp',
                'kaolin/cuda/furthest_point_sampling_cuda.cu',
            ]),
            CUDAExtension('kaolin.cuda.ball_query', [
                'kaolin/cuda/ball_query.cpp',
                'kaolin/cuda/ball_query_cuda.cu',
            ]),
            CUDAExtension('kaolin.cuda.three_nn', [
                'kaolin/cuda/three_nn.cpp',
                'kaolin/cuda/three_nn_cuda.cu',
            ]),
            CUDAExtension('kaolin.cuda.tri_distance', [
                'kaolin/cuda/triangle_distance.cpp',
                'kaolin/cuda/triangle_distance_cuda.cu',
            ]),
            CUDAExtension('kaolin.cuda.mesh_intersection', [
                'kaolin/cuda/mesh_intersection.cpp',
                'kaolin/cuda/mesh_intersection_cuda.cu',
            ]),
            CUDAExtension('kaolin.graphics.nmr.cuda.rasterize_cuda', [
                'kaolin/graphics/nmr/cuda/rasterize_cuda.cpp',
                'kaolin/graphics/nmr/cuda/rasterize_cuda_kernel.cu',
            ]),
            CUDAExtension('kaolin.graphics.dib_renderer.cuda.rasterizer', [
                'kaolin/graphics/dib_renderer/cuda/rasterizer.cpp',
                'kaolin/graphics/dib_renderer/cuda/rasterizer_cuda.cu',
                'kaolin/graphics/dib_renderer/cuda/rasterizer_cuda_back.cu',
            ]),
        ]
    else:
        cuda_extensions = []

    if use_cython:
        from Cython.Build import cythonize
        from Cython.Compiler import Options
        compiler_directives = Options.get_directive_defaults()
        compiler_directives["emit_code_comments"] = False
        cython_extensions = cythonize(cython_extensions, language='c++',
                                      compiler_directives=compiler_directives)

    return cython_extensions + cuda_extensions


cwd = os.path.dirname(os.path.abspath(__file__))


def get_requirements():
    return [
        'matplotlib<3.0.0',
        'scikit-image',
        'shapely',
        'trimesh>=3.0',
        'scipy',
        'sphinx==2.2.0',    # pinned to resolve issue with docutils 0.16b0.dev
        'pytest>=4.6',
        'pytest-cov>=2.7',
        'tqdm',
        'pytest',
        'pptk',
        'autopep8',
        'flake8',
    ]


if __name__ == '__main__':
    build_deps()
    setup(
        # Metadata
        name=PACKAGE_NAME,
        version=version['__version__'],
        author=AUTHOR,
        description=DESCRIPTION,
        url=URL,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        python_requires='>3.6',

        # Package info
        packages=find_packages(exclude=('docs', 'test', 'examples')),
        install_requires=get_requirements(),
        zip_safe=True,
        ext_modules=get_extensions(),
        include_dirs=[np.get_include()],
        cmdclass={'build_ext': BuildExtension}
    )
