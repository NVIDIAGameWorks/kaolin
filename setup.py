# some useful environment variables:
#
# TORCH_CUDA_ARCH_LIST
#   specify which CUDA architectures to build for
#
# IGNORE_VER_ERR
#   ignore version error for torch

from setuptools import setup, find_packages, dist
import importlib
from pkg_resources import parse_version
import warnings

TORCH_MIN_VER = '1.5.0'
TORCH_MAX_VER = '1.7.1'
CYTHON_MIN_VER = '0.29.20'

missing_modules = []
torch_spec = importlib.util.find_spec("torch")
if torch_spec is None:
    warnings.warn("Couldn't find torch installed, so this will try to install it. "
                  "If the installation fails we recommend to first install it.")
    missing_modules.append(f'torch>={TORCH_MIN_VER},<={TORCH_MAX_VER}')
else:
    import torch
    torch_ver = parse_version(torch.__version__)
    if torch_ver <= parse_version(TORCH_MIN_VER) and \
       torch_ver <= parse_version(TORCH_MAX_VER):
        warnings.warn('Kaolin is compatible with PyTorch >= 1.5.0, '
                      f'but found version {torch.__version__} instead. '
                      'This will try to install torch in the right version. '
                      'If the installation fails we recommend to first install it.')
        missing_modules.append(f'torch>={TORCH_MIN_VER},<={TORCH_MAX_VER}')

cython_spec = importlib.util.find_spec("cython")
if cython_spec is None:
    warnings.warn("Couldn't find cython installed, so this will try to install it. "
                  "If the installation fails we recommend to first instal it.")
    missing_modules.append(f'cython=={CYTHON_MIN_VER}')
else:
    import Cython
    cython_ver = parse_version(Cython.__version__)
    if cython_ver != parse_version('0.29.20'):
        warnings.warn('Kaolin is compatible with cython == 0.29.20, '
                      f'but found version {Cython.__version__} instead. '
                      'This will try to install torch in the right version. '
                      'If the installation fails we recommend to first install it.')
        missing_modules.append(f'cython=={CYTHON_MIN_VER}')


dist.Distribution().fetch_build_eggs(missing_modules)

import os
import logging

import numpy
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

cwd = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger()
logging.basicConfig(format='%(levelname)s - %(message)s')

if not torch.cuda.is_available():
    # From: https://github.com/NVIDIA/apex/blob/b66ffc1d952d0b20d6706ada783ae5b23e4ee734/setup.py
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    logging.warning(
        '\nWarning: Torch did not find available GPUs on this system.\n'
        'If your intention is to cross-compile, this is not an error.\n'
        'By default, Kaolin will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n'
        'Volta (compute capability 7.0), and Turing (compute capability 7.5).\n'
        'If you wish to cross-compile for a single specific architecture,\n'
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n')
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"

PACKAGE_NAME = 'kaolin'
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

version = '0.9.0'


def write_version_file():
    version_path = os.path.join(cwd, 'kaolin', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))


write_version_file()


def get_requirements():
    requirements = []
    if os.name != 'nt':  # no pypi torch for windows
        if os.getenv('PYTORCH_VERSION'):
            requirements.append('torch==%s' % os.getenv('PYTORCH_VERSION'))
        else:
            requirements.append(f'torch>={TORCH_MIN_VER},<={TORCH_MAX_VER}')
    requirements.append('scipy>=1.2.0,<=1.5.2')
    requirements.append('Pillow>=8.0.0')
    requirements.append('tqdm>=4.51.0')
    requirements.append('usd-core==20.11; python_version < "3.8"')

    return requirements


def filters_cu(sources, with_cuda):
    if with_cuda:
        return sources
    else:
        return [s for s in sources if not s.endswith('.cu')]


def get_extensions():
    extra_compile_args = {'cxx': ['-O3']}
    define_macros = []
    # FORCE_CUDA is for cross-compilation in docker build
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        with_cuda = True
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args.update({'nvcc': ['-O3']})
    else:
        extension = CppExtension
        with_cuda = False
    extensions = []
    extensions.append(
        extension(
            name='kaolin.ops.packed_simple_sum_cuda',
            sources=filters_cu(['kaolin/csrc/cuda/ops/packed_simple_sum.cpp',
                                'kaolin/csrc/cuda/ops/packed_simple_sum_impl.cu'],
                               with_cuda),
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        )
    )
    extensions.append(
        extension(
            name='kaolin.ops.tile_to_packed_cuda',
            sources=filters_cu(['kaolin/csrc/cuda/ops/tile_to_packed_cuda.cpp',
                                'kaolin/csrc/cuda/ops/tile_to_packed_cuda_kernel.cu'],
                               with_cuda),
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        )
    )
    extensions.append(
        extension(
            name='kaolin.metrics.sided_distance_cuda',
            sources=filters_cu(['kaolin/csrc/cuda/metrics/sided_distance.cpp',
                                'kaolin/csrc/cuda/metrics/sided_distance_cuda.cu'],
                               with_cuda),
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        )
    )
    extensions.append(
        extension(
            name='kaolin.metrics.unbatched_triangle_distance_cuda',
            sources=filters_cu(
                ['kaolin/csrc/cuda/metrics/unbatched_triangle_distance_cuda.cpp',
                 'kaolin/csrc/cuda/metrics/unbatched_triangle_distance_cuda_kernel.cu'],
                with_cuda),
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        )
    )
    extensions.append(
        extension(
            name='kaolin.ops.mesh.mesh_intersection_cuda',
            sources=filters_cu(
                ['kaolin/csrc/cuda/ops/mesh/mesh_intersection_cuda.cpp',
                 'kaolin/csrc/cuda/ops/mesh/mesh_intersection_cuda_kernel.cu'],
                with_cuda),
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        )
    )
    extensions.append(
        extension(
            name='kaolin.render.mesh.dibr_rasterization_cuda',
            sources=filters_cu(['kaolin/csrc/cuda/render/dibr.cpp',
                                'kaolin/csrc/cuda/render/dibr_for.cu',
                                'kaolin/csrc/cuda/render/dibr_back.cu'],
                               with_cuda),
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        )
    )

    extensions.append(
        extension(
            name='kaolin.ops.conversions.unbatched_mcube_cuda',
            sources=filters_cu(['kaolin/csrc/cuda/ops/conversions/unbatched_mcube/unbatched_mcube_cuda.cpp',
                                'kaolin/csrc/cuda/ops/conversions/unbatched_mcube/unbatched_mcube_cuda_kernel.cu'],
                               with_cuda),
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        )
    )


    # use cudart_static instead
    for extension in extensions:
        extension.libraries = ['cudart_static' if x == 'cudart' else x
                               for x in extension.libraries]

    use_cython = True
    ext = '.pyx' if use_cython else '.cpp'

    cython_extensions = [
        CppExtension(
            'kaolin.ops.mesh.triangle_hash',
            sources=[
                f'kaolin/csrc/cpu/ops/mesh/triangle_hash{ext}'
            ],
            include_dirs=[numpy.get_include()],
        ),
        CppExtension(
            'kaolin.ops.conversions.mise',
            sources=[
                f'kaolin/csrc/cpu/ops/conversions/mise{ext}'
            ],
        ),
    ]

    if use_cython:
        from Cython.Build import cythonize
        from Cython.Compiler import Options
        compiler_directives = Options.get_directive_defaults()
        compiler_directives["emit_code_comments"] = False
        cython_extensions = cythonize(cython_extensions, language='c++',
                                      compiler_directives=compiler_directives)

    return extensions + cython_extensions

def get_include_dirs():
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        cub_home = os.environ.get("CUB_HOME", os.path.join(cwd, 'third_party/cub'))
        return [cub_home]
    else:
        return None

if __name__ == '__main__':
    setup(
        # Metadata
        name=PACKAGE_NAME,
        version=version,
        author=AUTHOR,
        description=DESCRIPTION,
        url=URL,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        python_requires='~=3.6',

        # Package info
        packages=find_packages(exclude=('docs', 'tests', 'examples')),
        install_requires=get_requirements(),
        include_dirs=get_include_dirs(),
        zip_safe=True,
        ext_modules=get_extensions(),
        cmdclass={
            'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
        }
    )
