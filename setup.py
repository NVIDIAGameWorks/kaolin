# some useful environment variables:
#
# TORCH_CUDA_ARCH_LIST
#   specify which CUDA architectures to build for
#
# IGNORE_TORCH_VER
#   ignore version requirements for PyTorch

import os
from setuptools import setup, find_packages, dist
import importlib
from pkg_resources import parse_version
import subprocess
import warnings

TORCH_MIN_VER = '1.6.0'
TORCH_MAX_VER = '2.0.1'
CYTHON_MIN_VER = '0.29.20'
IGNORE_TORCH_VER = os.getenv('IGNORE_TORCH_VER') is not None

# Module required before installation
# trying to install it ahead turned out to be too unstable.
torch_spec = importlib.util.find_spec("torch")
if torch_spec is None:
    raise ImportError(
        f"Kaolin requires PyTorch >={TORCH_MIN_VER}, <={TORCH_MAX_VER}, "
        "but couldn't find the module installed."
    )
else:
    import torch
    torch_ver = parse_version(parse_version(torch.__version__).base_version)
    if (torch_ver < parse_version(TORCH_MIN_VER) or
        torch_ver > parse_version(TORCH_MAX_VER)):
        if IGNORE_TORCH_VER:
            warnings.warn(
                f'Kaolin is compatible with PyTorch >={TORCH_MIN_VER}, <={TORCH_MAX_VER}, '
                f'but found version {torch.__version__}. Continuing with the installed '
                'version as IGNORE_TORCH_VER is set.'
            )
        else:
            raise ImportError(
                f'Kaolin requires PyTorch >={TORCH_MIN_VER}, <={TORCH_MAX_VER}, '
                f'but found version {torch.__version__} instead.'
                'If you wish to install with this specific version set IGNORE_TORCH_VER=1.'
            )

missing_modules = []

cython_spec = importlib.util.find_spec("cython")
if cython_spec is None:
    warnings.warn(
        f"Kaolin requires cython == {CYTHON_MIN_VER}, "
        "but couldn't find the module installed. "
        "This setup is gonna try to install it..."
    )
    missing_modules.append(f'cython=={CYTHON_MIN_VER}')

else:
    import Cython
    cython_ver = parse_version(Cython.__version__)
    if cython_ver != parse_version('0.29.20'):
        warnings.warn('Kaolin requires cython == 0.29.20, '
                      f'but found version {Cython.__version__} instead.')

numpy_spec = importlib.util.find_spec("numpy")

if numpy_spec is None:
    warnings.warn(
        f"Kaolin requires numpy, but couldn't find the module installed. "
        "This setup is gonna try to install it..."
    )
    missing_modules.append('numpy')

dist.Distribution().fetch_build_eggs(missing_modules)

cython_spec = importlib.util.find_spec("cython")
if cython_spec is None:
    raise ImportError(
        f"Kaolin requires cython == {CYTHON_MIN_VER} "
        "but couldn't find or install it."
    )

numpy_spec = importlib.util.find_spec("numpy")
if numpy_spec is None:
    raise ImportError(
        f"Kaolin requires numpy but couldn't find or install it."
    )

import os
import sys
import logging
import glob

import numpy
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

cwd = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger()
logging.basicConfig(format='%(levelname)s - %(message)s')

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

if not torch.cuda.is_available():
    if os.getenv('FORCE_CUDA', '0') == '1':
        # From: https://github.com/NVIDIA/apex/blob/c4e85f7bf144cb0e368da96d339a6cbd9882cea5/setup.py
        # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
        # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
        logging.warning(
            "Torch did not find available GPUs on this system.\n"
            "If your intention is to cross-compile, this is not an error.\n"
            "By default, Kaolin will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
            "Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
            "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0),\n"
            "and, if the CUDA version is >= 11.8, Hopper (compute capability 9.0).\n"
            "If you wish to cross-compile for a single specific architecture,\n"
            'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n'
        )
        if os.getenv("TORCH_CUDA_ARCH_LIST", None) is None:
            _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
            if int(bare_metal_major) == 11:
                if int(bare_metal_minor) == 0:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
                elif int(bare_metal_minor) < 8:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"
        print(f'TORCH_CUDA_ARCH_LIST: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    else:
        logging.warning(
            "Torch did not find available GPUs on this system.\n"
            "Kaolin will install only with CPU support and will have very limited features.\n"
            "If your wish to cross-compile for GPU `export FORCE_CUDA=1` before running setup.py."
        )

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

version_txt = os.path.join(cwd, 'version.txt')
with open(version_txt) as f:
    version = f.readline().strip()

def write_version_file():
    version_path = os.path.join(cwd, 'kaolin', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))


write_version_file()


def get_requirements():
    requirements = []
    if sys.version_info >= (3, 10):
        warnings.warn("usd-core is not compatible with python_version >= 3.10 "
                      "and won't be installed, please use supported python_version "
                      "to use USD related features")
    with open(os.path.join(cwd, 'tools', 'viz_requirements.txt'), 'r') as f:
        for line in f.readlines():
            requirements.append(line.strip())
    with open(os.path.join(cwd, 'tools', 'requirements.txt'), 'r') as f:
        for line in f.readlines():
            requirements.append(line.strip())
    return requirements


def get_scripts():
    return ['kaolin/experimental/dash3d/kaolin-dash3d']


def get_extensions():
    extra_compile_args = {'cxx': ['-O3']}
    define_macros = []
    include_dirs = []
    sources = glob.glob('kaolin/csrc/**/*.cpp', recursive=True)
    # FORCE_CUDA is for cross-compilation in docker build
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        with_cuda = True
        define_macros += [("WITH_CUDA", None), ("THRUST_IGNORE_CUB_VERSION_CHECK", None)]
        sources += glob.glob('kaolin/csrc/**/*.cu', recursive=True)
        extension = CUDAExtension
        extra_compile_args.update({'nvcc': [
            '-O3',
            '-DWITH_CUDA',
            '-DTHRUST_IGNORE_CUB_VERSION_CHECK'
        ]})
        include_dirs = get_include_dirs()
    else:
        extension = CppExtension
        with_cuda = False
    extensions = []
    extensions.append(
        extension(
            name='kaolin._C',
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs
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
                f'kaolin/cython/ops/mesh/triangle_hash{ext}'
            ],
            include_dirs=[numpy.get_include()],
        ),
        CppExtension(
            'kaolin.ops.conversions.mise',
            sources=[
                f'kaolin/cython/ops/conversions/mise{ext}'
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
    include_dirs = []
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        _, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
        if "CUB_HOME" in os.environ:
            logging.warning(f'Including CUB_HOME ({os.environ["CUB_HOME"]}).')
            include_dirs.append(os.environ["CUB_HOME"])
        else:
            if int(bare_metal_major) < 11:
                logging.warning(f'Including default CUB_HOME ({os.path.join(cwd, "third_party/cub")}).')
                include_dirs.append(os.path.join(cwd, 'third_party/cub'))

    return include_dirs


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
        python_requires='~=3.7',

        # Package info
        packages=find_packages(exclude=('docs', 'tests', 'examples')),
        scripts=get_scripts(),
        include_package_data=True,
        install_requires=get_requirements(),
        zip_safe=False,
        ext_modules=get_extensions(),
        cmdclass={
            'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
        }
    )
