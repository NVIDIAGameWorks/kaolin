# some useful environment variables:
#
# TORCH_CUDA_ARCH_LIST
#   specify which CUDA architectures to build for
#
# IGNORE_TORCH_VER
#   ignore version requirements for PyTorch

import os
from setuptools import setup, find_packages
import importlib
from pkg_resources import parse_version
import warnings
import logging
import glob
import sys
import subprocess  # Added import

# Define version constraints
TORCH_MIN_VER = '1.6.0'
TORCH_MAX_VER = '2.7.0'  # Updated to support newer PyTorch versions
CYTHON_MIN_VER = '0.29.37'
IGNORE_TORCH_VER = os.getenv('IGNORE_TORCH_VER') is not None

# Check for PyTorch
torch_spec = importlib.util.find_spec("torch")
if torch_spec is None:
    raise ImportError(
        f"Kaolin requires PyTorch >= {TORCH_MIN_VER}, <= {TORCH_MAX_VER}. "
        "Please install it before proceeding."
    )
else:
    import torch
    torch_ver = parse_version(parse_version(torch.__version__).base_version)
    if torch_ver < parse_version(TORCH_MIN_VER) or torch_ver > parse_version(TORCH_MAX_VER):
        if IGNORE_TORCH_VER:
            warnings.warn(
                f"Kaolin is compatible with PyTorch >= {TORCH_MIN_VER}, <= {TORCH_MAX_VER}, "
                f"but found version {torch.__version__}. Continuing as IGNORE_TORCH_VER is set."
            )
        else:
            raise ImportError(
                f"Kaolin requires PyTorch >= {TORCH_MIN_VER}, <= {TORCH_MAX_VER}, "
                f"but found version {torch.__version__}. "
                "Set IGNORE_TORCH_VER=1 to proceed with this version."
            )

# Check for Cython
cython_spec = importlib.util.find_spec("cython")
if cython_spec is None:
    raise ImportError(
        f"Kaolin requires Cython >= {CYTHON_MIN_VER}. Please install it before proceeding."
    )
else:
    import Cython
    cython_ver = parse_version(Cython.__version__)
    if cython_ver < parse_version(CYTHON_MIN_VER):
        warnings.warn(
            f"Kaolin requires Cython >= {CYTHON_MIN_VER}, "
            f"but found version {Cython.__version__}. This may cause compatibility issues."
        )

# Check for NumPy
numpy_spec = importlib.util.find_spec("numpy")
if numpy_spec is None:
    raise ImportError(
        "Kaolin requires NumPy. Please install it before proceeding."
    )

import numpy
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

# Setup logging and working directory
cwd = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger()
logging.basicConfig(format='%(levelname)s - %(message)s')

def get_cuda_bare_metal_version(cuda_dir):
    """Get CUDA version from nvcc."""
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    return raw_output, release[0], release[1][0]

# Handle CUDA availability
if not torch.cuda.is_available() and os.getenv('FORCE_CUDA', '0') == '1':
    logging.warning(
        "Torch did not find available GPUs. Assuming cross-compilation.\n"
        "Default architectures: Pascal (6.0, 6.1, 6.2), Volta (7.0), Turing (7.5),\n"
        "Ampere (8.0) if CUDA >= 11.0, Hopper (9.0) if CUDA >= 11.8.\n"
        "Set TORCH_CUDA_ARCH_LIST for specific architectures."
    )
    if os.getenv("TORCH_CUDA_ARCH_LIST") is None:
        _, major, minor = get_cuda_bare_metal_version(CUDA_HOME)
        major, minor = int(major), int(minor)
        if major == 11:
            if minor == 0:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
            elif minor < 8:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9;9.0"
        elif major == 12:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"
        print(f'TORCH_CUDA_ARCH_LIST: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
elif not torch.cuda.is_available():
    logging.warning(
        "No GPUs found. Installing with CPU support only. "
        "Set FORCE_CUDA=1 for GPU cross-compilation."
    )

# Package metadata
PACKAGE_NAME = 'kaolin'
DESCRIPTION = 'Kaolin: A PyTorch library for accelerating 3D deep learning research'
URL = 'https://github.com/NVIDIAGameWorks/kaolin'
AUTHOR = 'NVIDIA'
LICENSE = 'Apache License 2.0'
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

# Version handling
version_txt = os.path.join(cwd, 'version.txt')
with open(version_txt) as f:
    version = f.readline().strip()

def write_version_file():
    """Write version to kaolin/version.py."""
    version_path = os.path.join(cwd, 'kaolin', 'version.py')
    with open(version_path, 'w') as f:
        f.write(f"__version__ = '{version}'\n")

write_version_file()

def get_requirements():
    """Read runtime dependencies from requirements files."""
    requirements = []
    with open(os.path.join(cwd, 'tools', 'viz_requirements.txt'), 'r') as f:
        requirements.extend(line.strip() for line in f)
    with open(os.path.join(cwd, 'tools', 'requirements.txt'), 'r') as f:
        requirements.extend(line.strip() for line in f)
    return requirements

def get_scripts():
    """Return list of scripts to install."""
    return ['kaolin/experimental/dash3d/kaolin-dash3d']

def get_extensions():
    """Define C++ and CUDA extensions."""
    extra_compile_args = {'cxx': ['-O3']}
    define_macros = []
    include_dirs = []
    sources = glob.glob('kaolin/csrc/**/*.cpp', recursive=True)
    is_cuda = torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1'
    
    if is_cuda:
        define_macros += [("WITH_CUDA", None), ("THRUST_IGNORE_CUB_VERSION_CHECK", None)]
        sources += glob.glob('kaolin/csrc/**/*.cu', recursive=True)
        extension = CUDAExtension
        extra_compile_args['nvcc'] = ['-O3', '-DWITH_CUDA', '-DTHRUST_IGNORE_CUB_VERSION_CHECK']
        include_dirs = get_include_dirs()
    else:
        extension = CppExtension

    extensions = [
        extension(
            name='kaolin._C',
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs
        )
    ]
    
    # Replace cudart with cudart_static
    for ext in extensions:
        ext.libraries = ['cudart_static' if x == 'cudart' else x for x in ext.libraries]

    # Cython extensions
    use_cython = True
    ext_suffix = '.pyx' if use_cython else '.cpp'
    cython_extensions = [
        CppExtension(
            'kaolin.ops.mesh.triangle_hash',
            sources=[f'kaolin/cython/ops/mesh/triangle_hash{ext_suffix}'],
            include_dirs=[numpy.get_include()],
        ),
        CppExtension(
            'kaolin.ops.conversions.mise',
            sources=[f'kaolin/cython/ops/conversions/mise{ext_suffix}'],
        ),
    ]
    
    if use_cython:
        from Cython.Build import cythonize
        from Cython.Compiler import Options
        compiler_directives = Options.get_directive_defaults()
        compiler_directives["emit_code_comments"] = False
        cython_extensions = cythonize(
            cython_extensions,
            language='c++',
            compiler_directives=compiler_directives
        )
    
    return extensions + cython_extensions

def get_include_dirs():
    """Get include directories for CUDA builds."""
    include_dirs = []
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        _, major, _ = get_cuda_bare_metal_version(CUDA_HOME)
        if "CUB_HOME" in os.environ:
            logging.warning(f"Including CUB_HOME: {os.environ['CUB_HOME']}")
            include_dirs.append(os.environ["CUB_HOME"])
        elif int(major) < 11:
            logging.warning(f"Including default CUB: {os.path.join(cwd, 'third_party/cub')}")
            include_dirs.append(os.path.join(cwd, 'third_party/cub'))
    return include_dirs

if __name__ == '__main__':
    setup(
        name=PACKAGE_NAME,
        version=version,
        author=AUTHOR,
        description=DESCRIPTION,
        url=URL,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        python_requires='~=3.7',
        packages=find_packages(exclude=('docs', 'tests', 'examples')),
        scripts=get_scripts(),
        include_package_data=True,
        install_requires=get_requirements(),
        zip_safe=False,
        ext_modules=get_extensions(),
        cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)}
    )
