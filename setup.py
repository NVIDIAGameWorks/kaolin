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
import shutil

# Define version constraints
TORCH_MIN_VER = '2.5.1'
TORCH_MAX_VER = '2.12.1'  # Updated to support newer PyTorch versions
IGNORE_TORCH_VER = os.getenv('IGNORE_TORCH_VER') is not None

# Module required before installation
# trying to install it ahead turned out to be too unstable
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
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.DEBUG)

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
        "Torch did not find available GPUs. Assuming cross-compilation and all supported architectures.\n"
        "Set TORCH_CUDA_ARCH_LIST for specific architectures."
    )
    if os.getenv("TORCH_CUDA_ARCH_LIST") is None:
        _, major, minor = get_cuda_bare_metal_version(CUDA_HOME)
        major, minor = int(major), int(minor)
        if major == 11:
            if minor == 0:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0+PTX"
            elif minor < 8:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6+PTX"
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9;9.0+PTX"
        elif major == 12:
            if minor <= 6:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0+PTX"
            elif minor == 8:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX"
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0;10.0;10.3;12.0;12.1+PTX"
        elif major == 13:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0;12.1+PTX"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5+PTX"

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

def get_app_extras():
    """Auto-discover optional app extras from kaolin/app/*/requirements.txt."""
    extras = {}
    app_dir = os.path.join(cwd, 'kaolin', 'app')
    if not os.path.isdir(app_dir):
        return extras
    for app_name in sorted(os.listdir(app_dir)):
        req_file = os.path.join(app_dir, app_name, 'requirements.txt')
        if os.path.isfile(req_file):
            with open(req_file) as f:
                reqs = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            extras[f'app.{app_name}'] = reqs
    return extras

def get_scripts():
    """Return list of scripts to install."""
    return ['kaolin/experimental/dash3d/kaolin-dash3d']

def get_package_data():
    """Return package data files to include."""
    return {
        'kaolin.visualize.dash.components.kaolin_viewer': [
            '*.js',
            '*.js.map',
            '*.json',
            '*.py'
        ],
        # Behavior manifest emitted by `npm run build:dash:manifest`; the
        # Python side (`kaolin.visualize.dash.behavior_manifest`) reads this
        # file via importlib.resources at runtime.
        'kaolin.visualize.dash.components.autogen': [
            '*.json',
        ],
    }

def get_extensions():
    """Define C++ and CUDA extensions."""
    extra_compile_args = {'cxx': ['-O3']}
    define_macros = []
    include_dirs = []
    sources = glob.glob('kaolin/csrc/**/*.cpp', recursive=True)
    # FORCE_CUDA is for cross-compilation in docker build
    is_cuda = torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1'

    if is_cuda:
        define_macros += [("WITH_CUDA", None), ("THRUST_IGNORE_CUB_VERSION_CHECK", None)]
        sources += glob.glob('kaolin/csrc/**/*.cu', recursive=True)
        extension = CUDAExtension
        extra_compile_args['nvcc'] = ['-O3', '-DWITH_CUDA', '-DTHRUST_IGNORE_CUB_VERSION_CHECK']
        include_dirs = get_include_dirs()
    else:
        extension = CppExtension

    # CUDA 13's CCCL headers hard-error on MSVC's traditional (legacy) preprocessor.
    # Opt into the standard-conforming preprocessor on Windows so CUDA 13+ compiles
    # cleanly. `/Zc:preprocessor` requires MSVC >= 19.27 (VS 2019 16.5+); the CI's
    # toolchain (MSVC 19.34 / VS 2022) is well past that.
    if sys.platform == 'win32':
        extra_compile_args['cxx'].append('/Zc:preprocessor')
        if is_cuda:
            extra_compile_args['nvcc'] += ['-Xcompiler', '/Zc:preprocessor']

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

    return extensions

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

NODE_MIN_MAJOR = 25


def check_node_version():
    """Verify the installed Node.js major version is >= NODE_MIN_MAJOR.

    Raises RuntimeError if `node` is missing or older than the required major.
    Set KAOLIN_SKIP_DASH_BUILD=1 to skip the dash build (and this check) entirely.
    """
    try:
        raw = subprocess.check_output(['node', '--version'], universal_newlines=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            f"node not found, failed to build dash components. "
            f"Install Node.js >= {NODE_MIN_MAJOR}, "
            "or set KAOLIN_SKIP_DASH_BUILD=1 to skip the dash component build."
        ) from e

    # `node --version` prints e.g. "v25.0.0"
    version_str = raw.lstrip('v')
    try:
        major = int(version_str.split('.', 1)[0])
    except ValueError as e:
        raise RuntimeError(f"Could not parse node version from output {raw!r}") from e

    if major < NODE_MIN_MAJOR:
        raise RuntimeError(
            f"Kaolin requires Node.js >= {NODE_MIN_MAJOR}, but found {raw}. "
            "Upgrade Node.js, "
            "or set KAOLIN_SKIP_DASH_BUILD=1 to skip the dash component build."
        )


def build_dash_components():
    """Build the dash components before installation."""
    if os.getenv('KAOLIN_SKIP_DASH_BUILD') is not None:
        logger.info("KAOLIN_SKIP_DASH_BUILD is set, skipping dash component build.")
        return

    check_node_version()

    dash_components_dir = os.path.join(cwd, 'kaolin', 'visualize', 'dash', 'components')
    # TODO: maybe disable import if not built?

    logger.info("Building dash components...")
    try:
        # Check if node_modules exists, if not run npm install
        if not os.path.exists(os.path.join(cwd, 'node_modules')):
            logger.info("Installing npm dependencies...")
            subprocess.check_call(['npm', 'install'], cwd=cwd)
            
        # Build the components using root-level commands
        logger.info("Building JavaScript bundle...")
        subprocess.check_call(['npm', 'run', 'build:dash'], cwd=cwd)
        logger.info(f"Dash components built successfully! See: {dash_components_dir}/autogen")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to build dash components (set KAOLIN_SKIP_DASH_BUILD=1 to skip; or install nodejs): {e}")
        raise e
    except FileNotFoundError:
        raise RuntimeError("npm not found, failed to build dash components (set KAOLIN_SKIP_DASH_BUILD=1 to skip)")

build_dash_components()

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
    package_data=get_package_data(),
    install_requires=get_requirements(),
    extras_require=get_app_extras(),
    zip_safe=False,
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)}
)
