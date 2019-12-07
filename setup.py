import os
import site
from pathlib import Path
from setuptools import setup, find_packages
import subprocess
from distutils.extension import Extension
cwd = os.path.dirname(os.path.abspath(__file__))

###############################################################
# Version, create_version_file, and package name
#
# Example for release (1.2.3):
#  KAOLIN_BUILD_VERSION=1.2.3 \
#  KAOLIN_BUILD_NUMBER=1 python setup.py install
###############################################################
package_name = os.getenv('KAOLIN_PACKAGE_NAME', 'kaolin')
version = '0.2.0'
if os.getenv('KAOLIN_PACKAGE_NAME'):
    assert os.getenv('KAOLIN_BUILD_NUMBER') is not None
    build_number = int(os.getenv('KAOLIN_BUILD_NUMBER'))
    version = os.getenv('KAOLIN_BUILD_VERSION')
    if build_number > 1:
        version += '.post' + str(build_number)
else:
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
        version += '+' + sha[:7]
    except Exception:
        pass
print('Building wheel {}-{}'.format(package_name, version))


# All the work we need to do _before_ setup runs
def build_deps():
    print('--Building version ' + version)
    # version_path = os.path.join(cwd, 'kaolin', 'version.py')
    # with open(version_path, 'w') as f:
    #     f.write('__version__ = \'{}\'\n'.format(version))

    # build nv-usd
    os.system('./buildusd.sh')


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fp:
        return fp.read()


requirements = [
    'matplotlib<3.0.0',
    'scikit-image',
    'shapely',
    'trimesh>=3.0',
    'scipy',
    'sphinx',
    # 'sphinx_rtd_theme',
    'pytest>=4.6',
    'pytest-cov>=2.7',
    'tqdm',
    'pytest',
    'pptk',
    'Cython',
    'autopep8',
    'flake8',
]


if __name__ == '__main__':

    build_deps()
    setup(
        # Metadata
        name=package_name,
        version=version,
        author='nvidia',
        description='Kaolin: A PyTorch library for accelerating 3D deep learning research',
        url='',
        long_description='',
        license='',
        python_requires='>3.6',

        # Package info
        packages=find_packages(exclude=('docs', 'test', 'examples')),

        zip_safe=True,
        install_requires=requirements
        # ext_modules=ext_modules,
        # cmdclass = {'build_ext': BuildExtension}

    )

    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    import Cython
    from Cython.Build import cythonize
    import numpy as np

    triangle_hash_module = Extension(
        'kaolin.triangle_hash',
        sources=[
            'kaolin/cython/triangle_hash.pyx'
        ],
        include_dirs=[np.get_include()],
        # libraries=['m']
    )

    mise_module = Extension(
        'kaolin.mise',
        sources=[
            'kaolin/cython/mise.pyx'
        ],
        # libraries=['m'],
        include_dirs=[np.get_include()]
    )
    mcubes_module = Extension(
        'kaolin.mcubes',
        sources=[
            'kaolin/cython/mcubes.pyx',
            'kaolin/cython/pywrapper.cpp',
            'kaolin/cython/marchingcubes.cpp'
        ],
        language='c++',
        extra_compile_args=['-std=c++11'],
        include_dirs=[np.get_include()]
    )
    nnsearch_module = Extension(
        'kaolin.nnsearch',
        sources=[
            'kaolin/cython/nnsearch.pyx'
        ],
        include_dirs=[np.get_include()]
    )

    ext_modules = [
        triangle_hash_module,
        mise_module,
        mcubes_module,
        nnsearch_module,
    ]
    
    # If building with readthedocs, don't compile CUDA extensions
    if os.getenv('READTHEDOCS') != 'True':
        ext_modules += [
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
            ])]

    setup(
        # Metadata
        name=package_name,
        version=version,
        author='nvidia',
        description='A 3D Deep Learning Library',
        url='',
        long_description='',
        license='',
        python_requires='>3.6',

        # Package info
        packages=find_packages(exclude=('docs', 'test', 'examples')),

        zip_safe=True,
        ext_modules=cythonize(ext_modules),
        cmdclass={'build_ext': BuildExtension}

    )
