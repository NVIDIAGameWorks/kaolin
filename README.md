<p align="center">
    <img src="assets/kaolin.png">
</p>


## Kaolin: Accelerating 3D deep learning research

**[Documentation](http://nvidiagameworks.github.io/kaolin)** | **[Paper](https://arxiv.org/abs/1911.05063)**


Kaolin is a PyTorch library aiming to accelerate 3D deep learning research. Kaolin provides efficient implementations of differentiable 3D modules for use in deep learning systems. With functionality to load and preprocess several popular 3D datasets, and native functions to manipulate meshes, pointclouds, signed distance functions, and voxel grids, Kaolin mitigates the need to write wasteful boilerplate code. Kaolin packages together several differentiable graphics modules including rendering, lighting, shading, and view warping. Kaolin also supports an array of loss functions and evaluation metrics for seamless evaluation and provides visualization functionality to render the 3D results. Importantly, we curate a comprehensive model zoo comprising many state-of-the-art 3D deep learning architectures, to serve as a starting point for future research endeavours.



## Table of Contents
- [Functionality](#functionality)
- [Installation And Usage](#installation-and-usage)
  - [Supported Platforms](#supported-platforms)
  - [Install Kaolin](#install-kaolin)
  - [Verify Installation](#verify-installation)
  - [Building the Documentation](#building-the-documentation)
  - [Running Unittests](#running-unittests)
- [Main Modules](#main-modules)
- [Getting Started](#getting-started)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Functionality

<p align="center">
    <img src="assets/kaolin_applications.png">
</p>

Currently, the (beta) release contains several processing functions for 3D deep learning on meshes, voxels, signed distance functions, and pointclouds. Loading of several popular datasets (eg. ShapeNet, ModelNet, SHREC) are supported out-of-the-box. We also implement several 3D conversion and transformation operations (both within and across the aforementioned representations). 

Kaolin supports several 3D tasks such as:
* Differentiable renderers (Neural Mesh Renderer, Soft Rasterizer, Differentiable Interpolation-based Renderer, and a modular and extensible abstract DifferentiableRenderer specification).
* Single-image based mesh reconstruction (Pixel2Mesh, GEOMetrics, OccupancyNets, and more...)
* Pointcloud classification and segmentation (PointNet, PoinNet++, DGCNN, ...)
* Mesh classification and segmentation
* 3D superresolution on voxel grids
* Basic graphics utilities (lighting, shading, etc.)

## Model Zoo

Kaolin curates a large _model zoo_ containing reference implementations of popular 3D DL architectures. Head over [here](kaolin/models) to check them out.


## Installation and Usage

> **NOTE**: The API is currently somewhat unstable, as we're making constant changes. (It's a beta release)

### Supported Platforms
Kaolin is officially supported on Linux platforms and has been built and tested on Ubuntu 18. Windows and Mac support should be considered experimental.

### Install Kaolin

We highly recommend installing Kaolin inside of a virtual environment (such as ones created using `conda` or `virtualenv`). Kaolin expects Python 3.6+, and currently needs a CUDA-enabled machine (i.e., with `nvcc` installed) for the build.

First create a virtual environment. In this example, we show how to create a `conda` virtual environment for installing kaolin.
```sh
$ conda create --name kaolin python=3.6
$ conda activate kaolin
```

Now, install the dependencies (`numpy` and `torch`). Note that the setup file does not automatically install these dependencies.
```sh
conda install numpy
```

Install PyTorch, by following instructions from https://pytorch.org/

Now, you can install the library. From the root directory of this repo (i.e., the directory containing this `README` file), run

```sh
$ python setup.py install
```

During installation, the *packman* package manager will 
download the nv-usd package to `~/packman-repo/` containing the necessary packages for reading and writing Universal Scene Description (USD) files. 

### Verify installation

To verify that `kaolin` has been installed, fire up your python interpreter, and execute the following commands.

```python
>>> import kaolin as kal
>>> print(kal.__version)
```

### Building the Documentation

To delve deeper into the library, build the documentation. From the root directory of the repository (i.e., the directory containing this `README` file), execute the following.

```bash
$ cd docs
$ sphinx-build . _build
```

### Running Unittests

To run unittests, from the root directory of the repository (i.e., the directory containing this `README` file), execute the following commands.

```bash
$ pytest tests/
```

## Main Modules

- **rep**: Supported 3D asset representations include: Triangle Meshes, Quad Meshes, Voxel Grids, Point Clouds, Signed Distance Functions (SDF).

- **conversions**: Supports conversion across all popular 3D representations.

- **models**: Implemented models include: 
    - DGCNN (https://arxiv.org/abs/1801.07829v1)
    - DIB-R (https://arxiv.org/abs/1908.01210)
    - GEOMetrics (https://arxiv.org/abs/1901.11461)
    - Image2Mesh (https://arxiv.org/abs/1711.10669)
    - Occupancy Network (https://arxiv.org/abs/1812.03828)
    - Pixel2Mesh (https://arxiv.org/abs/1804.01654)
    - PointNet (https://arxiv.org/abs/1612.00593)
    - PointNet++ (https://arxiv.org/abs/1706.02413)
    - MeshEncoder: A simple mesh encoder architecture.
    - GraphResNet: MeshEncoder with residual connections.
    - OccupancyNetworks (https://arxiv.org/abs/1812.03828)
    - And many more!

- **graphics**: Kaolin provides a flexible and modular framework for building differentiable renderers, making it simple to replace individual components with new ones. Kaolin also provides implementations of the following differentiable renderers:
    - DIB-R (https://arxiv.org/abs/1908.01210)
    - SoftRas (https://arxiv.org/abs/1904.01786)
    - Neural 3D Mesh Renderer (https://arxiv.org/abs/1711.07566)

- **metrics**: Implemented metrics and loss functions:
    - Mesh: Triangle Distance, Chamfer Distance, Edge Length regularization, Laplacian regularization, Point to Surface distance, Normal consistency
    - Point Cloud: Sided Distance, Chamfer Distance, Directed Distance
    - Voxel Grid: Intersection Over Union (3D IoU), F-Score

### 

## Getting Started

Take a look at some of our examples!! Examples include differentiable renderers, voxel superresolution, etc. Begin [here](examples).

> **Note:** We will (very soon) host our docs online. Stay tuned for the link. Until then, please follow instructions from [above](#building-the-documentation) to build docs.


## Contributors
[Krishna Murthy Jatavallabhula](https://krrish94.github.io/),
[Edward Smith](https://github.com/EdwardSmith1884),
[Jean-Francois Lafleche](https://www.linkedin.com/in/jflafleche),
[Clement Fuji Tsang](https://ca.linkedin.com/in/clement-fuji-tsang-b8028a82),
[Artem Rozantsev](https://sites.google.com/site/artemrozantsev/),
[Wenzheng Chen](http://www.cs.toronto.edu/~wenzheng/),
[Tommy Xiang](https://github.com/TommyX12),
[Rev Lebaredian](https://blogs.nvidia.com/blog/author/revlebaredian/),
[Gavriel State](https://ca.linkedin.com/in/gavstate),
[Sanja Fidler](https://www.cs.utoronto.ca/~fidler/),

## Acknowledgements
**[Acknowledgements](Acknowledgements.txt)**

We would like to thank [Amlan Kar](https://amlankar.github.io) for suggesting the need for this library. We also thank [Ankur Handa](http://ankurhanda.github.io) for his advice during the initial and final stages of the project. Many thanks to [Joanh Philion](https://scholar.google.com/citations?user=VVIAoY0AAAAJ&hl=en), [Daiqing Li](https://www.linkedin.com/in/daiqing-li-23873789?originalSubdomain=ca), [Mark Brophy](https://ca.linkedin.com/in/mark-brophy-3a298382), [Jun Gao](http://www.cs.toronto.edu/~jungao/), and [Huan Ling](http://www.cs.toronto.edu/~linghuan/) who performed detailed internal reviews, and provided constructive comments. We also thank [Gavriel State](https://ca.linkedin.com/in/gavstate) for all his help during the project.

Most importantly, we thank all 3D DL researchers who have made their code available as open-source. The field could use a lot more of it!



## License and Copyright
**[LICENSE](LICENSE)** | **[COPYRIGHT](COPYRIGHT)**

If you find this library useful, consider citing the following paper:
```
@article{kaolin2019arxiv,
    author = {J., {Krishna Murthy} and Smith, Edward and Lafleche, Jean-Francois and {Fuji Tsang}, Clement and Rozantsev, Artem and Chen, Wenzheng and Xiang, Tommy and Lebaredian, Rev and Fidler, Sanja},
    title = {Kaolin: A PyTorch Library for Accelerating 3D Deep Learning Research},
    journal = {arXiv:1911.05063},
    year = {2019},
}

```