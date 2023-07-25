:orphan:

.. _installation:

Installation
============

Most functions in Kaolin use PyTorch with custom high-performance code in C++ and CUDA. For this reason,
full Kaolin functionality is only available for systems with an NVIDIA GPU, supporting CUDA. While it is possible to install
Kaolin on other systems, only a fraction of operations will be available for a CPU-only install.

Requirements
------------

* Linux, Windows, or macOS (CPU-only)
* Python >= 3.7, <= 3.10
* `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ >= 10.0 (with 'nvcc' installed) See `CUDA Toolkit Archive <https://developer.nvidia.com/cuda-toolkit-archive>`_ to install older version.
* torch >= 1.8, <= 2.0.1

Quick Start (Linux, Windows)
----------------------------
| Make sure any of the supported CUDA and torch versions below are pre-installed.
| The latest version of Kaolin can be installed with pip:

.. code-block:: bash

    $ pip install kaolin==0.14.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{TORCH_VER}_cu{CUDA_VER}.html

.. Note::
    Replace *TORCH_VER* and *CUDA_VER* with any of the compatible options below.


.. rst-class:: center-align-center-col

    +------------------+-----------+-----------+-----------+-----------+-----------+
    | **torch / CUDA** | **cu113** | **cu115** | **cu116** | **cu117** | **cu118** |
    +==================+===========+===========+===========+===========+===========+
    | **torch-2.0.0**  |           |           |           |     ✓     |     ✓     |
    +------------------+-----------+-----------+-----------+-----------+-----------+
    | **torch-2.0.1**  |           |           |           |     ✓     |     ✓     |
    +------------------+-----------+-----------+-----------+-----------+-----------+
    | **torch-1.13.1** |           |           |     ✓     |     ✓     |           |
    +------------------+-----------+-----------+-----------+-----------+-----------+
    | **torch-1.13.0** |           |           |     ✓     |     ✓     |           |
    +------------------+-----------+-----------+-----------+-----------+-----------+
    | **torch-1.12.1** |     ✓     |           |     ✓     |           |           |
    +------------------+-----------+-----------+-----------+-----------+-----------+
    | **torch-1.12.0** |     ✓     |           |     ✓     |           |           |
    +------------------+-----------+-----------+-----------+-----------+-----------+
    | **torch-1.11.0** |     ✓     |     ✓     |           |           |           |
    +------------------+-----------+-----------+-----------+-----------+-----------+

For example, to install kaolin for torch 1.12.1 and CUDA 11.3:

.. code-block:: bash

    $ pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html

You can check https://nvidia-kaolin.s3.us-east-2.amazonaws.com/index.html to see all the wheels available.

Installation from source
------------------------

.. Note::
    We recommend installing Kaolin into a virtual environment. For instance to create a new environment with `Anaconda <https://www.anaconda.com/>`_:
    
    .. code-block:: bash
    
        $ conda create --name kaolin python=3.7
        $ conda activate kaolin

1. Clone Repository
^^^^^^^^^^^^^^^^^^^

Clone and optionally check out an `official release <https://github.com/NVIDIAGameWorks/kaolin/tags>`_:

.. code-block:: bash

    $ git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
    $ cd kaolin
    $ git checkout v0.13.0 # optional

2. Install dependencies
^^^^^^^^^^^^^^^^^^^^^^^

You can install the dependencies running:

.. code-block:: bash

    $ pip install -r tools/build_requirements.txt -r tools/viz_requirements.txt -r tools/requirements.txt

2. Test CUDA
^^^^^^^^^^^^

You can verify that CUDA is properly installed at the desired version with nvcc by running the following:

.. code-block:: bash

    $ nvidia-smi
    $ nvcc --version

3. Install Pytorch
^^^^^^^^^^^^^^^^^^

Follow `official instructions <https://pytorch.org>`_ to install PyTorch of a supported version.
Kaolin may be able to work with other PyTorch versions, but we only explicitly test within the version range 1.10.0 to 2.0.0.
See below for overriding PyTorch version check during install.

Here is how to install the latest Pytorch version supported by Kaolin for cuda 11.8:

.. code-block:: bash

    $ pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118


4. Optional Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* If trying Kaolin with an unsupported PyTorch version, set: ``export IGNORE_TORCH_VER=1``
* If using heterogeneous GPU setup, set the architectures for which to compile the CUDA code, e.g.: ``export TORCH_CUDA_ARCH_LIST="7.0 7.5"``
* In some setups, there may be a conflict between cub available with cuda install > 11 and ``third_party/cub`` that kaolin includes as a submodule. If conflict occurs or cub is not found, set ``CUB_HOME`` to the cuda one, e.g. typically on Linux: ``export CUB_HOME=/usr/local/cuda-*/include/``


5. Install Kaolin
^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ python setup.py develop

.. Note::
    Kaolin can be installed without GPU, however, CPU support is limited and many CUDA-only functions will be missing.

Testing your installation
-------------------------

Run a quick test of your installation and version:

.. code-block:: bash

    $ python -c "import kaolin; print(kaolin.__version__)"

Running tests
^^^^^^^^^^^^^

For an exhaustive check, install testing dependencies and run tests as follows:

.. code-block:: bash

    $ pip install -r tools/ci_requirements.txt
    $ export CI='true' # on Linux
    $ set CI='true' # on Windows
    $ pytest --import-mode=importlib -s tests/python/

.. Note::
    These tests rely on CUDA operations and will fail if you installed on CPU only, where not all functionality is available.
