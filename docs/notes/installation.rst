:orphan:

.. _installation:

Installation
============

Kaolin is written with PyTorch and uses C++ / CUDA for efficient custom ops.

Requirements
------------

* Linux, macOS (CPU-only) or Windows
* Python >= 3.6 (3.6 and 3.7 recommended for Windows)
* CUDA >= 10.0 (with 'nvcc' installed)

Dependencies
------------

* torch >= 1.5, <= 1.10.2
* cython == 0.29.20 (auto-installed)
* scipy >= 1.2.0 (auto-installed)
* Pillow >= 8.0.0 (auto-installed)
* usd-core >= 20.11 (auto-installed; required for USD I/O and 3D checkpoints with :class:`~kaolin.visualize.Timelapse`)

Installation from source
------------------------

.. Note::
    We recommend installing Kaolin into a virtual environment, for instance with `Anaconda <https://www.anaconda.com/>`_:
    
    .. code-block:: bash
    
        $ conda create --name kaolin python=3.7
        $ conda activate kaolin

1. Clone Repository
^^^^^^^^^^^^^^^^^^^

Clone and optionally check out an `official release <https://github.com/NVIDIAGameWorks/kaolin/tags>`_:

.. code-block:: bash

    $ git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
    $ cd kaolin
    $ git checkout v0.10.0

2. Install Pytorch
^^^^^^^^^^^^^^^^^^
Follow `official instructions <https://pytorch.org>`_ to install PyTorch of a supported version.
Kaolin may be able to work with other PyTorch versions. See below for overriding PyTorch version check during install.


3. Optional Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* If trying Kaolin with an unsupported PyTorch version, set: ``export IGNORE_TORCH_VER=1``
* To install experimental features (like :ref:`kaolin-dash3d <dash 3d>`), set: ``export KAOLIN_INSTALL_EXPERIMENTAL=1``
* If using heterogeneous GPU setup, set the architectures for which to compile the CUDA code, e.g.: ``export TORCH_CUDA_ARCH_LIST="7.0 7.5"``
* In some setups, there may be a conflict between cub available with cuda install > 11 and ``third_party/cub`` that kaolin includes as a submodule. If conflict occurs or cub is not found, set ``CUB_HOME`` to the cuda one, e.g. typically on Linux: ``export CUB_HOME=/usr/local/cuda-*/include/``


4. Install Kaolin
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
    $ pytest tests/python/

.. Note::
    These tests rely on CUDA operations and will fail if you installed on CPU only, where not all functionality is available.
