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
* cython == 0.29.20
* scipy >= 1.2.0
* Pillow >= 8.0.0
* usd-core >= 20.11 (optional, required for USD related features such as visualization and importer / exporter)

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

If instead of the latest version you want a specific release like 0.10.0, 0.9.0, 0.9.1 or 0.1 you can then select the tag, example:


    $ git checkout v0.10.0

* If trying Kaolin with an unsupported PyTorch version, set: ``export IGNORE_TORCH_VER=1``
* To install experimental features (like :ref:`kaolin-dash3d <dash 3d>`), set: ``export KAOLIN_INSTALL_EXPERIMENTAL=1``
* If using heterogeneous GPU setup, set the architectures for which to compile the CUDA code, e.g.: ``export TORCH_CUDA_ARCH_LIST="7.0 7.5"``
* In some setups, there may be a conflict between cub available with cuda install > 11 and ``third_party/cub`` that kaolin includes as a submodule. If conflict occurs or cub is not found, set ``CUB_HOME`` to the cuda one, e.g. typically on Linux: ``export CUB_HOME=/usr/local/cuda-*/include/``


4. Install Kaolin
^^^^^^^^^^^^^^^^^

.. Note::
    On CUDA >= 11.0, CUB is already available and ``CUB_HOME`` should be specified to avoid conflict with the submodule ``third_party/cub`` (typically on linux ``export CUB_HOME=/usr/local/cuda-*/include/``).

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
