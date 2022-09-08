:orphan:

.. _installation:

Installation
============

Most functions in Kaolin use PyTorch with custom high-performance code in C++ and CUDA. For this reason,
full Kaolin functionality is only available for systems with an NVIDIA GPU, supporting CUDA. While it is possible to install
Kaolin on other systems, only a fraction of operations will be available for a CPU-only install.

Requirements
------------

* Linux, macOS (CPU-only) or Windows
* Python >= 3.7, <= 3.9
* `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ >= 10.0 (with 'nvcc' installed) See `CUDA Toolkit Archive <https://developer.nvidia.com/cuda-toolkit-archive>`_ to install older version.

Dependencies
------------

* torch >= 1.8, <= 1.12.1
* cython == 0.29.20 (auto-installed)
* scipy >= 1.2.0 (auto-installed)
* Pillow >= 8.0.0 (auto-installed)
* usd-core >= 20.11 (auto-installed; required for USD I/O and 3D checkpoints with :class:`~kaolin.visualize.Timelapse`)

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
    $ git checkout v0.12.0 # optional

2. Test CUDA
^^^^^^^^^^^^

You can verify that CUDA is properly installed at the desired version with nvcc by running the following:

.. code-block:: bash

    $ nvidia-smi
    $ nvcc --version

3. Install Pytorch
^^^^^^^^^^^^^^^^^^

Follow `official instructions <https://pytorch.org>`_ to install PyTorch of a supported version.
Kaolin may be able to work with other PyTorch versions, but we only explicitly test within the version range listed above.
See below for overriding PyTorch version check during install.

Here is how to install the latest Pytorch version supported by Kaolin for cuda 11.3:

.. code-block:: bash

    $ pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113


4. Optional Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* If trying Kaolin with an unsupported PyTorch version, set: ``export IGNORE_TORCH_VER=1``
* To install experimental features (like :ref:`kaolin-dash3d <dash 3d>`), set: ``export KAOLIN_INSTALL_EXPERIMENTAL=1``
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
    $ pytest tests/python/

.. Note::
    These tests rely on CUDA operations and will fail if you installed on CPU only, where not all functionality is available.
