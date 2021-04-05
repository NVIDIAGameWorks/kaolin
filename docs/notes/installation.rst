Installation
============

Kaolin is written with Pytorch and C++ / CUDA for efficient custom ops.

Requirements
------------

* Linux, macOS or Windows
* Python >= 3.6 (3.6 or 3.7 recommended)
* CUDA >= 10.0 (with 'nvcc' installed)

Dependencies
------------

* torch >= 1.5
* cython == 0.29.20
* scipy >= 1.2.0
* Pillow >= 8.0.0
* usd-core == 20.11 (optional, required for USD related features such as visualization and importer / exporter)

Installation from source
------------------------

.. Note::
    If you just want to try Kaolin, we recommend using a virtual environment, for instance with `Anaconda <https://www.anaconda.com/>`_:
    
    .. code-block:: bash
    
        $ conda create --name kaolin python=3.7
        $ conda activate kaolin

We recommend following instructions from `https://pytorch.org <https://pytorch.org>`_ for installing PyTorch, and `https://cython.readthedocs.io <https://cython.readthedocs.io/en/latest/src/quickstart/install.html>`_ for installing cython, however Kaolin installation will attempt to automatically install the latest if none is installed (may fail on some systems).

To install the library. You must first clone the repository:

.. code-block:: bash

    $ git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
    $ cd kaolin

you can then select the tag if you want 0.9.0 or 0.1 specific release, example:

.. code-block:: bash

    $ git checkout v0.9.0

.. code-block:: bash

    $ python setup.py develop

.. Note::
    If you are using heterogeneous GPUs setup set the architectures for which you want to compile the cuda code using the ``TORCH_CUDA_ARCH_LIST`` environment variable.

    Example:
    
    .. code-block:: bash
    
        $ export TORCH_CUDA_ARCH_LIST="7.0 7.5"

.. Note::
    Kaolin can be installed without GPU, however, CPU support is limited to some ops.

Testing your installation
-------------------------

A quick test is to see if you can properly import kaolin and print the current version by running the following:

.. code-block:: bash

    $ python -c "import kaolin; print(kaolin.__version__)"

Running tests
^^^^^^^^^^^^^

A more exhaustive test is to execute all the official tests.

First, pytest dependencies are necessary to run those tests, to install those run:

.. code-block:: bash

    $ pip install -r tools/ci_requirements.txt
 
Then run the tests as following:

.. code-block:: bash

    $ pytest tests/python/

.. Note::
    These tests rely on cuda operations and will fail if you installed on CPU only, where not all functionality is available.
