Installation
=================================

Installing Kaolin should be really easy, especially if you work in virtual environments.

.. Note::
    We STRONGLY recommend using virtual environments for use with Kaolin (and in general too)!

Run the following command, from the root directory of this repository (i.e., the directory containing the `README.md` file).

.. code-block:: bash

    $ python setup.py develop

To verify your installation, fire up your python interpreter and try executing the following statements.

.. code-block:: python

    >>> import kaolin as kal
    >>> kal.__version__

You should then be able to see the version of the library installed.


.. contents::
    :local:


Build documentation
-------------------

Optionally, you might want to build the docs on your local machine. As `sphinx` has already been installed as a dependency, you only need to do

.. code-block:: bash
    
    $ cd docs
    $ sphinx-build . _build

This will build docs into the `docs/_build` directory. To access the docs, open `docs/_build/index.html` in your web browser, and voila!


Run unittests (optional)
------------------------

Another optional step. If you wish to run unittests, from the root directory of the repository (i.e., the directory containing the main `README.md` file), run

.. code-block:: bash

    $ pytest --cov=kaolin/ tests
