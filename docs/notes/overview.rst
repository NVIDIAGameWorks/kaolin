:orphan:

.. _overview:

API Overview
============

Below is a summary of Kaolin functionality. Refer to :ref:`tutorial_index` for specific use cases, examples
and recipes that use these building blocks.

Operators for 3D Data:
^^^^^^^^^^^^^^^^^^^^^^

:ref:`kaolin/ops<kaolin.ops>` contains operators for efficient processing functions of batched 3d models and tensors. We provide, conversions between 3d representations, primitives batching of heterogenenous data, and efficient mainstream functions on meshes and voxelgrids.

.. toctree::
   :maxdepth: 2

   ../modules/kaolin.ops

I/O:
^^^^

:ref:`kaolin/io<kaolin.io>` contains functionality to interact with files.

We provide, importers and exporters to popular format such as .obj and .usd, but also utility functions and classes to preprocess and cache datasets with specific transforms.

.. toctree::
   :maxdepth: 2

   ../modules/kaolin.io

Metrics:
^^^^^^^^

:ref:`kaolin/metrics<kaolin.metrics>` contains functions to compute distance and losses such as point_to_mesh distance, chamfer distance, IoU, or laplacian smoothing. 

.. toctree::
   :maxdepth: 2

   ../modules/kaolin.metrics

Differentiable Rendering:
^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`kaolin/render<kaolin.render>` provide functions related to differentiable rendering, such a DIB-R rasterization, application of camera projection / translation / rotation, lighting, and textures.

.. toctree::
   :maxdepth: 2

   ../modules/kaolin.render

3D Checkpoints and Visualization:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`kaolin/visualize<kaolin.visualize>` contains utilities for writing 3D checkpoints for visualization. Currently we provide timelapse exporter that can be quickly picked up by the `Omniverse Kaolin App <https://docs.omniverse.nvidia.com/app_kaolin/app_kaolin/user_manual.html#training-visualizer>`_.

.. toctree::
   :maxdepth: 2

   ../modules/kaolin.visualize

Utilities:
^^^^^^^^^^

:ref:`kaolin/utils<kaolin.utils>` contains utility functions to help development of application or research scripts. We provide functions to display and check informations about tensors, and features to fix seed.

.. toctree::
   :maxdepth: 2

   ../modules/kaolin.utils

Non Commercial
^^^^^^^^^^^^^^

:ref:`kaolin/non_commercial<kaolin.non_commercial>` contains features under `NSCL license <https://github.com/NVIDIAGameWorks/kaolin/blob/master/LICENSE.NSCL>`_ restricted to non commercial usage for research and evaluation purposes.

.. toctree::
   :maxdepth: 2

   ../modules/kaolin.non_commercial

Licenses
========

Most of Kaolin's repository is under `Apache v2.0 license <https://github.com/NVIDIAGameWorks/kaolin/blob/master/LICENSE>`_, except under :ref:`kaolin/non_commercial<kaolin.non_commercial>` which is under `NSCL license <https://github.com/NVIDIAGameWorks/kaolin/blob/master/LICENSE.NSCL>`_ restricted to non commercial usage for research and evaluation purposes. For example, FlexiCubes method is included under :ref:`non_commercial<kaolin.non_commercial>`.

Default `kaolin` import includes Apache-licensed components:

.. code-block:: python

   import kaolin

The non-commercial components need to be explicitly imported as:

.. code-block:: python

   import kaolin.non_commercial
