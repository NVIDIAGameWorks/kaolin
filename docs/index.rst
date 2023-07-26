Welcome to Kaolin Library Documentation
=======================================

.. image:: ../assets/kaolin.png

`NVIDIA Kaolin library <https://github.com/NVIDIAGameWorks/kaolin>`_ provides a PyTorch API for working with a variety of 3D representations and includes a growing collection of GPU-optimized operations such as modular differentiable rendering, fast conversions between representations, data loading, 3D checkpoints, differentiable camera API, differentiable lighting with spherical harmonics and spherical gaussians, powerful quadtree acceleration structure called Structured Point Clouds, interactive 3D visualizer for jupyter notebooks, convenient batched mesh container and more.
See :ref:`Installation <installation>`, :ref:`API Overview <overview>` and :ref:`Tutorials <tutorial_index>` to get started!

Note that Kaolin library is part of the larger `NVIDIA Kaolin effort <https://developer.nvidia.com/kaolin>`_ for 3D deep learning.


.. toctree::
   :titlesonly:
   :maxdepth: 1
   :caption: Tutorials:

   notes/tutorial_index
   notes/checkpoints
   notes/diff_render
   notes/spc_summary
   notes/differentiable_camera

.. toctree::
   :titlesonly:
   :maxdepth: 1
   :caption: API Reference:

   modules/kaolin.ops
   modules/kaolin.metrics
   modules/kaolin.io
   modules/kaolin.render
   modules/kaolin.rep
   modules/kaolin.utils
   modules/kaolin.visualize
