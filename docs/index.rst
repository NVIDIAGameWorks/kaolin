Welcome to Kaolin Library Documentation
=======================================

.. image:: ../assets/kaolin.png

NVIDIA Kaolin library provides a PyTorch API for working with a variety of 3D representations and includes a growing collection of GPU-optimized operations such as modular differentiable rendering, fast conversions between representations, data loading, 3D checkpoints and more.
See :ref:`Installation <installation>`, :ref:`API Overview <overview>` and :ref:`Tutorials <tutorial_index>` to get started!

Kaolin library is part of a larger suite of tools for 3D deep learning research. For example, `Omniverse Kaolin app <https://docs.omniverse.nvidia.com/app_kaolin/app_kaolin/overview.html>`_ allows interactive visualization of 3D checkpoints. To find out more about the Kaolin ecosystem, visit the `NVIDIA Kaolin Dev Zone page <https://developer.nvidia.com/kaolin>`_.


.. toctree::
   :titlesonly:
   :maxdepth: 1
   :caption: Tutorials:

   notes/tutorial_index
   notes/checkpoints
   notes/diff_render
   notes/spc_summary
   notes/camera_summary

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
