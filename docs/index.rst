Welcome to Kaolin Library Documentation
=======================================

.. image:: ../assets/kaolin.png


`NVIDIA Kaolin library <https://github.com/NVIDIAGameWorks/kaolin>`_ provides a PyTorch API for working with a variety of 3D representations and includes a growing collection of GPU-optimized operations such as modular differentiable rendering, fast conversions between representations, data loading, 3D checkpoints, differentiable camera API, differentiable lighting with spherical harmonics and spherical gaussians, powerful octree acceleration structure called Structured Point Clouds, interactive 3D visualizer for jupyter notebooks, convenient batched mesh container, quaternion operations, representation-agnostic physics simulation and more.
See :ref:`Installation <installation>`, :ref:`API Overview <overview>` and :ref:`Tutorials <tutorial_index>` to get started!


.. toctree::
   :titlesonly:
   :maxdepth: 1
   :caption: Tutorials:

   notes/tutorial_index
   notes/siggraph2025
   notes/cvpr2025
   notes/simplicits
   notes/conversions
   notes/volumetric_meshes
   notes/surface_meshes
   notes/diff_render
   notes/differentiable_camera
   notes/differentiable_lighting
   notes/pbr_shader
   notes/visualizer
   notes/spc_summary
   notes/quaternions
   notes/checkpoints


.. toctree::
   :titlesonly:
   :maxdepth: 1
   :caption: API Reference:

   modules/kaolin.ops
   modules/kaolin.math
   modules/kaolin.metrics
   modules/kaolin.io
   modules/kaolin.physics
   modules/kaolin.render
   modules/kaolin.rep
   modules/kaolin.utils
   modules/kaolin.visualize
   modules/kaolin.non_commercial
