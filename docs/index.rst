.. Kaolin documentation master file, created by
   sphinx-quickstart on Mon Jun 10 08:31:06 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

   
   
Kaolin
=================================

Kaolin is a PyTorch library aimed at accelerating 3D deep learning research. Kaolin provides efficient implementations of several differentiable modules for use in neural networks. With several native functions to manipulate meshes, pointclouds, signed distance functions, and voxel grids, researchers/practitioners need not dabble with writing boilerplate code anymore. Kaolin packages several differentiable graphics, vision, and robotics modules (eg. rendering, lighting, shading, view warping, etc.). It also provides easy access to loading and pre-processing several 3D datasets. Additionally, we curate a model zoo comprising several state-of-the-art 3D deep learning architectures, to serve as a starting point for future research endeavours.

.. image:: _static/img/kaolin.png

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Notes

   notes/installation
   notes/introduction
   notes/hello3d
   notes/usd_tutorial
   notes/datasets_tutorial
   notes/conversions_tutorial
   notes/vision_tutorial
   notes/graphics_tutorial
   notes/differentiable_rendering
   notes/pointnet
   notes/pixel2mesh
   notes/geometrics

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: API Reference:

   modules/conversions
   modules/datasets
   modules/graphics
   modules/mathutils
   modules/metrics
   modules/models
   modules/rep
   modules/transforms
   modules/vision
   modules/visualize



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
