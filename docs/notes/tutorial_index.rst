.. _tutorial_index:

Tutorial Index
==============

Kaolin provides tutorials as ipython notebooks, docs pages and simple scripts. Note that the links
point to master.


Detailed Tutorials
------------------

* `Deep Marching Tetrahedra <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/dmtet_tutorial.ipynb>`_: reconstructs a tetrahedral mesh from point clouds with `DMTet <https://nv-tlabs.github.io/DMTet/>`_, covering:
    * generating data with Omniverse Kaolin App
    * loading point clouds from a ``.usd`` file
    * chamfer distance as a loss function
    * differentiable marching tetrahedra
    * using Timelapse API for 3D checkpoints
    * visualizing 3D results of training
* `Understanding Structured Point Clouds (SPCs) <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/understanding_spcs_tutorial.ipynb>`_: walks through SPC features, covering:
    * under-the-hood explanation of SPC, why it's useful and key ops
    * loading a mesh
    * sampling a point cloud
    * converting a point cloud to SPC
    * setting up camera
    * rendering SPC with ray tracing
    * storing features in an SPC
* `Differentiable Rendering <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/dibr_tutorial.ipynb>`_: optimizes a triangular mesh from images using `DIB-R <https://github.com/nv-tlabs/DIB-R-Single-Image-3D-Reconstruction>`_ renderer, covering:
    * generating data with Omniverse Kaolin App, and loading this synthetic data
    * loading a mesh
    * computing mesh laplacian
    * DIB-R rasterization
    * differentiable texture mapping
    * computing mask intersection-over-union loss (IOU)
    * using Timelapse API for 3D checkpoints
    * visualizing 3D results of training
* `Fitting a 3D Bounding Box <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/bbox_tutorial.ipynb>`_: fits a 3D bounding box around an object in images using `DIB-R <https://github.com/nv-tlabs/DIB-R-Single-Image-3D-Reconstruction>`_ renderer, covering:
    * generating data with Omniverse Kaolin App, and loading this synthetic data
    * loading a mesh
    * DIB-R rasterization
    * computing mask intersection-over-union loss (IOU)
* :ref:`3d_viz`: explains saving 3D checkpoints and visualizing them, covering:
    * using Timelapse API for writing 3D checkpoints
    * understanding output file format
    * visualizing 3D checkpoints using Omniverse Kaolin App
    * visualizing 3D checkpoints using bundled ``kaolin-dash3d`` commandline utility
* `Reconstructing Point Cloud with DMTet <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/dmtet_tutorial.ipynb>`_: Trains an SDF estimator to reconstruct a mesh from a point cloud covering:
    * using point clouds data generated with Omniverse Kaolin App
    * loading point clouds from an USD file.
    * defining losses and regularizer for a mesh with point cloud ground truth
    * applying marching tetrahedra
    * using Timelapse API for 3D checkpoints
    * visualizing 3D checkpoints using ``kaolin-dash3d``

Simple Recipes
--------------

* I/O and Data Processing:
    * `usd_kitchenset.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/usd_kitchenset.py>`_: loading multiple meshes from a ``.usd`` file and saving
    * `spc_from_pointcloud.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/dataload/spc_from_pointcloud.py>`_: converting a point cloud to SPC object
    * `occupancy_sampling.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/preprocess/occupancy_sampling.py>`_: computing occupancy function of points in a mesh using ``check_sign``
    * `spc_basics.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/spc/spc_basics.py>`_: showing attributes of an SPC object
    * `spc_dual_octree.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/spc/spc_dual_octree.py>`_: computing and explaining the dual of an SPC octree
    * `spc_trilinear_interp.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/spc/spc_trilinear_interp.py>`_: computing trilinear interpolation of a point cloud on an SPC
    * `spc_batching.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/spc/spc_batching.py>`_: using SPC objects with unbatched ops in kaolin
* Visualization:
    * `visualize_main.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/visualize_main.py>`_: using Timelapse API to write mock 3D checkpoints

