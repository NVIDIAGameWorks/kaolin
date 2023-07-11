.. _tutorial_index:

Tutorial Index
==============

Kaolin provides tutorials as ipython notebooks, docs pages and simple scripts. Note that the links
point to master.


Detailed Tutorials
------------------

* `Camera and Rasterization <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/camera_and_rasterization.ipynb>`_: Rasterize ShapeNet mesh with nvdiffrast and camera:
    * Load ShapeNet mesh
    * Preprocess mesh and materials
    * Create a camera with ``from_args()`` general constructor
    * Render a mesh with multiple materials with nvdiffrast
    * Move camera and see the resulting rendering
* `Optimizing Diffuse Lighting <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/diffuse_lighting.ipynb>`_: Optimize lighting parameters with spherical gaussians and spherical harmonics:
    * Load an obj mesh with normals and materials
    * Rasterize the diffuse and specular albedo
    * Render and optimize diffuse lighting:
      * Spherical harmonics
      * Spherical gaussian with inner product implementation
      * Spherical gaussian with fitted approximation
* `Optimize Diffuse and Specular Lighting with Spherical Gaussians <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/sg_specular_lighting.ipynb>`_:
    * Load an obj mesh with normals and materials
    * Generate view rays from camera
    * Rasterize the diffuse and specular albedo
    * Render and optimize diffuse and specular lighting with spherical gaussians
* `Working with Surface Meshes <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/working_with_meshes.ipynb>`_:
    * loading and constructing :class:`kaolin.rep.SurfaceMesh` objects
    * batching of meshes
    * auto-computing common attributes (like ``face_normals``)
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
* Visualization:
    * `visualize_main.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/visualize_main.py>`_: using Timelapse API to write mock 3D checkpoints
    * `fast_mesh_sampling.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/preprocess/fast_mesh_sampling.py>`_: Using CachedDataset to preprocess a ShapeNet dataset we can sample point clouds efficiently at runtime
* Camera:
    * `cameras_differentiable.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/camera/cameras_differentiable.py>`_: optimize a camera position
    * `camera_transforms.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/camera/camera_transforms.py>`_: using :func:`Camera.transform()` function
    * `camera_ray_tracing.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/camera/camera_ray_tracing.py>`_: how to design a ray generating function using :class:`Camera` objects
    * `camera_properties.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/camera/camera_properties.py>`_: exposing some the camera attributes and properties
    * `camera_opengl_shaders.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/camera/camera_opengl_shaders.py>`_: Using the camera with glumpy
    * `camera_movement.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/camera/camera_movement.py>`_: Manipulating a camera position and zoom
    * `camera_init_simple.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/camera/camera_init_simple.py>`_: Making Camera objects with the flexible :func:`Camera.from_args()` constructor
    * `camera_init_explicit.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/camera/camera_init_explicit.py>`_: Making :class:`CameraIntrinsics` and :class:`CameraExtrinsics` with all the different constructors available
    * `camera_coordinate_systems.py <https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/recipes/camera/camera_coordinate_systems.py>`_: Changing coordinate system in a :class:`Camera` object

