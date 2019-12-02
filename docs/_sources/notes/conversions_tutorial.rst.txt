Conversions across various representations
===========================================

Kaolin supports conversions across all popular 3D representations. Here's a quickfire introduction.

For this tutorial, we will load the `model.obj` file inside of the `tests` directory (`tests` is contained in the base directory of the Kaolin repository).

.. code-block:: python

    >>> mesh = kal.rep.TriangleMesh.from_obj('model.obj')

As a sanity check to ensure the mesh is read in correctly, run.

.. code-block:: python

    >>> mesh.vertices.shape
    torch.Size([482, 3])
    >>> mesh.faces.shape
    torch.Size([482, 3])


.. image:: images/reps.png


TriangleMesh to VoxelGrid
----------------------------

To convert a triangle mesh to a `32 x 32 x 32` voxel grid, use

.. code-block:: python

    >>> voxels = kal.conversions.trianglemesh_to_voxelgrid(mesh, 32)

TriangleMesh to PointCloud
----------------------------

To convert a triangle mesh to a pointcloud with 1000 points, use

.. code-block:: python

    >>> voxels = kal.conversions.trianglemesh_to_pointcloud(mesh, 1000)


TriangleMesh to Signed Distance Function (SDF)
------------------------------------------------

To convert a triangle mesh to a signed distance function, use

.. code-block:: python

    >>> sdf = kal.conversions.trianglemesh_to_sdf(mesh)


VoxelGrid to PointCloud
------------------------

To convert a voxel grid to a pointcloud (containing 1000 points), use

.. code-block:: python

    >>> sdf = kal.conversions.voxelgrid_to_pointcloud(voxels, 1000)


VoxelGrid to TriangleMesh
--------------------------

To convert a voxel grid to a triangle mesh, use

.. code-block:: python

    >>> verts, faces = kal.conversions.voxelgrid_to_trianglemesh(voxel, mode='marching_cubes')


VoxelGrid to QuadMesh
----------------------

To convert a voxel grid to a quad mesh, use

.. code-block:: python

    >>> verts, faces = kal.conversions.voxelgrid_to_quadmesh(voxels, thresh=.1)


VoxelGrid to SDF
-----------------

To convert a voxel grid to a signed distance function (SDF), use

.. code-block:: python

    >>> sdf = kal.conversions.voxelgrid_to_sdf(voxels, thresh=.5)


SDF to PointCloud
------------------

To convert an SDF to a pointcloud, use

.. code-block:: python

    >>> points = kal.conversions.sdf_to_pointcloud(sdf, bbox_center=0., resolution=10, bbox_dim=1,  num_points = 10000)


SDF to TriangleMesh
--------------------

To convert an SDF to a triangle mesh, use

.. code-block:: python

    >>> verts, faces = kal.conversions.sdf_to_trianglemesh(sdf, bbox_center=0., resolution=10, bbox_dim=1)


SDF to VoxelGrid
-----------------

To convert an SDF to a voxel grid, use

.. code-block:: python

    >>> voxels = kal.conversions.sdf_to_voxelgrid(sdf, bbox_center=0., resolution=10, bbox_dim=1)


PointCloud to VoxelGrid
------------------------

To convert a pointcloud to a voxel grid, use

.. code-block:: python

    >>> voxels = kal.conversions.pointcloud_to_voxelgrid(points, 32, 0.1)


PointCloud to TriangleMesh
-----------------------------

To convert a pointcloud to a triangle mesh, use

.. code-block:: python

    >>> mesh_ = kal.conversions.pointcloud_to_trianglemesh(points)


PointCloud to SDF
-------------------

To convert a pointcloud to an SDF, use

.. code-block:: python

    >>> sdf_ = kal.conversions.pointcloud_to_trianglemesh(points)
