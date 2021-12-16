.. _kaolin.metrics:

kaolin.metrics
==============

Metrics are differentiable operators that can be used to compute loss or accuracy.

We currently provide an IoU for voxelgrid, sided distance based metrics such as chamfer distance,
point_to_mesh_distance and other simple regularization such as uniform_laplacian_smoothing.
For tetrahedral mesh, we support the equivolume and AMIPS losses. 

.. toctree::
   :maxdepth: 2
   :titlesonly:

   kaolin.metrics.pointcloud
   kaolin.metrics.render
   kaolin.metrics.trianglemesh
   kaolin.metrics.voxelgrid
   kaolin.metrics.tetmesh
