# ==============================================================================================================
# The following snippet demonstrates the basic usage of kaolin's dual octree, an octree which keeps features
# at the 8 corners of each cell (the primary octree keeps a single feature at each cell center).
# In this example we sample an interpolated value according to the 8 corners of a cell.
# The implementation is realized through kaolin's "Structured Point Cloud (SPC)".
# Note this is a low level structure: practitioners are encouraged to visit the references below.
# ==============================================================================================================
# See also:
#
#  - Code: kaolin.ops.spc.SPC
#    https://kaolin.readthedocs.io/en/latest/modules/kaolin.rep.html?highlight=SPC#kaolin.rep.Spc
#
#  - Tutorial: Understanding Structured Point Clouds (SPCs)
#    https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/understanding_spcs_tutorial.ipynb
#
#  - Documentation: Structured Point Clouds
#    https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.spc.html?highlight=spc#kaolin-ops-spc
# ==============================================================================================================

import torch
import kaolin

# Construct SPC from some points data. Point coordinates are expected to be normalized to the range [-1, 1].
# To keep the example readable, by default we set the SPC level to 1: root + 8 cells
# (note that with a single LOD, only 2 cells should be occupied due to quantization)
level = 1
points = torch.tensor([[-1.0, -1.0, -1.0], [-0.9, -0.95, -1.0], [1.0, 1.0, 1.0]], device='cuda')
spc = kaolin.ops.conversions.pointcloud.unbatched_pointcloud_to_spc(pointcloud=points, level=level)

# Construct the dual octree with an unbatched operation, each cell is now converted to 8 corners
# More info about batched / packed tensors at:
# https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#kaolin-ops-batch
pyramid = spc.pyramids[0]  # The pyramids field is batched, we select the singleton entry, #0
point_hierarchy = spc.point_hierarchies   # point_hierarchies is a packed tensor, so no need to unbatch
point_hierarchy_dual, pyramid_dual = kaolin.ops.spc.unbatched_make_dual(point_hierarchy=point_hierarchy,
                                                                        pyramid=pyramid)
# kaolin allows for interchangeable usage of the primary and dual octrees via the "trinkets" mapping
# trinkets are indirection pointers (in practice, indices) from the nodes of the primary octree
# to the nodes of the dual octree. The nodes of the dual octree represent the corners of the voxels
# defined by the primary octree.
trinkets, _ = kaolin.ops.spc.unbatched_make_trinkets(point_hierarchy, pyramid, point_hierarchy_dual, pyramid_dual)

# We'll now apply the dual octree and trinkets to perform trilinaer interpolation.
# First we'll generate some features for the corners.
# The first dimension of pyramid / pyramid_dual specifies how many unique points exist per level.
# For the pyramid_dual, this means how many "unique corners" are in place (as neighboring cells may share corners!)
num_of_corners_at_last_lod = pyramid_dual[0, level]
feature_dims = 32
feats = torch.rand([num_of_corners_at_last_lod, feature_dims], device='cuda')

# Create some query coordinate with normalized values in the range [-1, 1], here we pick (0.5, 0.5, 0.5).
# We'll also modify the dimensions of the query tensor to match the interpolation function api:
# batch dimension refers to the unique number of spc cells we're querying.
# samples_count refers to the number of interpolations we perform per cell.
query_coord = points.new_tensor((0.5, 0.5, 0.5)).unsqueeze(0)  # Tensor of (batch, 3), in this case batch=1
sampled_query_coords = query_coord.unsqueeze(1)  # Tensor of (batch, samples_count, 3), in this case samples_count=1

# unbatched_query converts from normalized coordinates to the index of the cell containing this point.
# The query_index can be used to pick the point from point_hierarchy
query_index = kaolin.ops.spc.unbatched_query(spc.octrees, spc.exsum, query_coord, level, with_parents=False)

# The unbatched_interpolate_trilinear function uses the query coordinates to perform trilinear interpolation.
# Here, unbatched specifies this function supports only a single SPC at a time.
# Per single SPC, we may interpolate a batch of coordinates and samples
interpolated = kaolin.ops.spc.unbatched_interpolate_trilinear(coords=sampled_query_coords,
                                                              pidx=query_index.int(),
                                                              point_hierarchy=point_hierarchy,
                                                              trinkets=trinkets, feats=feats, level=level)
print(f'Interpolated a tensor of shape {interpolated.shape} with values: {interpolated}')