# ==============================================================================================================
# The following snippet demonstrates the basic usage of kaolin's dual octree, an octree which keeps features
# at the 8 corners of each cell (the primary octree keeps a single feature at each cell center).
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

# Let's compare the primary and dual octrees.
# The function 'unbatched_get_level_points' yields a tensor which lists all points / sparse cell coordinates occupied
# at a certain level.
#          [Primary octree]                        [Dual octree]
#          . . . . . . . .                        X . . .X. . . X
#          | .   X  .  X  | .                     | .      .     | .
#          |   . . . . . . . .           ===>     |   X . . X . . . X
#          |   | .   X  . | X   .                 X   | .      . |     .
#          |   |   . . . . . . . .                |   |   X . . .X. . . X
#          |   |    |     |       |               |   X    |     |       |
#           . .|. . | . . .       |      ===>      X .|. . X . . X       |
#             .| X  |.  X   .     |                  .|    |.      .     X
#               . . | . . . . .   |                    X . | . X . . X   |
#                 . | X  .  X   . |                      . |    .      . |
#                   . . . . . . . .                        X . . X . . . X
#
primary_lod0 = kaolin.ops.spc.unbatched_get_level_points(point_hierarchy, pyramid, level=0)
primary_lod1 = kaolin.ops.spc.unbatched_get_level_points(point_hierarchy, pyramid, level=1)
dual_lod0 = kaolin.ops.spc.unbatched_get_level_points(point_hierarchy_dual, pyramid_dual, level=0)
dual_lod1 = kaolin.ops.spc.unbatched_get_level_points(point_hierarchy_dual, pyramid_dual, level=1)
print(f'Primary octree: Level 0 (root cells): \n{primary_lod0}')
print(f'Dual octree: Level 0 (root corners): \n{dual_lod0}')
print(f'Primary octree: Level 1 (cells): \n{primary_lod1}')
print(f'Dual octree: Level 1 (corners): \n{dual_lod1}')

# kaolin allows for interchangeable usage of the primary and dual octrees.
# First we have to create a mapping between them:
trinkets, _ = kaolin.ops.spc.unbatched_make_trinkets(point_hierarchy, pyramid, point_hierarchy_dual, pyramid_dual)

# trinkets are indirection pointers (in practice, indices) from the nodes of the primary octree
# to the nodes of the dual octree. The nodes of the dual octree represent the corners of the voxels
# defined by the primary octree.
print(f'point_hierarchy is of shape {point_hierarchy.shape}')
print(f'point_hierarchy_dual is of shape {point_hierarchy_dual.shape}')
print(f'trinkets is of shape {trinkets.shape}')
print(f'Trinket indices are multilevel: {trinkets}')
# See also spc_trilinear_interp.py for a practical application which uses the dual octree & trinkets
