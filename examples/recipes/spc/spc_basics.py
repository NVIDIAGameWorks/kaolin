# ==============================================================================================================
# The following snippet demonstrates the basic usage of kaolin's compressed octree,
# termed "Structured Point Cloud (SPC)".
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
points = torch.tensor([[-1.0, -1.0, -1.0], [-0.9, -0.95, -1.0], [1.0, 1.0, 1.0]], device='cuda')

# In kaolin, operations are batched by default
# Here, in contrast, we use a single point cloud and therefore invoke an unbatched conversion function.
# The Structured Point Cloud will be using 3 levels of detail
spc = kaolin.ops.conversions.pointcloud.unbatched_pointcloud_to_spc(pointcloud=points, level=3)

# SPC is a batched object, and most of its fields are packed.
# (see: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#kaolin-ops-batch )
# spc.length defines the boundaries between different batched SPC instances the same object holds.
# Here we keep track of a single entry batch, which has 8 octree non-leaf cells.
print(f'spc.batch_size: {spc.batch_size}')
print(f'spc.lengths (cells per batch entry): {spc.lengths}')

# SPC is hierarchical and keeps information for every level of detail from 0 to 3.
# spc.point_hierarchies keeps the sparse, zero indexed coordinates of each occupied cell, per level.
print(f'SPC keeps track of total of {spc.point_hierarchies.shape[0]} parent + leaf cells:')

# To separate the boundaries, the spc.pyramids field is used.
# This field is not-packed, unlike the other SPC fields.
pyramid_of_first_entry_in_batch = spc.pyramids[0]
cells_per_level = pyramid_of_first_entry_in_batch[0]
cumulative_cells_per_level = pyramid_of_first_entry_in_batch[1]
for i, lvl_cells in enumerate(cells_per_level[:-1]):
    print(f'LOD #{i} has {lvl_cells} cells.')

# The spc.octrees field keeps track of the fundamental occupancy information of each cell in the octree.
print('The occupancy of each octant parent cell, in Morton / Z-curve order is:')
print(['{0:08b}'.format(octree_byte) for octree_byte in spc.octrees])

# Since SPCs are low level objects, they require bookkeeping of multiple fields.
# For ease of use, these fields are collected and tracked within a single class: kaolin.ops.spc.SPC
# See references at the header for elaborate information on how to use this object.
