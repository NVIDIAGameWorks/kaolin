# ==============================================================================================================
# The following snippet demonstrates how to build kaolin's compressed octree,
# "Structured Point Cloud (SPC)", from raw point cloud data.
# ==============================================================================================================
# See also:
#
#  - Tutorial: Understanding Structured Point Clouds (SPCs)
#    https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/understanding_spcs_tutorial.ipynb
#
#  - Documentation: Structured Point Clouds
#    https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.spc.html?highlight=spc#kaolin-ops-spc
# ==============================================================================================================

import torch
import kaolin

# Create some point data with features
# Point coordinates are expected to be normalized to the range [-1, 1].
points = torch.tensor([
    [-1.0, -1.0, -1.0],
    [-0.9, -0.95, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, -0.1, 0.3],
    [1.0, 1.0, 1.0]
], device='cuda')
features = torch.tensor([
    [0.1, 1.1, 2.1],
    [0.2, 1.2, 2.2],
    [0.3, 1.3, 2.3],
    [0.4, 1.4, 2.4],
    [0.5, 1.5, 2.5],
], device='cuda')

# Structured Point Cloud will be using 3 levels of detail
level = 3

# In kaolin, operations are batched by default
# Here, in contrast, we use a single point cloud and therefore invoke an unbatched conversion function.
# For more information about batched operations, see:
# https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#kaolin-ops-batch
spc = kaolin.ops.conversions.pointcloud.unbatched_pointcloud_to_spc(pointcloud=points,
                                                                    level=level,
                                                                    features=features)

# SPC is an object which keep tracks of the various octree component
print(spc)
print(f'SPC keeps track of the following cells in {level} levels of detail (parents + leaves):\n'
      f' {spc.point_hierarchies}\n')

# Note that the point cloud coordinates are quantized to integer coordinates.
# During conversion, when points fall within the same cell, their features are averaged
print(f'Features for leaf cells:\n {spc.features}')
