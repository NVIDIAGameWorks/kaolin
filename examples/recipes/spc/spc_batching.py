# ==============================================================================================================
# The following snippet demonstrates how to use a batched Spc object with unbatched operations.
# ==============================================================================================================
# See also:
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
from kaolin.rep.spc import Spc

lod = 3                         # In this example we set the SPC to 3 levels of detail
pts_per_spc = 100, 500, 1000    # This example works with a batch of SPCs, each of different size

# Construct SPC from some points data. Point coordinates are expected to be normalized to the range [-1, 1].
# We start by creating each individual entry separately, using the SPC low level api
octrees = list()
for size in pts_per_spc:
    points = torch.rand(size, 3, device='cuda')                              # Randomize some points
    points = points * 2.0 - 1.0                                              # Normalize points between [-1, 1]
    points = kaolin.ops.spc.quantize_points(points.contiguous(), level=lod)  # Quantize them for octree
    octree = kaolin.ops.spc.unbatched_points_to_octree(points, level=lod)    # Generate single octree entry
    octrees.append(octree)

# Then we join them together to form the batched SPC instance, of 3 entries.
# From this point onwards we can access the SPC via the object-oriented api.
spc = Spc.from_list(octrees)

# The Spc is a batched object. In turn, its fields are either packed or batched.
# (see: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#kaolin-ops-batch )
# spc.length defines the boundaries between different batched SPC instances the same object holds.
print(f'spc.batch_size: {spc.batch_size}')
print(f'spc.lengths (cells per batch entry): {spc.lengths}')

# Batched fields can be accessed directly
print(f'spc.pyramids is of shape (#num_entries, 2, lod+2): {spc.pyramids.shape}')

# Packed fields are flat, and require extra consideration when accessing individual octree entries.
# (boundaries must be tracked to separate the batched Structured Point Clouds)
print(f'spc.point_hierarchies is of shape (#num_entries x total points, 3): {spc.point_hierarchies.shape}')

# Occasionally, you may come across unbatched operations, which take as an argument a single spc entry.
# In such cases, kaolin supports iteration over the spc structure.
print('Example of iteration:')
query_points = torch.tensor([[1,1,0], [1,0,1]], device='cuda', dtype=torch.short)
for idx, single_spc in enumerate(spc):
    # Invoke the unbatched operation per entry, this is of course slower.
    # Always prefer a batched operations over an unbatched ones, if an implementation is available!
    query_results = kaolin.ops.spc.unbatched_query(single_spc.octrees, single_spc.exsum, query_points, level=2)
    print(f'Spc #{idx+1}: unbatched query results are {query_results}')

# kaolin also supports indexing of specific spc entries
print('Example of indexing:')
query_results = kaolin.ops.spc.unbatched_query(single_spc.octrees, single_spc.exsum, query_points, level=lod)
print(f'SPC #2: {spc[1]}')
print(f'Spc #2: unbatched query results are {query_results}')
