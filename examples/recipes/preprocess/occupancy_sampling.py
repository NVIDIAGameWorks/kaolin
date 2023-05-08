# ==============================================================================================================
# The following snippet shows how to use kaolin to test sampled values of an occupancy function
# against a watertight mesh.
# ==============================================================================================================
# See also:
#  - Documentation: Triangular meshes
#    https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.mesh.html#triangular-meshes
# ==============================================================================================================

import os
import torch
import kaolin

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
mesh_path = os.path.join(FILE_DIR, os.pardir, os.pardir, "samples", "sphere.obj")   # Path to some .obj file with textures
num_samples = 100000                    # Number of sample points

# 1. Load a watertight mesh from obj file
mesh = kaolin.io.obj.import_mesh(mesh_path)
print(f'Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.')

# 2. Preprocess mesh:
# Move tensors to CUDA device
vertices = mesh.vertices.cuda()
faces = mesh.faces.cuda()

# Kaolin assumes an exact batch format, we make sure to convert from: (V, 3) to (1, V, 3), where 1 is the batch size
vertices = vertices.unsqueeze(0)

# 3. Sample random points uniformly in space, from the bounding box of the mesh + 10% margin
min_bound, _ = vertices.min(dim=1)
max_bound, _ = vertices.max(dim=1)
margin = (max_bound - min_bound) * 0.1
max_bound += margin
min_bound -= margin
occupancy_coords = (max_bound - min_bound) * torch.rand(1, num_samples, 3, device='cuda') + min_bound

# 4. Calculate occupancy value
occupancy_value = kaolin.ops.mesh.check_sign(vertices, faces, occupancy_coords)

# Unbatch to obtain a torch.Tensor of (V, 3) and (V, 1)
occupancy_coords = occupancy_coords.squeeze(0)
occupancy_value = occupancy_value.squeeze(0)

percent_in_mesh = torch.count_nonzero(occupancy_value) / len(occupancy_value)
print(f'Sampled a tensor of points uniformly in space '
      f'with {occupancy_coords.shape[0]} points of {occupancy_coords.shape[1]}D coordinates.')
print(f'{"{:.3f}".format(percent_in_mesh)}% of the sampled points are inside the mesh volume.')
