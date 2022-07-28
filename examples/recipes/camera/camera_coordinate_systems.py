# ==============================================================================================================
# The following snippet demonstrates how to change the coordinate system of the camera.
# ==============================================================================================================

import math
import torch
import numpy as np
from kaolin.render.camera import Camera, blender_coords

device = 'cuda'

camera = Camera.from_args(
    eye=torch.tensor([4.0, 4.0, 4.0]),
    at=torch.tensor([0.0, 0.0, 0.0]),
    up=torch.tensor([0.0, 1.0, 0.0]),
    fov=30 * np.pi / 180,  # In radians
    width=800, height=800,
    device=device
)

print(camera.basis_change_matrix)
camera.change_coordinate_system(blender_coords())
print(camera.basis_change_matrix)
camera.reset_coordinate_system()
print(camera.basis_change_matrix)
