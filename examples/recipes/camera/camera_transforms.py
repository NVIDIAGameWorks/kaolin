# ==============================================================================================================
# The following snippet demonstrates how to use the camera transform directly on vectors
# ==============================================================================================================

import math
import torch
import numpy as np
from kaolin.render.camera import Camera

device = 'cuda'

camera = Camera.from_args(
    eye=torch.tensor([4.0, 4.0, 4.0]),
    at=torch.tensor([0.0, 0.0, 0.0]),
    up=torch.tensor([0.0, 1.0, 0.0]),
    fov=30 * np.pi / 180,  # In radians
    width=800, height=800,
    dtype=torch.float32,
    device=device
)

print('View projection matrix')
print(camera.view_projection_matrix())

print('View matrix: world2cam')
print(camera.view_matrix())

print('Inv View matrix: cam2world')
print(camera.inv_view_matrix())

print('Projection matrix')
print(camera.projection_matrix())

vectors = torch.randn(10, 3).to(camera.device, camera.dtype)   # Create a batch of points

# For ortho and perspective: this is equivalent to multiplying camera.projection_matrix() @ vectors
# and then dividing by the w coordinate (perspective division)
print(camera.transform(vectors))

# For ray tracing we have camera.inv_transform_rays which performs multiplication with inv_view_matrix()
# (just for the extrinsics part)

# Can also access properties directly:
# --
# View matrix components (camera space)
print(camera.R)
print(camera.t)

# Camera axes and position in world coordinates
print(camera.cam_pos())
print(camera.cam_right())
print(camera.cam_pos())
print(camera.cam_forward())

print(camera.focal_x)
print(camera.focal_y)
print(camera.x0)
print(camera.y0)

