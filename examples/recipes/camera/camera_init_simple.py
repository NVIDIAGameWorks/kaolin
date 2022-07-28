# ==============================================================================================================
# The following snippet demonstrates how to initialize instances of kaolin's pinhole / ortho cameras.
# ==============================================================================================================

import math
import torch
import numpy as np
from kaolin.render.camera import Camera

device = 'cuda'

perspective_camera_1 = Camera.from_args(
    eye=torch.tensor([4.0, 4.0, 4.0]),
    at=torch.tensor([0.0, 0.0, 0.0]),
    up=torch.tensor([0.0, 1.0, 0.0]),
    fov=30 * np.pi / 180,  # In radians
    x0=0.0, y0=0.0,
    width=800, height=800,
    near=1e-2, far=1e2,
    dtype=torch.float64,
    device=device
)

print('--- Perspective Camera 1 ---')
print(perspective_camera_1)

perspective_camera_2 = Camera.from_args(
    eye=torch.tensor([4.0, 4.0, 4.0]),
    at=torch.tensor([0.0, 0.0, 0.0]),
    up=torch.tensor([0.0, 1.0, 0.0]),
    fov=30 * np.pi / 180,  # In radians
    width=800, height=800,
    device=device
)

print('--- Perspective Camera 2 ---')
print(perspective_camera_2)

ortho_camera_1 = Camera.from_args(
    eye=torch.tensor([4.0, 4.0, 4.0]),
    at=torch.tensor([0.0, 0.0, 0.0]),
    up=torch.tensor([0.0, 1.0, 0.0]),
    width=800, height=800,
    near=-800, far=800,
    fov_distance=1.0,
    dtype=torch.float64,
    device=device
)

print('--- Orthographic Camera 1 ---')
print(ortho_camera_1)


ortho_camera_2 = Camera.from_args(
    view_matrix=torch.tensor([[1.0, 0.0, 0.0, 0.5],
                              [0.0, 1.0, 0.0, 0.5],
                              [0.0, 0.0, 1.0, 0.5],
                              [0.0, 0.0, 0.0, 1.0]]),
    width=800, height=800,
    dtype=torch.float64,
    device=device
)

print('--- Orthographic Camera 2 ---')
print(ortho_camera_2)
