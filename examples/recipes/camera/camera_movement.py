# ==============================================================================================================
# The following snippet demonstrates how to manipulate kaolin's camera.
# ==============================================================================================================

import torch
from kaolin.render.camera import Camera


camera = Camera.from_args(
    eye=torch.tensor([0.0, 0.0, -1.0]),
    at=torch.tensor([0.0, 0.0, 0.0]),
    up=torch.tensor([0.0, 1.0, 0.0]),
    width=800, height=600,
    fov=1.0,
    device='cuda'
)

# Extrinsic rigid transformations managed by CameraExtrinsics
camera.move_forward(amount=10.0)               # Translate forward in world coordinates (this is wisp's mouse zoom)
camera.move_right(amount=-5.0)                 # Translate left in world coordinates
camera.move_up(amount=5.0)                     # Translate up in world coordinates
camera.rotate(yaw=0.1, pitch=0.02, roll=1.0)   # Rotate the camera

# Intrinsic lens transformations managed by CameraIntrinsics
# Zoom in to decrease field of view - for Orthographic projection the internal implementation differs
# as there is no acual fov or depth concept (hence we use a "made up" fov distance parameter, see the projection matrix)
camera.zoom(amount=0.5)
