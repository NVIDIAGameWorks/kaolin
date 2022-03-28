# ==============================================================================================================
# The following snippet demonstrates various camera properties
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

print(camera.width)
print(camera.height)
print(camera.lens_type)

print(camera.device)
camera = camera.cpu()
print(camera.device)

# Create a batched camera and view single element
camera = Camera.cat((camera, camera))
print(camera)
camera = camera[0]
print(camera)

print(camera.dtype)
camera = camera.half()
print(camera.dtype)
camera = camera.double()
print(camera.dtype)
camera = camera.float()
print(camera.dtype)

print(camera.extrinsics.requires_grad)
print(camera.intrinsics.requires_grad)

print(camera.to('cuda', torch.float64))
