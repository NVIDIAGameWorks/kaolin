# ==============================================================================================================
# The following snippet demonstrates how to use the camera for implementing a ray-generation function
# for ray based applications.
# ==============================================================================================================

import torch
import numpy as np
from kaolin.render.camera import Camera, \
    generate_rays, generate_pinhole_rays, \
    generate_centered_pixel_coords, generate_centered_custom_resolution_pixel_coords

camera = Camera.from_args(
    eye=torch.tensor([4.0, 4.0, 4.0]),
    at=torch.tensor([0.0, 0.0, 0.0]),
    up=torch.tensor([0.0, 1.0, 0.0]),
    fov=30 * np.pi / 180,  # In radians
    x0=0.0, y0=0.0,
    width=800, height=800,
    near=1e-2, far=1e2,
    dtype=torch.float64,
    device='cuda'
)

# General raygen functiontional version -- will invoke raygen according to the camera lens type
ray_orig, ray_dir = generate_rays(camera)
print(f'Created a ray grid of dimensions: {ray_orig.shape}')
print('Ray origins:')
print(ray_orig)
print('Ray directions:')
print(ray_dir)
print('\n')

# General raygen function OOP version -- can also be invoked directly on the camera object
ray_orig, ray_dir = camera.generate_rays()
print(f'Created a ray grid of dimensions: {ray_orig.shape}')
print('Ray origins:')
print(ray_orig)
print('Ray directions:')
print(ray_dir)
print('\n')

# A specific raygen function can also be invoked directly. You may also add your own custom raygen functions that way
ray_orig, ray_dir = generate_pinhole_rays(camera)
print(f'Created a ray grid of dimensions: {ray_orig.shape}')
print('Ray origins:')
print(ray_orig)
print('Ray directions:')
print(ray_dir)
print('\n')

# By using a custom grid input, other effects like lower resolution images can be supported
height = 200
width = 400
pixel_grid = generate_centered_custom_resolution_pixel_coords(camera.width, camera.height, width, height, camera.device)
ray_orig, ray_dir = generate_pinhole_rays(camera, pixel_grid)
print(f'Created a ray grid of different dimensions from camera image plane resolution: {ray_orig.shape}')
print('Ray origins:')
print(ray_orig)
print('Ray directions:')
print(ray_dir)
print('\n')