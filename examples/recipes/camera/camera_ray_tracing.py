# ==============================================================================================================
# The following snippet demonstrates how to use the camera for implementing a ray-generation function
# for ray based applications.
# ==============================================================================================================

import torch
import numpy as np
from typing import Tuple
from kaolin.render.camera import Camera, CameraFOV

def generate_pixel_grid(res_x=None, res_y=None, device='cuda'):
    h_coords = torch.arange(res_x, device=device)
    w_coords = torch.arange(res_y, device=device)
    pixel_y, pixel_x = torch.meshgrid(h_coords, w_coords)
    pixel_x = pixel_x + 0.5
    pixel_y = pixel_y + 0.5
    return pixel_y, pixel_x


def generate_perspective_rays(camera: Camera, pixel_grid: Tuple[torch.Tensor, torch.Tensor]):
    # coords_grid should remain immutable (a new tensor is implicitly created here)
    pixel_y, pixel_x = pixel_grid
    pixel_x = pixel_x.to(camera.device, camera.dtype)
    pixel_y = pixel_y.to(camera.device, camera.dtype)

    # Account for principal point offset from canvas center
    pixel_x = pixel_x - camera.x0
    pixel_y = pixel_y + camera.y0

    # pixel values are now in range [-1, 1], both tensors are of shape res_y x res_x
    pixel_x = 2 * (pixel_x / camera.width) - 1.0
    pixel_y = 2 * (pixel_y / camera.height) - 1.0

    ray_dir = torch.stack((pixel_x * camera.tan_half_fov(CameraFOV.HORIZONTAL),
                           -pixel_y * camera.tan_half_fov(CameraFOV.VERTICAL),
                           -torch.ones_like(pixel_x)), dim=-1)

    ray_dir = ray_dir.reshape(-1, 3)    # Flatten grid rays to 1D array
    ray_orig = torch.zeros_like(ray_dir)

    # Transform from camera to world coordinates
    ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)
    ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
    ray_orig, ray_dir = ray_orig[0], ray_dir[0]  # Assume a single camera

    return ray_orig, ray_dir, camera.near, camera.far


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

pixel_grid = generate_pixel_grid(200, 200)
ray_orig, ray_dir, near, far = generate_perspective_rays(camera, pixel_grid)

print('Ray origins:')
print(ray_orig)
print('Ray directions:')
print(ray_dir)
print('Near clipping plane:')
print(near)
print('Far clipping plane:')
print(far)
