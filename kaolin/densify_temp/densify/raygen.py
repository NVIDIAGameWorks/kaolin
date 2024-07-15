import torch
from kaolin.render.camera import Camera, CameraFOV


def generate_default_grid(width, height, device=None):
    h_coords = torch.arange(height, device=device, dtype=torch.float)
    w_coords = torch.arange(width, device=device, dtype=torch.float)
    return torch.meshgrid(h_coords, w_coords)  # return pixel_y, pixel_x

def generate_centered_pixel_coords(img_width, img_height, res_x=None, res_y=None, device=None):
    pixel_y, pixel_x = generate_default_grid(res_x, res_y, device)
    scale_x = 1.0 if res_x is None else float(img_width) / res_x
    scale_y = 1.0 if res_y is None else float(img_height) / res_y
    pixel_x = pixel_x * scale_x + 0.5   # scale and add bias to pixel center
    pixel_y = pixel_y * scale_y + 0.5   # scale and add bias to pixel center
    return pixel_y, pixel_x


# -- Ray gen --
def _to_ndc_coords(pixel_x, pixel_y, camera):
    pixel_x = 2 * (pixel_x / camera.width) - 1.0
    pixel_y = 2 * (pixel_y / camera.height) - 1.0
    return pixel_x, pixel_y

def generate_pinhole_rays(camera: Camera, coords_grid: torch.Tensor):
    """Default ray generation function for pinhole cameras.

    This function assumes that the principal point (the pinhole location) is specified by a 
    displacement (camera.x0, camera.y0) in pixel coordinates from the center of the image. 

    The Kaolin camera class does not enforce a coordinate space for how the principal point is specified,
    so users will need to make sure that the correct principal point conventions are followed for 
    the cameras passed into this function.

    Args:
        camera (kaolin.render.camera): The camera class. 
        coords_grid (torch.FloatTensor): Grid of coordinates of shape [H, W, 2].

    Returns:
        (wisp.core.Rays): The generated pinhole rays for the camera.
    """
    if camera.device != coords_grid[0].device:
        raise Exception(f"Expected camera and coords_grid[0] to be on the same device, but found {camera.device} and {coords_grid[0].device}.")
    if camera.device != coords_grid[1].device:
        raise Exception(f"Expected camera and coords_grid[1] to be on the same device, but found {camera.device} and {coords_grid[1].device}.")
    # coords_grid should remain immutable (a new tensor is implicitly created here)
    pixel_y, pixel_x = coords_grid
    pixel_x = pixel_x.to(camera.device, camera.dtype)
    pixel_y = pixel_y.to(camera.device, camera.dtype)

    # Account for principal point (offsets from the center)
    pixel_x = pixel_x - camera.x0
    pixel_y = pixel_y + camera.y0

    # pixel values are now in range [-1, 1], both tensors are of shape res_y x res_x
    pixel_x, pixel_y = _to_ndc_coords(pixel_x, pixel_y, camera)

    ray_dir = torch.stack((pixel_x * camera.tan_half_fov(CameraFOV.HORIZONTAL),
                           -pixel_y * camera.tan_half_fov(CameraFOV.VERTICAL),
                           -torch.ones_like(pixel_x)), dim=-1)

    ray_dir = ray_dir.reshape(-1, 3)    # Flatten grid rays to 1D array
    ray_orig = torch.zeros_like(ray_dir)

    # Transform from camera to world coordinates
    ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)
    ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
    ray_orig, ray_dir = ray_orig[0], ray_dir[0]  # Assume a single camera

    return {ray_orig, ray_dir}




def generate_ortho_rays(camera: Camera, coords_grid: torch.Tensor):
    # coords_grid should remain immutable (a new tensor is implicitly created here)
    pixel_y, pixel_x = coords_grid
    pixel_y = pixel_y.to(camera.device, camera.dtype)
    pixel_x = pixel_x.to(camera.device, camera.dtype)

    # pixel values are now in range [-1, 1], both tensors are of shape res_y x res_x
    pixel_x, pixel_y = _to_ndc_coords(pixel_x, pixel_y, camera)

    # Rescale according to distance from camera
    aspect_ratio = camera.width / camera.height
    pixel_x *= camera.fov_distance * aspect_ratio
    pixel_y *= camera.fov_distance

    zeros = torch.zeros_like(pixel_x)
    ray_dir = torch.stack((zeros, zeros, -torch.ones_like(pixel_x)), dim=-1)    # Ortho rays are parallel
    ray_orig = torch.stack((pixel_x, -pixel_y, zeros), dim=-1)
    ray_dir = ray_dir.reshape(-1, 3)    # Flatten grid rays to 1D array
    ray_orig = ray_orig.reshape(-1, 3)  # Flatten grid rays to 1D array

    # Transform from camera to world coordinates
    ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)
    ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
    ray_orig, ray_dir = ray_orig[0], ray_dir[0]  # Assume a single camera

    return {ray_orig, ray_dir}

