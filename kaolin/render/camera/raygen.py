import torch
from typing import Optional
from kaolin.render.camera import Camera, CameraFOV


__all__ = [
    'generate_default_grid',
    'generate_centered_pixel_coords',
    'generate_centered_custom_resolution_pixel_coords',
    'generate_pinhole_rays',
    'generate_ortho_rays',
    'generate_rays'
]


def generate_default_grid(width, height, device=None):
    r""" Creates a pixel grid of integer coordinates with resolution width x height.

    Args:
        width (int): width of image.
        height (int): height of image.
        device (torch.device, optional): Device on which the meshgrid tensors will be created.

    Returns:
        meshgrid (torch.FloatTensor, torch.FloatTensor):
            A tuple of two tensors of shapes :math:`(\text{height, width})`.

            Tensor 0 contains rows of running indices:
            :math:`(\text{0, 0, ..., 0})` up to
            :math:`(\text{height-1, height-1... height-1})`.

            Tensor 1 contains repeated rows of indices:
            :math:`(\text{0, 1, ..., width-1})`.
    """
    h_coords = torch.arange(height, device=device, dtype=torch.float)
    w_coords = torch.arange(width, device=device, dtype=torch.float)
    return torch.meshgrid(h_coords, w_coords, indexing='ij')  # return pixel_y, pixel_x


def generate_centered_pixel_coords(img_width, img_height, device=None):
    r""" Creates a pixel grid with rays intersecting the center of each pixel.
    The ray grid is of resolution img_width x img_height.

    Args:
        img_width (int): width of image.
        img_height (int): height of image.
        device (torch.device, optional): Device on which the grid tensors will be created.

    Returns:
        meshgrid (torch.FloatTensor, torch.FloatTensor):
        A tuple of two tensors of shapes :math:`(\text{height, width})`.

        Tensor 0 contains rows of running indices:
        :math:`(\text{0.5, 0.5, ..., 0.5})` up to
        :math:`(\text{height-0.5, height-0.5... height-0.5})`.

        Tensor 1 contains repeated rows of indices:
        :math:`(\text{0.5, 1.5, ..., width-0.5})`.
    """
    pixel_y, pixel_x = generate_default_grid(img_width, img_height, device)
    pixel_x = pixel_x + 0.5   # add bias to pixel center
    pixel_y = pixel_y + 0.5   # add bias to pixel center
    return pixel_y, pixel_x


def generate_centered_custom_resolution_pixel_coords(img_width, img_height, res_x=None, res_y=None, device=None):
    r""" Creates a pixel grid with a custom resolution, with the rays spaced out according to the scale.
    The scale is determined by the ratio of :math:`\text{img_width / res_x, img_height / res_y}`.
    The ray grid is of resolution :math:`\text{res_x} \times \text{res_y}`.

    Args:
        img_width (int): width of camera image plane.
        img_height (int): height of camera image plane.
        res_x (int): x resolution of pixel grid to be created
        res_y (int): y resolution of pixel grid to be created
        device (torch.device, optional): Device on which the grid tensors will be created.

    Returns:
        meshgrid (torch.FloatTensor, torch.FloatTensor):
        A tuple of two tensors of shapes :math:`(\text{height, width})`.

        Tensor 0 contains rows of running indices:
        :math:`(\text{s, s, ..., s})` up to
        :math:`(\text{height-s, height-s... height-s})`.

        Tensor 1 contains repeated rows of indices:
        :math:`(\text{s, s+1, ..., width-s})`.

        :math:`\text{s}` is :math:`\text{scale/2}` where :math:`\text{scale}` is
        :math:`(\text{img_width / res_x, img_height, res_y})`.
    """
    if res_x is None:
        res_x = img_width
    if res_y is None:
        res_y = img_height
    scale_x = img_width / res_x
    scale_y = img_height / res_y
    pixel_y, pixel_x = generate_default_grid(res_x, res_y, device)
    pixel_x = scale_x * pixel_x + (scale_x / 2.0)   # add bias to pixel center
    pixel_y = scale_y * pixel_y + (scale_y / 2.0)   # add bias to pixel center
    return pixel_y, pixel_x

# -- Ray gen --
def _to_ndc_coords(pixel_x, pixel_y, camera):
    pixel_x = 2 * (pixel_x / camera.width) - 1.0
    pixel_y = 2 * (pixel_y / camera.height) - 1.0
    return pixel_x, pixel_y


def generate_pinhole_rays(camera: Camera, coords_grid: Optional[torch.Tensor] = None):
    r"""Default ray generation function for pinhole cameras.

    This function assumes that the principal point (the pinhole location) is specified by a 
    displacement (camera.x0, camera.y0) in pixel coordinates from the center of the image. 

    The Kaolin camera class does not enforce a coordinate space for how the principal point is specified,
    so users will need to make sure that the correct principal point conventions are followed for 
    the cameras passed into this function.

    Args:
        camera (kaolin.render.camera.Camera): A single camera object (batch size 1).
        coords_grid (torch.FloatTensor, optional):
            Pixel grid of ray-intersecting coordinates of shape :math:`(\text{H, W, 2})`.
            Coordinates integer parts represent the pixel :math:`(\text{i, j})` coords, and the fraction part of
            :math:`[\text{0,1}]` represents the location within the pixel itself.
            For example, a coordinate of :math:`(\text{0.5, 0.5})` represents the center of the top-left pixel.

    Returns:
        (torch.FloatTensor, torch.FloatTensor):
            The generated pinhole rays for the camera, as ray origins and ray direction tensors of
            :math:`(\text{HxW, 3})`.
    """
    assert len(camera) == 1, "generate_pinhole_rays() supports only camera input of batch size 1"
    if coords_grid is None:
        coords_grid = generate_centered_pixel_coords(camera.width, camera.height, device=camera.device)
    else:
        assert camera.device == coords_grid[0].device, \
            f"Expected camera and coords_grid[0] to be on the same device, " \
            f"but found {camera.device} and {coords_grid[0].device}."

        assert camera.device == coords_grid[1].device, \
            f"Expected camera and coords_grid[1] to be on the same device, " \
            f"but found {camera.device} and {coords_grid[1].device}."

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

    return ray_orig, ray_dir


def generate_ortho_rays(camera: Camera, coords_grid: Optional[torch.Tensor] = None):
    r"""Default ray generation function for ortho cameras.

    Args:
        camera (kaolin.render.camera.Camera): A single camera object (batch size 1).
        coords_grid (torch.FloatTensor, optional):
            Pixel grid of ray-intersecting coordinates of shape :math:`(\text{H, W, 2})`.
            Coordinates integer parts represent the pixel :math:`(\text{i, j})` coords, and the fraction part of
            :math:`[\text{0,1}]` represents the location within the pixel itself.
            For example, a coordinate of :math:`(\text{0.5, 0.5})` represents the center of the top-left pixel.

    Returns:
        (torch.FloatTensor, torch.FloatTensor):
            The generated ortho rays for the camera, as ray origins and ray direction tensors of
            :math:`(\text{HxW, 3})` .
    """
    assert len(camera) == 1, "generate_ortho_rays() supports only camera input of batch size 1"
    if coords_grid is None:
        coords_grid = generate_centered_pixel_coords(camera.width, camera.height, device=camera.device)
    else:
        assert camera.device == coords_grid[0].device, \
            f"Expected camera and coords_grid[0] to be on the same device, " \
            f"but found {camera.device} and {coords_grid[0].device}."

        assert camera.device == coords_grid[1].device, \
            f"Expected camera and coords_grid[1] to be on the same device, " \
            f"but found {camera.device} and {coords_grid[1].device}."

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

    return ray_orig, ray_dir


def generate_rays(camera, coords_grid: Optional[torch.Tensor] = None):
    r"""Default ray generation function for unbatched kaolin cameras.
    The camera lens type will determine the exact raygen logic that runs (i.e. pinhole, ortho..)

    Args:
        camera (kaolin.render.camera.Camera): A single camera object (batch size 1).
        coords_grid (torch.FloatTensor, optional):
            Pixel grid of ray-intersecting coordinates of shape :math:`(\text{H, W, 2})`.
            Coordinates integer parts represent the pixel :math:`(\text{i, j})` coords, and the fraction part of
            :math:`[\text{0,1}]` represents the location within the pixel itself.
            For example, a coordinate of :math:`(\text{0.5, 0.5})` represents the center of the top-left pixel.

    Returns:
        (torch.FloatTensor, torch.FloatTensor):
            The generated camera rays according to the camera lens type, as ray origins and ray direction tensors of
            :math:`(\text{HxW, 3})`.
    """
    if camera.lens_type == 'pinhole':
        return generate_pinhole_rays(camera, coords_grid)
    elif camera.lens_type == 'ortho':
        return generate_ortho_rays(camera, coords_grid)
    else:
        raise NotImplementedError(f'generate_rays does not support camera type: {camera.lens_type}')
