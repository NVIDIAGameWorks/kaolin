import json
import math
import torch
import kaolin.render.camera

def up_axis_to_tensor(up_axis: str) -> torch.Tensor:
    """Convert an up-axis string (x, y, z, -x, -y, -z) to a unit vector tensor."""
    mapping = {
        'x':  [1., 0., 0.], '-x': [-1., 0., 0.],
        'y':  [0., 1., 0.], '-y': [0., -1., 0.],
        'z':  [0., 0., 1.], '-z': [0., 0., -1.],
    }
    up_axis = up_axis.lower()
    if up_axis not in mapping:
        raise ValueError(f'Unknown up-axis "{up_axis}". Choose from: {list(mapping)}')
    return torch.tensor(mapping[up_axis])


def default_camera(up: torch.Tensor = None):
    if up is None:
        up = torch.tensor([0., 1., 0.])
    return kaolin.render.camera.Camera.from_args(
        eye=torch.tensor([0.3, -0.5, 0.6]),
        at=torch.tensor([0.0, 0.0, 0.3]),
        up=up,
        fov=math.pi * 50 / 180, height=512, width=512)

def load_cameras_json(path, device='cpu'):
    """Load cameras from an Inria 3DGS cameras.json file.

    Each entry stores rotation as the camera-to-world rotation matrix (row-major)
    and position as the camera center in world space.
    """
    with open(path) as f:
        cam_list = json.load(f)
    cameras = []
    for cam in cam_list:
        R_c2w = torch.tensor(cam['rotation'], dtype=torch.float32)  # (3,3) stored row-major
        pos = torch.tensor(cam['position'], dtype=torch.float32)
        # Columns are camera axes in world (OpenCV: X=right, Y=down, Z=forward)
        forward = R_c2w[:, 2]
        up = -R_c2w[:, 1]
        cameras.append(kaolin.render.camera.Camera.from_args(
            eye=pos, at=pos + forward, up=up,
            focal_x=float(cam['fx']), focal_y=float(cam['fy']),
            width=int(cam['width']), height=int(cam['height']),
            device=device,
        ))
    return cameras


def mask_from_message(mask_orig):
    """ Depending on how it's encoded mask, can be a multi-channel PNG. """
    if len(mask_orig.shape) == 3:  # hwc
        if mask_orig.shape[-1] == 4:
            mask_orig = mask_orig[..., -1]
        else:
            mask_orig = mask_orig.mean(dim=-1)

    return mask_orig