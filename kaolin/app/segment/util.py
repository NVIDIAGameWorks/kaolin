import math
import torch
import kaolin.render.camera

def default_camera():
    return kaolin.render.camera.Camera.from_args(
        eye=torch.tensor([0.3, -0.5, 0.6]),
        at=torch.tensor([0.0, 0.0, 0.3]),
        up=torch.tensor([0., 0, 1.]),
        fov=math.pi * 50 / 180, height=512, width=512)

def mask_from_message(mask_orig):
    """ Depending on how it's encoded mask, can be a multi-channel PNG. """
    if len(mask_orig.shape) == 3:  # hwc
        if mask_orig.shape[-1] == 4:
            mask_orig = mask_orig[..., -1]
        else:
            mask_orig = mask_orig.mean(dim=-1)

    return mask_orig