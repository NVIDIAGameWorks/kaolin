import os
import torch
import kaolin


PHYS_NOTEBOOKS_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

# TODO(shumash): all of these should move to core library; address in v0.19.0
def transform_xyz(xyz: torch.Tensor, transform: torch.Tensor):
    if len(transform.shape) == 2:  # single transform for all the gaussians
        transform = transform.unsqueeze(0)
    res = (transform[..., :3, :3] @ xyz[:, :, None] + transform[..., :3, 3:]).squeeze(-1)
    return res


def transform_rot(rot: torch.Tensor, transform: torch.Tensor):
    if len(transform.shape) == 2:  # single transform for all the gaussians
        transform = transform.unsqueeze(0)

    rot_quat = kaolin.math.quat.quat_from_rot33(transform[..., :3, :3])
    rot_unit = rot / torch.linalg.norm(rot, dim=-1).unsqueeze(-1)

    # Note: gsplats use Hamiltonion convention [real, imag], whereas Kaolin uses the other convention[imag, real]
    rot_unit = torch.cat([rot_unit[:, 1:], rot_unit[:, :1]], dim=-1)

    result = kaolin.math.quat.quat_mul(rot_quat, rot_unit)
    result = torch.cat([result[:, 3:], result[:, :3]], dim=-1)
    return result


def decompose_4x4_transform(transform):
    """ Decompose 4x4 transform into translation, rotation, scale.
    Returns:
        translation, rotation, scale
    """
    translation = transform[..., :3, 3:]
    scale = torch.linalg.norm(transform[..., :3, :3], dim=-2)
    rotation = transform[..., :3, :3] / scale.unsqueeze(-2)

    return translation, rotation, scale


def transform_gaussians(xyz, rotations, raw_scales, transform, use_log_scales=True):
    if len(transform.shape) == 2:  # single transform for all the gaussians
        transform = transform.unsqueeze(0)

    # transforms: n x 4 x 4, where 4 x 4 transform T is applied to pt as T @ pt.
    translation, rotation, scale = decompose_4x4_transform(transform)

    new_xyz = transform_xyz(xyz, transform)
    new_rotations = transform_rot(rotations, rotation)

    if not use_log_scales:
        new_scales = raw_scales * scale
    else:
        scaling_norm_factor = torch.log(scale) / raw_scales + 1
        new_scales = raw_scales * scaling_norm_factor

    return new_xyz, new_rotations, new_scales

def pad_transforms(obj_tfms):
    """
    Args:
        obj_tfms: N x 3 x 4

    Returns:
        same transforms but N x 4 x 4
    """
    padding_row = torch.tensor([[0., 0., 0., 1.]], device=obj_tfms.device).expand(obj_tfms.shape[0], -1, -1)
    padded_tensor = torch.cat([obj_tfms, padding_row], dim=1)
    return padded_tensor

def transform_gaussians_lbs(xyz, rotations, raw_scales, skinning_weights, transforms):
    # N x 4 x 4 = sum((N x H x 1 x 1) * (1 x H x 4 x 4), dim=1)
    per_pt_transforms = torch.sum(skinning_weights.unsqueeze(-1).unsqueeze(-1) * transforms, dim=1)
    # log_tensor(per_pt_transforms, 'pt transforms', logger)

    # convert relative transforms to absolute transforms
    per_pt_transforms = per_pt_transforms + torch.eye(4, dtype=per_pt_transforms.dtype,
                                                      device=per_pt_transforms.device).unsqueeze(0)

    new_xyz, new_rot, new_scales = transform_gaussians(xyz, rotations, raw_scales, per_pt_transforms)

    return new_xyz, new_rot, new_scales