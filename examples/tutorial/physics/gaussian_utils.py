import os
from collections.abc import Sequence
import math
import torch
import kaolin


PHYS_NOTEBOOKS_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

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

def transform_gaussians_lbs(xyz, rotations, raw_scales, skinning_weights, transforms, shs_feat=None):
    with torch.no_grad():
        # N x 4 x 4 = sum((N x H x 1 x 1) * (1 x H x 4 x 4), dim=1)
        per_pt_transforms = torch.sum(skinning_weights.unsqueeze(-1).unsqueeze(-1) * transforms, dim=1)
        # log_tensor(per_pt_transforms, 'pt transforms', logger)
    
        # convert relative transforms to absolute transforms
        per_pt_transforms = per_pt_transforms + torch.eye(4, dtype=per_pt_transforms.dtype,
                                                          device=per_pt_transforms.device).unsqueeze(0)
    
        new_xyz, new_rot, new_scales, new_shs_feat = kaolin.ops.gaussians.transform_gaussians(
            xyz, rotations, raw_scales, per_pt_transforms, shs_feat=shs_feat)
    
        return new_xyz, new_rot, new_scales, new_shs_feat

def concat_gaussians(gaussians):
    from gaussian_renderer import GaussianModel
    assert isinstance(gaussians, Sequence)
    xyz = []
    rotation = []
    scaling = []
    opacity = []
    features_dc = []
    features_rest = []
    max_sh_degree = gaussians[0].max_sh_degree
    for g in gaussians:
        assert isinstance(g, GaussianModel) and g.max_sh_degree == max_sh_degree
        xyz.append(g._xyz)
        rotation.append(g._rotation)
        scaling.append(g._scaling)
        opacity.append(g._opacity)
        features_dc.append(g._features_dc)
        features_rest.append(g._features_rest)
    output = GaussianModel(max_sh_degree)
    output._xyz = torch.cat(xyz, dim=0).float()
    output._rotation = torch.cat(rotation, dim=0).float()
    output._scaling = torch.cat(scaling, dim=0).float()
    output._opacity = torch.cat(opacity, dim=0).float()
    output._features_dc = torch.cat(features_dc, dim=0).float()
    output._features_rest = torch.cat(features_rest, dim=0).float()
    return output

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def quat_wxyz2xyzw(quat):
    return torch.cat([quat[:, 1:], quat[:, :1]], dim=-1)

def quat_xyzw2wxyz(quat):
    return torch.cat([quat[:, -1:], quat[:, :-1]], dim=-1)

inria_fields = ['xyz', 'rotation', 'scaling', 'opacity', 'features_dc', 'features_rest']
usd_fields = ['positions', 'orientations', 'scales', 'opacities', 'sh_coeff']

def inria_to_usd(gaussians): #xyz, rotation, scaling, opacity, features_dc, features_rest):
    return {
        'positions': gaussians._xyz,
        'orientations': quat_wxyz2xyzw(gaussians._rotation),
        'scales': torch.exp(gaussians._scaling),
        'opacities': torch.sigmoid(gaussians._opacity).unsqueeze(-1),
        'sh_coeff': torch.cat([
            gaussians._features_dc,
            gaussians._features_rest
        ], dim=1)
    }

def usd_to_inria(positions, orientations, scales, opacities, sh_coeff, **kwargs):
    from gaussian_renderer import GaussianModel
    degrees = math.isqrt(sh_coeff.shape[1]) - 1
    gaussians = GaussianModel(degrees)
    gaussians._xyz = positions.cuda()
    gaussians._rotation = quat_xyzw2wxyz(orientations).cuda()
    gaussians._scaling = torch.log(scales).cuda()
    gaussians._opacity = inverse_sigmoid(opacities).cuda()
    gaussians._features_dc = sh_coeff[:, :1].cuda()
    gaussians._features_rest = sh_coeff[:, 1:].cuda()
    return gaussians

