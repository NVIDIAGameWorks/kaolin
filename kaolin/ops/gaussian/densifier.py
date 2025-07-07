# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import torch
import logging
from typing import Optional, Callable
from kaolin.ops.random import sample_spherical_coords
from kaolin.ops.conversions import gs_to_voxelgrid
from kaolin.ops.spc import scan_octrees, morton_to_points, bf_recon, unbatched_points_to_octree
from kaolin.ops.spc.raytraced_spc_dataset import RayTracedSPCDataset
from kaolin.ops.spc.bf_recon import bf_recon, unbatched_query
from kaolin import _C

logger = logging.getLogger(__name__)

__all__ = [
    'sample_points_in_volume'
]


def _generate_default_viewpoints():
    r""" Generates a collection of default viewpoints used to 'carve' out seen space.
    These anchors are chosen based on empirical heuristics
    """
    anchors = torch.tensor([
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0],
        [-4.0, 0.0, 0.0],
        [0.0, -4.0, 0.0],
        [0.0, 0.0, -4.0],
        [2.3, 2.3, 2.3],
        [-2.3, 2.3, 2.3],
        [2.3, -2.3, 2.3],
        [2.3, 2.3, -2.3],
        [-2.3, -2.3, 2.3],
        [-2.3, 2.3, -2.3],
        [2.3, -2.3, -2.3],
        [-2.3, -2.3, -2.3]
    ])

    phi = (1 + math.sqrt(5.0)) / 2
    icosahedron = torch.tensor([
        [+phi, +1.0, 0.0],
        [+phi, -1.0, 0.0],
        [-phi, -1.0, 0.0],
        [-phi, +1.0, 0.0],
        [+1.0, 0.0, +phi],
        [-1.0, 0.0, +phi],
        [-1.0, 0.0, -phi],
        [+1.0, 0.0, -phi],
        [0.0, +phi, +1.0],
        [0.0, +phi, -1.0],
        [0.0, -phi, -1.0],
        [0.0, -phi, +1.0]
    ])

    # Degrees to radians
    deg_to_rad = torch.pi / 180.0

    # Rotation angles
    theta_x = 15 * deg_to_rad
    theta_y = 27 * deg_to_rad
    theta_z = 49 * deg_to_rad

    # Rotation matrix
    R_x = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta_x), -np.sin(theta_x)],
        [0.0, np.sin(theta_x), np.cos(theta_x)]]
    )
    R_y = torch.tensor([
        [np.cos(theta_y), 0.0, np.sin(theta_y)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta_y), 0.0, np.cos(theta_y)]])
    R_z = torch.tensor([
        [np.cos(theta_z), -np.sin(theta_z), 0.0],
        [np.sin(theta_z), np.cos(theta_z), 0.0],
        [0.0, 0.0, 1.0]]
    )
    R = (R_z @ R_y @ R_x).unsqueeze(0).float()

    viewpoints = torch.cat([
        anchors,
        icosahedron,
        (R @ (2.0 * icosahedron)[:, :, None]).squeeze(-1),
        (R @ R @ (3.0 * icosahedron)[:, :, None]).squeeze(-1),
        (R @ R @ R @ (4.0 * icosahedron)[:, :, None]).squeeze(-1),
        (R @ R @ R @ R @ (5.0 * icosahedron)[:, :, None]).squeeze(-1),
        (R @ R @ R @ R @ R @ (6.0 * icosahedron)[:, :, None]).squeeze(-1),
    ], dim=0)
    delta = 0.001*(0.5 - torch.rand(3*viewpoints.size(0))).reshape(-1,3)
    return viewpoints + delta


def _jitter(pts, octree_level):
    r""" Applies random perturbations to a set of voxelized points.
    The perturbations are small enough such that each point remains in the voxel cell it belongs in.
    """
    N = pts.shape[0]
    device = pts.device
    dtype = pts.dtype
    cell_radius = 2.0 / (2.0 ** octree_level)  # octree spans from [-1,1], number of cells due to octree level
    radius = cell_radius * torch.sqrt(torch.rand(N, device=device))

    azimuth, elevation = sample_spherical_coords((N,),
        azimuth_low=0., azimuth_high=math.pi * 2., elevation_low=-math.pi * .5, elevation_high=math.pi * .5,
        device=device, dtype=dtype
    )

    # spherical coordinates to cartesian
    x = radius * torch.sin(elevation) * torch.cos(azimuth)
    y = radius * torch.sin(elevation) * torch.sin(azimuth)
    z = radius * torch.cos(elevation)
    delta = torch.stack([x,y,z], dim=1)
    return pts + delta

def _solidify(xyz, scales, rots, opacities, opacity_threshold, gs_level, query_level, viewpoints, scaling_activation, scaling_inverse_activation):
    r"""Creates a tensor of uniform samples 'inside' collection of Gaussian Splats.

    Args:
        xyz (torch.FloatTensor) : Gaussian Splat means, of shape :math:`(\text{num_gaussians, 3})`.
        scales (torch.FloatTensor) : Gaussian Splat scales, of shape :math:`(\text{num_gaussians, 3})`.
        rots (torch.FloatTensor) : Gaussian Splat rots, of shape :math:`(\text{num_gaussians, 4})`.
        opacities (torch.FloatTensor) : Gaussian Splat opacities, of shape :math:`(\text{num_gaussians})`.
        opacity_threshold (float): Threshold to cull away voxelized cells with low accumulated opacity.
        gs_level (int): The level of the interal octree created.
        query_level (int): The level of the uniform sample grid.
        viewpoints (optional, torch.Tensor):
            Collection of viewpoints used to 'carve' out seen voxel space around the shell after it's voxelized.
            These is a :math:`(\text{C, 3})` tensor of camera viewpoints facing the center,
            chosen based on empirical heuristics.
        scaling_activation (Callable): activation applied on scaling.
        scaling_inverse_activation (Callable): inverse of activation applied on scaling.


    Returns:
        pidx (torch.LongTensor):

            The indices into the point hierarchy of shape :math:`(\text{num_query})`.
            If with_parents is True, then the shape will be :math:`(\text{num_query, level+1})`.

    """
    device = xyz.device

    # AABB of Gaussian means
    pmin = torch.min(xyz, dim=0, keepdim=False)[0]
    pmax = torch.max(xyz, dim=0, keepdim=False)[0]

    # find the AABB diagonal vector and centroid
    diff = pmax - pmin
    cen = (0.5 * (pmin + pmax))

    # find the maximum diagonal component, add tiny amount to compensate for covariance vectors (a hack!)
    dmax = (0.5 * torch.max(diff) + 0.05)

    # transform Gaussians to [-1,-1,-1]x[1,1,1]
    xyz = (xyz - cen) / dmax
    raw_scales = scaling_inverse_activation(scales)
    scales = scaling_activation(
        scaling_inverse_activation(1. / dmax).unsqueeze(0) + raw_scales
    )

    # some constants
    scale_voxel_tolerance = 0.125
    iso = 11.345  # 99th percentile

    # compute spc from gsplats
    voxels, merged_opacities = gs_to_voxelgrid(
        xyz, scales, rots, opacities, gs_level, iso=iso, tol=scale_voxel_tolerance, step=10
    )

    # filter out low opacities
    mask = merged_opacities[:] >= opacity_threshold
    voxels = voxels[mask].contiguous()
    gs_octree = unbatched_points_to_octree(voxels, gs_level)

    # create depthmaps
    dataset = RayTracedSPCDataset(viewpoints.contiguous(), gs_octree.contiguous())

    # fuse depthmaps into seen/unseen aware spc
    bf_octree, bf_empty, _, _ = bf_recon(dataset, final_level=query_level, sigma=0.0005)
    if bf_octree is None or bf_empty is None or len(bf_octree) == 0 or len(bf_empty) == 0:
        logging.warning(
            "3D Gaussian densifier failed to produce a voxelized volume out of the shape.\n"
            "Usually, it means the shape represented by 3D Gaussians has regions of very low "
            "quality which resulted in holes in the shell. \n"
            "To fix this issue, double check your shape, "
            "or try adjusting args passed to the volume sampling function.\n"
            "For example, try reducing the opacity_threshold towards 0.0, or octree_level towards 7 or 6."
        )
        return xyz.new_empty([0, 3])

    # scan resulting octrees for subsequent querying
    lengths = torch.tensor([len(bf_octree)], dtype=torch.int)
    level, pyramid, exsum = scan_octrees(bf_octree, lengths)
    # should check level == query_level

    # create uniform samples, query volume
    query_points = morton_to_points(torch.arange(8 ** query_level, dtype=torch.long, device=device))
    result = unbatched_query(bf_octree, bf_empty, exsum, query_points, query_level)

    # filter out 'empty' space; keep inside and occupied points
    mask = result[:] != -1
    sample_points = query_points[mask]

    # still need to untransform
    sample_points = 2 ** (1 - query_level) * sample_points - torch.ones((3), device=device)
    return dmax * sample_points + cen

@torch.no_grad()
def sample_points_in_volume(
    xyz: torch.Tensor,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
    num_samples: Optional[int] = None,
    octree_level: int = 8,
    opacity_threshold: float = 0.35,
    post_scale_factor: float = 1.0,
    jitter: bool = True,
    clip_samples_to_input_bbox: bool = True,
    viewpoints: Optional[torch.Tensor] = None,
    scaling_activation: Optional[Callable] = None,
    scaling_inverse_activation: Optional[Callable] = None,
):
    r""" Logic for sampling additional points inside a shape represented by sparse 3D Gaussian Splats.
    Reconstructions based on 3D Gaussian Splats result in shells of objects, leaving the internal volume unfilled.
    In addition, the resulting shell is not a watertight shell, but rather,
    a collection of anisotropic gaussian samples on it.

    Certain applications require additional samples within the volume.
    For example: physics simulations are more accurate when using volumetric mass.
    The logic in this class approximates the volumetric interior with additional points injected to the interior of
    objects.

    The algorithm will attempt to voxelize the Gaussians and predict an approximated surface,
    and then sample additional points within it making sure the volume is evenly spaced (e.g. no voxel is sampled
    more than once).
    Note that reconstructions of poor quality may obtain samples with varying degrees of quality.

    .. note::

        **Choosing Densifier Args.**

        The *densifier* is a non-learned method described in details below.
        Consequentially, different models of varying quality may require adjusting the parameters.
        As a rule of thumb, `octree_level` controls the density of volume samples. Higher density
        ensures more points are sampled within the volume, but may also expose holes within the shape shell.
        `opacity_threshold` controls how quickly low opacity cells get culled away. Lower quality models may
        want to lower this parameter as low as 0.0 to avoid exposing holes.
        `jitter` ensures the returned points are random, the exact usage should vary by application.
        The default `viewpoints` provide adequate coverage for common objects, but more complex objects with many
        cavities may benefit from a more specialized set of viewpoints.
        `post_scale_factor` downscales the returned points using their mean as the center of scale, to ensure
        they reside within the shape shell. It is recommended to leave this value lower than and
        close to 1.0 -- for concave shapes downscaling too much may cause the points to drift away from the shape shell.

    .. note::

        **Implementation Details.** The object sampling takes place in 2 steps.

        1. The set of 3D Gaussians is converted to voxels using a novel hierarchical algorithm which builds on
           kaolin’s :ref:`Structured Point Cloud (SPC)<spc>` (which functions as an octree).
           The axis aligned bounding box of the Gaussians is enclosed in a cubical root node of an octree.
           This node is subdivided in an 8-way split, and a list of overlapping gaussian IDs is
           maintained for each sub node. The nodes that contain voxels are subdivided again and tested for overlap.
           This process repeats until a desired resolution is achieved according to the octree level.
           The nodes at the frontier of the octree are a voxelization of the Gaussians, represented by an SPC.
           The opacity_threshold parameter may cause some cells to get culled if they haven't accumulated enough density
           from the 3D Gaussians overlapping them.
           At the end of this step, the SPC does not include voxels ‘inside’ the object represented by the Gaussians,
           but rather, voxels that represent the shape shell.

        2. Volume filling of voxelized shell by carving the space of voxels using rendered depth maps.
           This is achieved by ray-tracing the SPC from an icosahedral collection of viewpoints to create a depth map
           for each view. These depth maps are fused together into a second sparse SPC using a novel algorithm that
           maintains the occupancy state for each node of the full octree.
           These states are: empty, occupied, or unseen.
           Finally, the occupancy state of points in a regular grid are determined by querying this SPC.
           The union of the sets of occupied and unseen points serves as a sampling of the solid object.

        Post process: the Gaussians are now converted to dense voxels including the volume.
        A point is sampled at the center of each voxel.
        If jitter is true, a small perturbation is also applied to
        each point. The perturbation is small enough such that each point remains within the voxel.
        Each voxel should contain at most one point by the end of this phase.
        If num_samples is specified, the points are randomly subsampled to return a sized sample pool.

    Args:
        xyz (torch.FloatTensor):
            A tensor of shape :math:`(\text{N, 3})` containing the Gaussian means.
            For example, using the original Inria codebase, this corresponds to `GaussianModel.get_xyz`.
        scale (torch.FloatTensor):
            A tensor of shape :math:`(\text{N, 3})` containing the Gaussian covariance scale components, in a format
            of a 3D scale vector per Gaussian. The scale is assumed to be post-activation.
            For example, using the original Inria codebase, this corresponds to `GaussianModel.get_scaling`.
        rotation (torch.FloatTensor):
            A tensor of shape :math:`(\text{N, 4})` containing the Gaussian covariance rotation components, in a format
            of a 4D quaternion per Gaussian. The rotation is assumed to be post-activation.
            For example, using the original Inria codebase, this corresponds to `GaussianModel.get_rotation`.
        opacity (torch.FloatTensor):
            A tensor of shape :math:`(\text{N, 1})` or :math:`(\text{N,})` containing the Gaussian opacities.
            For example, using the original Inria codebase, this corresponds to `GaussianModel.get_opacity`.
        mask (optional, torch.BoolTensor):
            An optional :math:`(\text{N,})` binary mask which selects only a subset of the gaussian to use
            for predicting the shell. Useful if some Gaussians are suspected as noise.
            By default, the mask is assumed to be a tensor of ones to select all Gaussians.
        num_samples (optional, int):
            An optional upper cap on the number of points sampled in the predicted volume.
            If not specified, the volume is evenly sampled in space according to the octree resolution.
        octree_level (int):
            A Structured Point Cloud of cubic resolution :math:`(\text{3**level})` will be constructed to voxelize and
            fill the volume with points. A single point will be sampled within each voxel.
            Higher values require more memory, and may suffer from holes in the shell.
            At the same time, higher values provide more points within the shape volume.
            octree_level range supported is in :math:`[\text{6, 10}]`.
        opacity_threshold (float):
            The densification algorithm starts by voxelizing space using the gaussian responses and their
            associated opacities. Each cell accumulated the opacity induced by the Gaussians overlapping it.
            Voxels with accumulated opacity below this threshold will be masked away.
            If :math:`\text{opacity_threshold} > 0.0`, no culling will take place.
        post_scale_factor (float):
            Postprocess: if :math:`\text{post_scale_factor} < 1.0`, the returned pointcloud will be rescaled to ensure
            it fits inside the hull of the input points.
            It is recommended to avoid values significantly lower than 1 with concave or multi-part objects.
        jitter (bool):
            If true, applies a small jitter to the returned volume points.
            If false, the returned points lie on an equally distanced grid.
        clip_samples_to_input_bbox (bool):
            If true, the *densifier* will compute a bounding box out of the input gaussian means.
            Any points sampled outside of this bounding box will be rejected.
            For most purposes, it is recommended to leave this "safety mechanism" toggled on.
        viewpoints (optional, torch.Tensor):
            Collection of viewpoints used to 'carve' out seen voxel space around the shell after it's voxelized.
            These is a :math:`(\text{C, 3})` tensor of camera viewpoints facing the center,
            chosen based on empirical heuristics.
            If not specified, kaolin will opt to use its own set of default views.
        scaling_activation (Callable): activation applied on scaling.
        scaling_inverse_activation (Callable): inverse of activation applied on scaling.

    Return:
        (torch.FloatTensor): A tensor of :math:`(\text{K, 3})` points sampled inside the approximated shape volume.
    """
    assert octree_level >= 6 and octree_level <= 10, \
        'octree_level range supported is in [6, 10]. \n' \
        'Higher values, while generating more points within the shape volume, ' \
        'require more memory and may suffer from holes in the shape shell.'

    # Reshape to single dim if needed
    if opacity.ndim == 2:
        opacity = opacity.squeeze(1)

    if viewpoints is None:
        viewpoints = _generate_default_viewpoints()

    if scaling_activation is None:
        scaling_activation = torch.exp
    if scaling_inverse_activation is None:
        scaling_inverse_activation = torch.log
    device = xyz.device

    # Select the required points and solidify
    num_surface_pts = xyz.shape[0]
    mask = mask.to(device=device).clone() if mask is not None \
        else xyz.new_ones((num_surface_pts,), device=device, dtype=torch.bool)
    xyz = xyz[mask].clone()
    scale = scale[mask].clone()
    rotation = rotation[mask].clone()
    opacity = opacity[mask].clone()

    volume_pts = _solidify(
        xyz.contiguous(), scale.contiguous(), rotation.contiguous(), opacity.contiguous(),
        opacity_threshold=opacity_threshold,
        gs_level=octree_level, query_level=octree_level, viewpoints=viewpoints,
        scaling_activation=scaling_activation,
        scaling_inverse_activation=scaling_inverse_activation,
    )
    # An empty tensor means densification failed - in that case we return an empty point set
    # and skip post-processing
    if len(volume_pts) == 0:
        return volume_pts
    # Postprocess: apply small jitter to sampled points so they don't reside exactly on voxel centers
    if jitter:
        volume_pts = _jitter(volume_pts, octree_level)
    # Postprocess: rescale returned points to fit in input points hull
    # post_scale_factor should be very close to 1.0, otherwise concave shapes may rescale out of the shape boundary.
    if post_scale_factor < 1.0:
        mean = volume_pts.mean(dim=0)
        volume_pts -= mean
        volume_pts *= post_scale_factor
        volume_pts += mean
    # Failsafe mechanism: eradicate points sampled out of the original gaussians bounding box
    if clip_samples_to_input_bbox:
        bbox_min = xyz.min(dim=0)[0]
        bbox_max = xyz.max(dim=0)[0]
        out_of_box = (bbox_max <= volume_pts).any(dim=1) | (bbox_min >= volume_pts).any(dim=1)
        volume_pts = volume_pts[~out_of_box]
    # Subsample the returned point set if num_samples is specified
    num_total_pts = volume_pts.shape[0]
    if num_samples is not None and num_samples < num_total_pts:
        sample_indices = torch.randperm(num_total_pts)[:num_samples]
        volume_pts = volume_pts[sample_indices]
    return volume_pts
