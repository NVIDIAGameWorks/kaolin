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
from typing import Optional
from kaolin.ops.spc import scan_octrees, morton_to_points, bf_recon, unbatched_query
from kaolin.ops.spc.raytraced_spc_dataset import RayTracedSPCDataset
from kaolin import _C

__all__ = [
    'VolumeDensifier'
]

class VolumeDensifier:
    """ Logic for sampling additional points inside a shape represented by sparse 3D Gaussian Splats.
    Reconstructions based on 3D Gaussian Splats result in shells of objects, leaving the internal volume unfilled.
    In addition, the resulting shell is not a watertight shell, but rather,
    a collection of anisotropic gaussian samples on it.

    Certain applications require additional samples within the volume.
    For example: physics simulations are more accurate when using volumetric mass.
    The logic in this class approximates the volumetric interior with additional points injected to the interior of
    objects.
    """

    def __init__(
        self,
        resolution: int = 8,
        opacity_threshold: float = 0.35,
        post_scale_factor: float = 0.93,
        jitter: bool = True,
        viewpoints: Optional[torch.Tensor]=None
    ):
        r"""Creates a new VolumeDensifier, which allows for sampling points within the approximated volume of a
        shape represented by Gaussian Splatting.

        Args:
            resolution (int):
                A Structured Point Cloud of cubic resolution :math:`(\text{2**res})` will be constructed to fill
                the volume with points. Higher values require more memory. Max resolution supported is 10.
            opacity_threshold (float):
                Preprocess: if opacity_threshold < 1.0, points with opacity below the threshold will be masked away.
            post_scale_factor (float):
                Postprocess: if post_scale_factor < 1.0, the returned pointcloud will be rescaled to ensure
                it fits inside the hull of the input points.
            jitter (bool):
                If true, applies a small jitter to the returned volume points.
                If false, the returned points lie on an equally distanced grid
            viewpoints (optional, torch.Tensor):
                Collection of viewpoints used to 'carve' out seen voxel space around the shell after it's voxelized.
                These is a :math:`(\text{C, 3})` tensor of camera viewpoints facing the center,
                chosen based on empirical heuristics.
                If not specified, kaolin will opt to use its own set of default views.
        """
        self.resolution = resolution
        self.opacity_threshold = opacity_threshold
        self.post_scale_factor = post_scale_factor
        self.jitter = jitter
        self.viewpoints = viewpoints or self._generate_default_viewpoints()

    @classmethod
    def _generate_default_viewpoints(cls):
        r""" Generates a collection of default viewpoints used to 'carve' out seen space.
        These anchors are chosen based on empirical heuristics
        """
        anchors = torch.tensor([
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [-4.0, 0.0, 0.0],
            [0.0, -4.0, 0.0],
            [2.3, 2.3, 2.3],
            [-2.3, 2.3, 2.3],
            [2.3, -2.3, 2.3],
            [-2.3, -2.3, 2.3]
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
        return viewpoints

    def _jitter(self, pts):
        r""" Applies random pertubrations to a set of voxelized points.
        The pertubrations are small enough such that each point remains in the voxel cell it belongs in.
        """
        N = pts.shape[0]
        device = pts.device
        cell_radius = 1.0 / 2.0 **self.resolution
        radius = cell_radius * torch.sqrt(torch.rand(N, device=device))

        # azimuth [0~2pi]
        theta = torch.rand(N, device=device) * 2.0 * torch.pi
        # polar [0~pi]
        phi = torch.rand(N, device=device) * torch.pi

        # spherical coordinates to cartesian
        x = radius * torch.sin(phi) * torch.cos(theta)
        y = radius * torch.sin(phi) * torch.sin(theta)
        z = radius * torch.cos(phi)
        delta = torch.stack([x,y,z], dim=1)
        return pts + delta

    def _solidify(self, xyz, scales, rots, opacities, gs_level, query_level):
        r"""Creates a tensor of uniform samples 'inside' collection of Gaussian Splats.

        Args:
            xyz (torch.FloatTensor) : Gaussian Splat means, of shape :math:`(\text{num_guaasians, 3})`.
            scales (torch.FloatTensor) : Gaussian Splat scales, of shape :math:`(\text{num_guaasians, 3})`.
            rots (torch.FloatTensor) : Gaussian Splat rots, of shape :math:`(\text{num_guaasians, 4})`.
            opacities (torch.FloatTensor) : Gaussian Splat opacities, of shape :math:`(\text{num_guaasians})`.

            gs_level (int): The level of the interal octree created.
            query_level (int): The level of the uniform sample grid.

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
        cen = (0.5 * (pmin + pmax)).to(device=device)

        # find the maximum diagonal component, add tiny amount to compensate for covariance vectors (a hack!)
        dmax = (0.5 * torch.max(diff) + 0.05).to(device=device)

        # transform Gaussians to [-1,-1,-1]x[1,1,1]
        xyz = (xyz - cen) / dmax
        scales = scales / dmax

        # some constants
        scale_voxel_tolerance = 0.125
        iso = 11.345  # 99th percentile

        # compute spc from gsplats
        morton, merged_opacities, gs_per_voxel = _C.ops.conversions.gs_to_spc_cuda(
            xyz, scales, rots, opacities, iso, scale_voxel_tolerance, gs_level
        )

        # filter out low opacities
        opacity_tol = 0.1
        mask = merged_opacities[:] > opacity_tol
        morton = morton[mask]
        gs_octree = _C.ops.spc.morton_to_octree(morton, gs_level)

        # create depthmaps
        dataset = RayTracedSPCDataset(self.viewpoints, gs_octree)

        # fuse depthmaps into seen/unseen aware spc
        bf_octree, bf_empty, _ = bf_recon(dataset, final_level=query_level, sigma=0.005)

        # scan resulting octrees for subsequent querying
        lengths = torch.tensor([len(bf_octree)], dtype=torch.int)
        level, pyramid, exsum = scan_octrees(bf_octree, lengths)
        # should check level == query_level

        # create uniform samples, query volume
        query_points = morton_to_points(torch.arange(8 ** query_level, dtype=torch.long, device=device))
        result = unbatched_query(bf_octree, bf_empty, exsum, query_points, query_level)

        # filter out 'empty' space; keep inside and boundary points
        mask = result[:] != -1
        sample_points = query_points[mask]

        # still need to untransform
        sample_points = 2 ** (1 - query_level) * sample_points - torch.ones((3), device=device)
        return dmax * sample_points + cen

    @torch.no_grad()
    def sample_points_in_volume(self, xyz, scale, rotation, opacity, mask=None, count=None):
        r""" Samples 3D points inside the approximated volume of a radiance field, represented by Gaussian Splats.
        The gaussians are assumed to reside on the shell of the object. The algorithm will attempt to voxelize
        the gaussians and predict an approximated surface, and then sample additional points within it.

        .. note::  Implementation Details:

        The object sampling takes place in 2 steps.

        1. The set of 3D Gaussians is converted to voxels using a novel hierarchical algorithm which builds on
           kaolin’s :ref:`Structured Point Cloud (SPC)<spc>` (which functions as an octree).
           The axis aligned bounding box of the gaussians is enclosed in a cubical root node of an octree.
           This node is subdivided in an 8-way split, and a list of overlapping gaussian IDs is
           maintained for each sub node. The nodes that contain voxels are subdivided again and tested for overlap.
           This process repeats until a desired resolution is achieved.
           The nodes at the frontier of the octree are a voxelization of the gaussians, represented by an SPC.
           At the end of this step, the SPC does not include voxels ‘inside’ the object represented by the gaussians.

        2. Volume filling of voxelized shell by carving the space of voxels using rendered depth maps.

           This is achieved by ray-tracing the SPC from an icosahedral collection of viewpoints to create a depth map
           for each view. These depth maps are fused together into a second sparse SPC using a novel algorithm that
           maintains the occupancy state for each node of the full octree.
           These states are: empty, occupied, or unseen.
           Finally, the occupancy state of points in a regular grid are determined by querying this SPC.
           The union of the sets of occupied and unseen points serves as a sampling of the solid object.

        Args:
            xyz (torch.FloatTensor):
                A tensor of shape :math:`(\text{N, 3})` containing the Gaussian means.
                For example, using the original Inria codebase, this corresponds to GaussianModel.get_xyz
            scale (torch.FloatTensor):
                A tensor of shape :math:`(\text{N, 3})` containing the Gaussian covariance scale components, in a format
                of a 3D scale vector per Gaussian. The scale is assumed to be post-activation.
                For example, using the original Inria codebase, this corresponds to GaussianModel.get_scaling
            rotation (torch.FloatTensor):
                A tensor of shape :math:`(\text{N, 4})` containing the Gaussian covariance rotation components, in a format
                of a 4D quaternion per Gaussian. The rotation is assumed to be post-activation.
                For example, using the original Inria codebase, this corresponds to GaussianModel.get_rotation
            opacity (torch.FloatTensor):
                A tensor of shape :math:`(\text{N, 1})` or :math:`(\text{N,})` containing the Gaussian opacities.
                For example, using the original Inria codebase, this corresponds to GaussianModel.get_opacity
            mask (optional, torch.BoolTensor):
                An optional :math:`(\text{N,})` binary mask which selects only a subset of the gaussian to use
                for predicting the shell. Useful if some Gaussians are suspected as noise.
                By default, the mask is assumed to be a tensor of ones to select all Gaussians.
            count (optional, int):
                An optional upper cap on the number of points sampled in the predicted volume.
                If not specified, the volume is evenly sampled in space according to the VolumeDensifier resolution.

        Return:
            (torch.FloatTensor): a tensor of (K, 3) points sampled inside the approximated shape volume.
        """
        # Reshape to single dim if needed
        if opacity.ndim == 2:
            opacity = opacity.squeeze(1)

        device = xyz.device

        # Select the required points and solidify
        num_surface_pts = xyz.shape[0]
        mask = mask.to(device=device).clone() if mask is not None \
            else xyz.new_ones((num_surface_pts,), device=device, dtype=torch.bool)
        # Preprocess: prune low opacity points
        if self.opacity_threshold < 1.0:
            opacity_mask = opacity.reshape(-1) < self.opacity_threshold
            mask &= ~opacity_mask
        xyz = xyz[mask]
        scale = scale[mask]
        rotation = rotation[mask]
        opacity = opacity[mask]
        volume_pts = self._solidify(xyz.contiguous(), scale.contiguous(), rotation.contiguous(), opacity.contiguous(),
                                    gs_level=self.resolution, query_level=self.resolution)
        if self.jitter:
            volume_pts = self._jitter(volume_pts)
        # Postprocess: rescale returned points to ensure they fit in input points hull
        if self.post_scale_factor < 1.0:
            mean = volume_pts.mean(dim=0)
            volume_pts -= mean
            volume_pts *= self.post_scale_factor
            volume_pts += mean

        num_total_pts = volume_pts.shape[0]
        if count is not None and count < num_total_pts:
            sample_indices = torch.randperm(num_total_pts)[:count]
            volume_pts = volume_pts[sample_indices]
        return volume_pts
