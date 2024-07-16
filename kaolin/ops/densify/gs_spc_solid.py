import math
import numpy as np
import torch

from kaolin.ops.spc import scan_octrees, morton_to_points
from kaolin import _C

from kaolin.ops.densify.bf_recon import bf_recon, unbatched_query
from kaolin.ops.densify.raytraced_spc_dataset import RayTracedSPCDataset

# collection of viewpoints used to 'carve' out seen space (might need more!)
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
R = (R_z @ R_y @ R_z).unsqueeze(0).float()

viewpoints = torch.cat([
    anchors,
    icosahedron,
    (R @ (2.0 * icosahedron)[:,:,None]).squeeze(-1),
    (R @ R @ (3.0 * icosahedron)[:,:,None]).squeeze(-1),
    (R @ R @ R @ (4.0 * icosahedron)[:,:,None]).squeeze(-1),
    (R @ R @ R @ R @ (5.0 * icosahedron)[:,:,None]).squeeze(-1),
    (R @ R @ R @ R @ R @ (6.0 * icosahedron)[:,:,None]).squeeze(-1),
], dim=0)

def solidify(xyz, scales, rots, opacities, gs_level, query_level):
    r"""Create tensor of uniform samples 'inside' collection of Gaussian Splats.

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
    # AABB of Gaussian means
    pmin = torch.min(xyz, dim=0, keepdim=False)[0]
    pmax = torch.max(xyz, dim=0, keepdim=False)[0]

    # find the AABB diagonal vector and centroid
    diff = pmax-pmin
    cen = (0.5*(pmin+pmax)).cuda()

    # find the maximum diagonal component, add tiny amount to compensate for covariance vectors (a hack!)
    dmax = (0.5*torch.max(diff) + 0.05).cuda()

    # transform Gaussians to [-1,-1,-1]x[1,1,1]
    xyz = (xyz - cen)/dmax
    scales = scales/dmax

    # some constants
    scale_voxel_tolerance = 0.125
    iso = 11.345 # 99th percentile

    # compute spc from gsplats
    morton, merged_opacities, gs_per_voxel =  _C.gs_to_spc_cuda(xyz, scales, rots, opacities, iso, scale_voxel_tolerance, gs_level)

    # filter out low opacities
    opacity_tol = 0.1
    mask =  merged_opacities[:] > opacity_tol
    morton = morton[mask]
    gs_octree = _C.ops.spc.morton_to_octree(morton, gs_level)

    # create depthmaps
    dataset = RayTracedSPCDataset(viewpoints, gs_octree)

    # fuse depthmaps into seen/unseen aware spc
    bf_octree, bf_empty, _ = bf_recon(dataset, final_level=query_level, sigma=0.005)

    # scan resulting octrees for subsequent querying
    lengths = torch.tensor([len(bf_octree)], dtype=torch.int)
    level, pyramid, exsum = scan_octrees(bf_octree, lengths)
    # should check level == query_level

    # create uniform samples, query volume
    query_points = morton_to_points(torch.arange(8**query_level, dtype=torch.long, device='cuda'))
    result = unbatched_query(bf_octree, bf_empty, exsum, query_points, query_level)

    # filter out 'empty' space; keep inside and boundary points
    mask = result[:] != -1
    sample_points =  query_points[mask]

    # still need to untransform
    sample_points =  2**(1-query_level) * sample_points - torch.ones((3), device='cuda')
    return dmax * sample_points + cen
