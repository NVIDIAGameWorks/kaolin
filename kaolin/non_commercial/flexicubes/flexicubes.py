# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from .tables import *
from kaolin.utils.testing import check_tensor

__all__ = [
    'FlexiCubes'
]


class FlexiCubes:
    """

    This class implements the FlexiCubes method for extracting meshes from scalar fields. 
    It maintains a series of lookup tables and indices to support the mesh extraction process. 
    FlexiCubes, a differentiable variant of the Dual Marching Cubes (DMC) scheme, enhances 
    the geometric fidelity and mesh quality of reconstructed meshes by dynamically adjusting 
    the surface representation through gradient-based optimization.

    During instantiation, the class loads DMC tables from a file and transforms them into 
    PyTorch tensors on the specified device.

    .. image:: ../img/flexicubes.png
        :width: 600
        :alt: flexicubes

    .. note::
        For more details and example usage in optimization, refer to the 
        `Flexible Isosurface Extraction for Gradient-Based Mesh Optimization`_ SIGGRAPH 2023 paper,
        and the `project page`_.

    Attributes:
        device (str): Specifies the computational device (default is "cuda").
        dmc_table (torch.Tensor): Dual Marching Cubes (DMC) table that encodes the edges 
            associated with each dual vertex in 256 Marching Cubes (MC) configurations.
        num_vd_table (torch.Tensor): Table holding the number of dual vertices in each of 
            the 256 MC configurations.
        check_table (torch.Tensor): Table resolving ambiguity in cases C16 and C19 
            of the DMC configurations.
        tet_table (torch.Tensor): Lookup table used in tetrahedralizing the isosurface.
        quad_split_1 (torch.Tensor): Indices for splitting a quad into two triangles 
            along one diagonal.
        quad_split_2 (torch.Tensor): Alternative indices for splitting a quad into 
            two triangles along the other diagonal.
        quad_split_train (torch.Tensor): Indices for splitting a quad into four triangles 
            during training by connecting all edges to their midpoints.
        cube_corners (torch.Tensor): Defines the positions of a standard unit cube's 
            eight corners in 3D space, ordered starting from the origin (0,0,0), 
            moving along the x-axis, then y-axis, and finally z-axis. 
            Used as a blueprint for generating a voxel grid.
        cube_corners_idx (torch.Tensor): Cube corners indexed as powers of 2, used 
            to retrieve the case id.
        cube_edges (torch.Tensor): Edge connections in a cube, listed in pairs. 
            Used to retrieve edge vertices in DMC.
        edge_dir_table (torch.Tensor): A mapping tensor that associates edge indices with 
            their corresponding axis. For instance, edge_dir_table[0] = 0 indicates that the 
            first edge is oriented along the x-axis. 
        dir_faces_table (torch.Tensor): A tensor that maps the corresponding axis of shared edges 
            across four adjacent cubes to the shared faces of these cubes. For instance, 
            dir_faces_table[0] = [5, 4] implies that for four cubes sharing an edge along 
            the x-axis, the first and second cubes share faces indexed as 5 and 4, respectively. 
            This tensor is only utilized during isosurface tetrahedralization.
        adj_pairs (torch.Tensor): 
            A tensor containing index pairs that correspond to neighboring cubes that share the same edge.

    .. _Flexible Isosurface Extraction for Gradient-Based Mesh Optimization:
        https://research.nvidia.com/labs/toronto-ai/flexicubes/
    .. _project page:
        https://github.com/nv-tlabs/FlexiCubes/tree/main
    """

    def __init__(self, device="cuda"):

        self.device = device
        self.dmc_table = torch.tensor(dmc_table, dtype=torch.long, device=device, requires_grad=False)
        self.num_vd_table = torch.tensor(num_vd_table,
                                         dtype=torch.long, device=device, requires_grad=False)
        self.check_table = torch.tensor(
            check_table,
            dtype=torch.long, device=device, requires_grad=False)

        self.tet_table = torch.tensor(tet_table, dtype=torch.long, device=device, requires_grad=False)
        self.quad_split_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=device, requires_grad=False)
        self.quad_split_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=device, requires_grad=False)
        self.quad_split_train = torch.tensor(
            [0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.long, device=device, requires_grad=False)

        self.cube_corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [
                                         1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=torch.float, device=device)
        self.cube_corners_idx = torch.pow(2, torch.arange(8, requires_grad=False))
        self.cube_edges = torch.tensor([0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6,
                                       2, 0, 3, 1, 7, 5, 6, 4], dtype=torch.long, device=device, requires_grad=False)

        self.edge_dir_table = torch.tensor([0, 2, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1],
                                           dtype=torch.long, device=device)
        self.dir_faces_table = torch.tensor([
            [[5, 4], [3, 2], [4, 5], [2, 3]],
            [[5, 4], [1, 0], [4, 5], [0, 1]],
            [[3, 2], [1, 0], [2, 3], [0, 1]]
        ], dtype=torch.long, device=device)
        self.adj_pairs = torch.tensor([0, 1, 1, 3, 3, 2, 2, 0], dtype=torch.long, device=device)

    def construct_voxel_grid(self, resolution):
        r"""
        Generates a voxel grid based on the specified resolution.

        The voxelgrid is represented as a mesh, with the voxels corners as vertices.

        Args:
            resolution (int or list[int]): The resolution of the voxel grid. If an integer
                is provided, it is used for all three dimensions. If a list or tuple 
                of 3 integers is provided, they define the resolution for the x, 
                y, and z dimensions respectively.

        Return:
            (torch.Tensor, torch.Tensor):

                - The corners of the voxels of the constructed voxelgrid,
                  centered and normalized (i.e: the vertices are in the range :math:`[-0.5, 0.5]),
                  of shape :math:`((\text{res} + 1) ^ 3, 3)`.
                - The indices into vertices for each voxel,
                  of shape :math:`(\text{res}, 8)`.
        """
        base_cube_f = torch.arange(8).to(self.device)
        if isinstance(resolution, int):
            resolution = (resolution, resolution, resolution)
        voxel_grid_template = torch.ones(resolution, device=self.device)

        resolution = torch.tensor([resolution], dtype=torch.float, device=self.device)
        coords = torch.nonzero(voxel_grid_template).float() / resolution  # N, 3
        verts = (self.cube_corners.unsqueeze(0) / resolution + coords.unsqueeze(1)).reshape(-1, 3)
        cubes = (base_cube_f.unsqueeze(0) +
                 torch.arange(coords.shape[0], device=self.device).unsqueeze(1) * 8).reshape(-1)

        verts_rounded = torch.round(verts * 10**5) / (10**5)
        verts_unique, inverse_indices = torch.unique(verts_rounded, dim=0, return_inverse=True)
        cubes = inverse_indices[cubes.reshape(-1)].reshape(-1, 8)

        return verts_unique - 0.5, cubes

    def __call__(self, voxelgrid_vertices, scalar_field, cube_idx, resolution, qef_reg_scale=1e-3,
                 weight_scale=0.99, beta=None, alpha=None, gamma_f=None, training=False,
                 output_tetmesh=False, grad_func=None):
        r"""
        Main function for mesh extraction from scalar field using FlexiCubes. This function converts 
        discrete signed distance fields, encoded on voxel grids and additional per-cube parameters, 
        to triangle or tetrahedral meshes using a differentiable operation as described in 
        `Flexible Isosurface Extraction for Gradient-Based Mesh Optimization`_. FlexiCubes enhances 
        mesh quality and geometric fidelity by adjusting the surface representation based on gradient 
        optimization. The output surface is differentiable with respect to the input vertex positions, 
        scalar field values, and weight parameters.

        If you intend to extract a surface mesh from a fixed Signed Distance Field without the 
        optimization of parameters, it is suggested to provide the "grad_func" which should 
        return the surface gradient at any given 3D position. When grad_func is provided, the process 
        to determine the dual vertex position adapts to solve a Quadratic Error Function (QEF), as 
        described in the `Manifold Dual Contouring`_ paper, and employs an smart splitting strategy. 
        Please note, this approach is non-differentiable.

        For more details and example usage in optimization, refer to the 
        `Flexible Isosurface Extraction for Gradient-Based Mesh Optimization`_ SIGGRAPH 2023 paper.

        Args:
            voxelgrid_vertices (torch.Tensor):
                Coordinates of the voxel grid vertices, can be deformed,
                of shape :math:`(\text{num_vertices}, 3)`.
            scalar_field (torch.Tensor):
                Scalar field values at each vertex of the voxel grid. Negative values 
                denote that the corresponding vertex resides inside the isosurface,
                of shape :math:`(\text{num_vertices},)`. This affects 
                the directions of the extracted triangle faces and volume to be tetrahedralized.
            cube_idx (torch.Tensor):
                Indices of 8 vertices for each cube in the voxel grid,
                of shape :math:`(\text{num_cube}, 8)`.
            resolution (int or list[int]):
                The resolution of the voxel grid.
                If an integer is provided, it is used for all three dimensions.
                If a list or tuple of 3 integers is provided,
                they specify the resolution for the x, y, and z dimensions respectively.
            qef_reg_scale (float, optional):
                The scaling factor applied to the regularization loss to
                prevent issues with singularity when solving the QEF.
                This parameter is only used when a 'grad_func' is specified.
                Default: 1e-3.
            weight_scale (float, optional):
                The scale of weights in FlexiCubes. Should be between 0 and 1.
                Default: 0.99.
            beta (torch.Tensor, optional):
                Weight parameters for the cube edges to adjust dual vertices positioning,
                of shape :math:`(\text{num_cube}, 12)`. Default: uniform value for all edges.
            alpha (torch.Tensor, optional):
                Weight parameters for the cube corners to adjust dual vertices positioning,
                of shape :math:`(\text{num_cubes}, 8)`. Default: uniform value for all vertices.
            gamma_f (torch.Tensor, optional):
                Weight parameters to control the splitting of quadrilaterals into triangles,
                of shape :math:`(\text{num_cubes},)`.
                Default: uniform value for all cubes.
            training (bool, optional):
                If set to True, applies differentiable quad splitting for training.
                Default: False.
            output_tetmesh (bool, optional):
                If set to True, outputs a tetrahedral mesh, otherwise, 
                outputs a triangular mesh. Defaults: False.
            grad_func (callable, optional):
                A function to compute the surface gradient at specified 3D positions
                (input: Nx3 positions). The function should return gradients as an Nx3 tensor.
                If None, the original FlexiCubes algorithm is utilized. Defaults to None.

        Return:
            (torch.Tensor, torch.LongTensor, torch.Tensor):

                - Vertices for the extracted triangular/tetrahedral mesh,
                  of shape :math:`(\text{num_vertices},  3)`.
                - Faces for the extracted triangular/tetrahedral mesh,
                  of shape :math:`(\text{num_faces}, 3)`/:math:`(\text{num_faces}, 4)`.
                - Regularizer L_dev, computed per dual vertex,
                  of shape :math:`(\text{num_dual_vertices},)`.

        .. _Flexible Isosurface Extraction for Gradient-Based Mesh Optimization:
            https://research.nvidia.com/labs/toronto-ai/flexicubes/
        .. _Manifold Dual Contouring:
            https://people.engr.tamu.edu/schaefer/research/dualsimp_tvcg.pdf
        """
        assert torch.is_tensor(voxelgrid_vertices) and \
            check_tensor(voxelgrid_vertices, (None, 3), throw=False), \
            "'voxelgrid_vertices' should be a tensor of shape (num_vertices, 3)"
        num_vertices = voxelgrid_vertices.shape[0]
        assert torch.is_tensor(scalar_field) and \
            check_tensor(scalar_field, (num_vertices,), throw=False), \
            "'scalar_field' should be a tensor of shape (num_vertices,)"
        assert torch.is_tensor(cube_idx) and \
            check_tensor(cube_idx, (None, 8), throw=False), \
            "'cube_idx' should be a tensor of shape (num_cubes, 8)"
        num_cubes = cube_idx.shape[0]
        assert beta is None or (
            torch.is_tensor(beta) and
            check_tensor(beta, (num_cubes, 12), throw=False)
        ), "'beta' should be a tensor of shape (num_cubes, 12)"
        assert alpha is None or (
            torch.is_tensor(alpha) and
            check_tensor(alpha, (num_cubes, 8), throw=False)
        ), "'alpha' should be a tensor of shape (num_cubes, 8)"
        assert gamma_f is None or (
            torch.is_tensor(gamma_f) and
            check_tensor(gamma_f, (num_cubes,), throw=False)
        ), "'gamma_f' should be a tensor of shape (num_cubes,)"

        surf_cubes, occ_fx8 = self._identify_surf_cubes(scalar_field, cube_idx)
        if surf_cubes.sum() == 0:
            return (
                torch.zeros((0, 3), device=self.device),
                torch.zeros((0, 4), dtype=torch.long, device=self.device) if output_tetmesh else
                torch.zeros((0, 3), dtype=torch.long, device=self.device),
                torch.zeros((0), device=self.device)
            )
        beta, alpha, gamma_f = self._normalize_weights(
            beta, alpha, gamma_f, surf_cubes, weight_scale)

        case_ids = self._get_case_id(occ_fx8, surf_cubes, resolution)

        surf_edges, idx_map, edge_counts, surf_edges_mask = self._identify_surf_edges(
            scalar_field, cube_idx, surf_cubes
        )

        vd, L_dev, vd_gamma, vd_idx_map = self._compute_vd(
            voxelgrid_vertices, cube_idx[surf_cubes], surf_edges, scalar_field,
            case_ids, beta, alpha, gamma_f, idx_map, grad_func, qef_reg_scale
        )
        vertices, faces, s_edges, edge_indices = self._triangulate(
            scalar_field, surf_edges, vd, vd_gamma, edge_counts, idx_map,
            vd_idx_map, surf_edges_mask, training, grad_func
        )
        if not output_tetmesh:
            return vertices, faces, L_dev
        else:
            vertices, tets = self._tetrahedralize(
                voxelgrid_vertices, scalar_field, cube_idx, vertices, faces, surf_edges, s_edges, vd_idx_map, case_ids, edge_indices,
                surf_cubes, training)
            return vertices, tets, L_dev

    def _compute_reg_loss(self, vd, ue, edge_group_to_vd, vd_num_edges):
        """
        Regularizer L_dev as in Equation 8
        """
        dist = torch.norm(ue - torch.index_select(input=vd, index=edge_group_to_vd, dim=0), dim=-1)
        mean_l2 = torch.zeros_like(vd[:, 0])
        mean_l2 = (mean_l2).index_add_(0, edge_group_to_vd, dist) / vd_num_edges.squeeze(1).float()
        mad = (dist - torch.index_select(input=mean_l2, index=edge_group_to_vd, dim=0)).abs()
        return mad

    def _normalize_weights(self, beta, alpha, gamma_f, surf_cubes, weight_scale):
        """
        Normalizes the given weights to be non-negative. If input weights are None, it creates and returns a set of weights of ones.
        """
        n_cubes = surf_cubes.shape[0]

        if beta is not None:
            beta = (torch.tanh(beta) * weight_scale + 1)
        else:
            beta = torch.ones((n_cubes, 12), dtype=torch.float, device=self.device)

        if alpha is not None:
            alpha = (torch.tanh(alpha) * weight_scale + 1)
        else:
            alpha = torch.ones((n_cubes, 8), dtype=torch.float, device=self.device)

        if gamma_f is not None:
            gamma_f = torch.sigmoid(gamma_f) * weight_scale + (1 - weight_scale) / 2
        else:
            gamma_f = torch.ones((n_cubes), dtype=torch.float, device=self.device)

        return beta[surf_cubes], alpha[surf_cubes], gamma_f[surf_cubes]

    @torch.no_grad()
    def _get_case_id(self, occ_fx8, surf_cubes, res):
        """
        Obtains the ID of topology cases based on cell corner occupancy. This function resolves the 
        ambiguity in the Dual Marching Cubes (DMC) configurations as described in Section 1.3 of the 
        supplementary material. It should be noted that this function assumes a regular grid.
        """
        case_ids = (occ_fx8[surf_cubes] * self.cube_corners_idx.to(self.device).unsqueeze(0)).sum(-1)

        problem_config = self.check_table.to(self.device)[case_ids]
        to_check = problem_config[..., 0] == 1
        problem_config = problem_config[to_check]
        if not isinstance(res, (list, tuple)):
            res = [res, res, res]

        # The 'problematic_configs' only contain configurations for surface cubes. Next, we construct a 3D array,
        # 'problem_config_full', to store configurations for all cubes (with default config for non-surface cubes).
        # This allows efficient checking on adjacent cubes.
        problem_config_full = torch.zeros(list(res) + [5], device=self.device, dtype=torch.long)
        vol_idx = torch.nonzero(problem_config_full[..., 0] == 0)  # N, 3
        vol_idx_problem = vol_idx[surf_cubes][to_check]
        problem_config_full[vol_idx_problem[..., 0], vol_idx_problem[..., 1], vol_idx_problem[..., 2]] = problem_config
        vol_idx_problem_adj = vol_idx_problem + problem_config[..., 1:4]

        within_range = (
            vol_idx_problem_adj[..., 0] >= 0) & (
            vol_idx_problem_adj[..., 0] < res[0]) & (
            vol_idx_problem_adj[..., 1] >= 0) & (
            vol_idx_problem_adj[..., 1] < res[1]) & (
            vol_idx_problem_adj[..., 2] >= 0) & (
            vol_idx_problem_adj[..., 2] < res[2])

        vol_idx_problem = vol_idx_problem[within_range]
        vol_idx_problem_adj = vol_idx_problem_adj[within_range]
        problem_config = problem_config[within_range]
        problem_config_adj = problem_config_full[vol_idx_problem_adj[..., 0],
                                                 vol_idx_problem_adj[..., 1], vol_idx_problem_adj[..., 2]]
        # If two cubes with cases C16 and C19 share an ambiguous face, both cases are inverted.
        to_invert = (problem_config_adj[..., 0] == 1)
        idx = torch.arange(case_ids.shape[0], device=self.device)[to_check][within_range][to_invert]
        case_ids.index_put_((idx,), problem_config[to_invert][..., -1])
        return case_ids

    @torch.no_grad()
    def _identify_surf_edges(self, scalar_field, cube_idx, surf_cubes):
        """
        Identifies grid edges that intersect with the underlying surface by checking for opposite signs. As each edge 
        can be shared by multiple cubes, this function also assigns a unique index to each surface-intersecting edge 
        and marks the cube edges with this index.
        """
        occ_n = scalar_field < 0
        all_edges = cube_idx[surf_cubes][:, self.cube_edges].reshape(-1, 2)
        unique_edges, _idx_map, counts = torch.unique(all_edges, dim=0, return_inverse=True, return_counts=True)

        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1

        surf_edges_mask = mask_edges[_idx_map]
        counts = counts[_idx_map]

        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=cube_idx.device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), device=cube_idx.device)
        # Shaped as [number of cubes x 12 edges per cube]. This is later used to map a cube edge to the unique index
        # for a surface-intersecting edge. Non-surface-intersecting edges are marked with -1.
        idx_map = mapping[_idx_map]
        surf_edges = unique_edges[mask_edges]
        return surf_edges, idx_map, counts, surf_edges_mask

    @torch.no_grad()
    def _identify_surf_cubes(self, scalar_field, cube_idx):
        """
        Identifies grid cubes that intersect with the underlying surface by checking if the signs at 
        all corners are not identical.
        """
        occ_n = scalar_field < 0
        occ_fx8 = occ_n[cube_idx.reshape(-1)].reshape(-1, 8)
        _occ_sum = torch.sum(occ_fx8, -1)
        surf_cubes = (_occ_sum > 0) & (_occ_sum < 8)
        return surf_cubes, occ_fx8

    def _linear_interp(self, edges_weight, edges_x):
        """
        Computes the location of zero-crossings on 'edges_x' using linear interpolation with 'edges_weight'.
        """
        edge_dim = edges_weight.dim() - 2
        assert edges_weight.shape[edge_dim] == 2
        edges_weight = torch.cat([torch.index_select(input=edges_weight, index=torch.tensor(1, device=self.device), dim=edge_dim), -
                                 torch.index_select(input=edges_weight, index=torch.tensor(0, device=self.device), dim=edge_dim)]
                                 , edge_dim)
        denominator = edges_weight.sum(edge_dim)
        ue = (edges_x * edges_weight).sum(edge_dim) / denominator
        return ue

    def _solve_vd_QEF(self, p_bxnx3, norm_bxnx3, c_bx3, qef_reg_scale):
        p_bxnx3 = p_bxnx3.reshape(-1, 7, 3)
        norm_bxnx3 = norm_bxnx3.reshape(-1, 7, 3)
        c_bx3 = c_bx3.reshape(-1, 3)
        A = norm_bxnx3
        B = ((p_bxnx3) * norm_bxnx3).sum(-1, keepdims=True)

        A_reg = (torch.eye(3, device=p_bxnx3.device) * qef_reg_scale).unsqueeze(0).repeat(p_bxnx3.shape[0], 1, 1)
        B_reg = (qef_reg_scale * c_bx3).unsqueeze(-1)
        A = torch.cat([A, A_reg], 1)
        B = torch.cat([B, B_reg], 1)
        dual_verts = torch.linalg.lstsq(A, B).solution.squeeze(-1)
        return dual_verts

    def _compute_vd(self, voxelgrid_vertices, surf_cubes_fx8, surf_edges, scalar_field,
                    case_ids, beta, alpha, gamma_f, idx_map, grad_func, qef_reg_scale):
        """
        Computes the location of dual vertices as described in Section 4.2
        """
        alpha_nx12x2 = torch.index_select(input=alpha, index=self.cube_edges, dim=1).reshape(-1, 12, 2)
        surf_edges_x = torch.index_select(input=voxelgrid_vertices, index=surf_edges.reshape(-1), dim=0).reshape(-1, 2, 3)
        surf_edges_s = torch.index_select(input=scalar_field, index=surf_edges.reshape(-1), dim=0).reshape(-1, 2, 1)
        zero_crossing = self._linear_interp(surf_edges_s, surf_edges_x)

        idx_map = idx_map.reshape(-1, 12)
        num_vd = torch.index_select(input=self.num_vd_table, index=case_ids, dim=0)
        edge_group, edge_group_to_vd, edge_group_to_cube, vd_num_edges, vd_gamma = [], [], [], [], []

        total_num_vd = 0
        vd_idx_map = torch.zeros((case_ids.shape[0], 12), dtype=torch.long, device=self.device, requires_grad=False)
        if grad_func is not None:
            normals = torch.nn.functional.normalize(grad_func(zero_crossing), dim=-1)
            vd = []
        for num in torch.unique(num_vd):
            cur_cubes = (num_vd == num)  # consider cubes with the same numbers of vd emitted (for batching)
            curr_num_vd = cur_cubes.sum() * num
            curr_edge_group = self.dmc_table[case_ids[cur_cubes], :num].reshape(-1, num * 7)
            curr_edge_group_to_vd = torch.arange(
                curr_num_vd, device=self.device).unsqueeze(-1).repeat(1, 7) + total_num_vd
            total_num_vd += curr_num_vd
            curr_edge_group_to_cube = torch.arange(idx_map.shape[0], device=self.device)[
                cur_cubes].unsqueeze(-1).repeat(1, num * 7).reshape_as(curr_edge_group)

            curr_mask = (curr_edge_group != -1)
            edge_group.append(torch.masked_select(curr_edge_group, curr_mask))
            edge_group_to_vd.append(torch.masked_select(curr_edge_group_to_vd.reshape_as(curr_edge_group), curr_mask))
            edge_group_to_cube.append(torch.masked_select(curr_edge_group_to_cube, curr_mask))
            vd_num_edges.append(curr_mask.reshape(-1, 7).sum(-1, keepdims=True))
            vd_gamma.append(torch.masked_select(gamma_f, cur_cubes).unsqueeze(-1).repeat(1, num).reshape(-1))

            if grad_func is not None:
                with torch.no_grad():
                    cube_e_verts_idx = idx_map[cur_cubes]
                    curr_edge_group[~curr_mask] = 0

                    verts_group_idx = torch.gather(input=cube_e_verts_idx, dim=1, index=curr_edge_group)
                    verts_group_idx[verts_group_idx == -1] = 0
                    verts_group_pos = torch.index_select(
                        input=zero_crossing, index=verts_group_idx.reshape(-1), dim=0).reshape(-1, num.item(), 7, 3)
                    v0 = voxelgrid_vertices[surf_cubes_fx8[cur_cubes][:, 0]].reshape(-1, 1, 1, 3).repeat(1, num.item(), 1, 1)
                    curr_mask = curr_mask.reshape(-1, num.item(), 7, 1)
                    verts_centroid = (verts_group_pos * curr_mask).sum(2) / (curr_mask.sum(2))

                    normals_bx7x3 = torch.index_select(input=normals, index=verts_group_idx.reshape(-1), dim=0).reshape(
                        -1, num.item(), 7, 3
                    )
                    curr_mask = curr_mask.squeeze(2)
                    vd.append(self._solve_vd_QEF((verts_group_pos - v0) * curr_mask, normals_bx7x3 * curr_mask,
                                                 verts_centroid - v0.squeeze(2), qef_reg_scale) + v0.reshape(-1, 3))
        edge_group = torch.cat(edge_group)
        edge_group_to_vd = torch.cat(edge_group_to_vd)
        edge_group_to_cube = torch.cat(edge_group_to_cube)
        vd_num_edges = torch.cat(vd_num_edges)
        vd_gamma = torch.cat(vd_gamma)

        if grad_func is not None:
            vd = torch.cat(vd)
            L_dev = torch.zeros([1], device=self.device)
        else:
            vd = torch.zeros((total_num_vd, 3), device=self.device)
            beta_sum = torch.zeros((total_num_vd, 1), device=self.device)

            idx_group = torch.gather(input=idx_map.reshape(-1), dim=0, index=edge_group_to_cube * 12 + edge_group)

            x_group = torch.index_select(input=surf_edges_x, index=idx_group.reshape(-1), dim=0).reshape(-1, 2, 3)
            s_group = torch.index_select(input=surf_edges_s, index=idx_group.reshape(-1), dim=0).reshape(-1, 2, 1)

            zero_crossing_group = torch.index_select(
                input=zero_crossing, index=idx_group.reshape(-1), dim=0).reshape(-1, 3)

            alpha_group = torch.index_select(input=alpha_nx12x2.reshape(-1, 2), dim=0,
                                             index=edge_group_to_cube * 12 + edge_group).reshape(-1, 2, 1)
            ue_group = self._linear_interp(s_group * alpha_group, x_group)

            beta_group = torch.gather(input=beta.reshape(-1), dim=0,
                                      index=edge_group_to_cube * 12 + edge_group).reshape(-1, 1)
            beta_sum = beta_sum.index_add_(0, index=edge_group_to_vd, source=beta_group)
            vd = vd.index_add_(0, index=edge_group_to_vd, source=ue_group * beta_group) / beta_sum
            L_dev = self._compute_reg_loss(vd, zero_crossing_group, edge_group_to_vd, vd_num_edges)

        v_idx = torch.arange(vd.shape[0], device=self.device)  # + total_num_vd

        vd_idx_map = (vd_idx_map.reshape(-1)).scatter(dim=0, index=edge_group_to_cube *
                                                      12 + edge_group, src=v_idx[edge_group_to_vd])

        return vd, L_dev, vd_gamma, vd_idx_map

    def _triangulate(self, scalar_field, surf_edges, vd, vd_gamma, edge_counts, idx_map, vd_idx_map, surf_edges_mask, training, grad_func):
        """
        Connects four neighboring dual vertices to form a quadrilateral. The quadrilaterals are then split into 
        triangles based on the gamma parameter, as described in Section 4.3.
        """
        with torch.no_grad():
            group_mask = (edge_counts == 4) & surf_edges_mask  # surface edges shared by 4 cubes.
            group = idx_map.reshape(-1)[group_mask]
            vd_idx = vd_idx_map[group_mask]
            edge_indices, indices = torch.sort(group, stable=True)
            quad_vd_idx = vd_idx[indices].reshape(-1, 4)

            # Ensure all face directions point towards the positive SDF to maintain consistent winding.
            s_edges = scalar_field[surf_edges[edge_indices.reshape(-1, 4)[:, 0]].reshape(-1)].reshape(-1, 2)
            flip_mask = s_edges[:, 0] > 0
            quad_vd_idx = torch.cat((quad_vd_idx[flip_mask][:, [0, 1, 3, 2]],
                                     quad_vd_idx[~flip_mask][:, [2, 3, 1, 0]]))
        if grad_func is not None:
            # when grad_func is given, split quadrilaterals along the diagonals with more consistent gradients.
            with torch.no_grad():
                vd_gamma = torch.nn.functional.normalize(grad_func(vd), dim=-1)
                quad_gamma = torch.index_select(input=vd_gamma, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4, 3)
                gamma_02 = (quad_gamma[:, 0] * quad_gamma[:, 2]).sum(-1, keepdims=True)
                gamma_13 = (quad_gamma[:, 1] * quad_gamma[:, 3]).sum(-1, keepdims=True)
        else:
            quad_gamma = torch.index_select(input=vd_gamma, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4)
            gamma_02 = torch.index_select(input=quad_gamma, index=torch.tensor(
                0, device=self.device), dim=1) * torch.index_select(input=quad_gamma, 
                                                                    index=torch.tensor(2, device=self.device), dim=1)
            gamma_13 = torch.index_select(input=quad_gamma, index=torch.tensor(
                1, device=self.device), dim=1) * torch.index_select(input=quad_gamma, 
                                                                    index=torch.tensor(3, device=self.device), dim=1)
        if not training:
            mask = (gamma_02 > gamma_13).squeeze(1)
            faces = torch.zeros((quad_gamma.shape[0], 6), dtype=torch.long, device=quad_vd_idx.device)
            faces[mask] = quad_vd_idx[mask][:, self.quad_split_1]
            faces[~mask] = quad_vd_idx[~mask][:, self.quad_split_2]
            faces = faces.reshape(-1, 3)
        else:
            vd_quad = torch.index_select(input=vd, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4, 3)
            vd_02 = (torch.index_select(input=vd_quad, index=torch.tensor(0, device=self.device), dim=1) +
                     torch.index_select(input=vd_quad, index=torch.tensor(2, device=self.device), dim=1)) / 2
            vd_13 = (torch.index_select(input=vd_quad, index=torch.tensor(1, device=self.device), dim=1) +
                     torch.index_select(input=vd_quad, index=torch.tensor(3, device=self.device), dim=1)) / 2
            weight_sum = (gamma_02 + gamma_13) + 1e-8
            vd_center = ((vd_02 * gamma_02.unsqueeze(-1) + vd_13 * gamma_13.unsqueeze(-1)) /
                         weight_sum.unsqueeze(-1)).squeeze(1)
            vd_center_idx = torch.arange(vd_center.shape[0], device=self.device) + vd.shape[0]
            vd = torch.cat([vd, vd_center])
            faces = quad_vd_idx[:, self.quad_split_train].reshape(-1, 4, 2)
            faces = torch.cat([faces, vd_center_idx.reshape(-1, 1, 1).repeat(1, 4, 1)], -1).reshape(-1, 3)
        return vd, faces, s_edges, edge_indices

    def _tetrahedralize(
            self, voxelgrid_vertices, scalar_field, cube_idx, vertices, faces, surf_edges, s_edges, vd_idx_map, case_ids, edge_indices,
            surf_cubes, training):
        """
        Tetrahedralizes the interior volume to produce a tetrahedral mesh, as described in Section 4.5.
        """
        occ_n = scalar_field < 0
        occ_fx8 = occ_n[cube_idx.reshape(-1)].reshape(-1, 8)
        occ_sum = torch.sum(occ_fx8, -1)

        inside_verts = voxelgrid_vertices[occ_n]
        mapping_inside_verts = torch.ones((occ_n.shape[0]), dtype=torch.long, device=self.device) * -1
        mapping_inside_verts[occ_n] = torch.arange(occ_n.sum(), device=self.device) + vertices.shape[0]
        """ 
        For each grid edge connecting two grid vertices with different
        signs, we first form a four-sided pyramid by connecting one
        of the grid vertices with four mesh vertices that correspond
        to the grid edge and then subdivide the pyramid into two tetrahedra
        """
        inside_verts_idx = mapping_inside_verts[surf_edges[edge_indices.reshape(-1, 4)[:, 0]].reshape(-1, 2)[
            s_edges < 0]]
        if not training:
            inside_verts_idx = inside_verts_idx.unsqueeze(1).expand(-1, 2).reshape(-1)
        else:
            inside_verts_idx = inside_verts_idx.unsqueeze(1).expand(-1, 4).reshape(-1)

        tets_surface = torch.cat([faces, inside_verts_idx.unsqueeze(-1)], -1)
        """ 
        For each grid edge connecting two grid vertices with the
        same sign, the tetrahedron is formed by the two grid vertices
        and two vertices in consecutive adjacent cells
        """
        inside_cubes = (occ_sum == 8)
        inside_cubes_center = voxelgrid_vertices[cube_idx[inside_cubes].reshape(-1)].reshape(-1, 8, 3).mean(1)
        inside_cubes_center_idx = torch.arange(
            inside_cubes_center.shape[0], device=inside_cubes.device) + vertices.shape[0] + inside_verts.shape[0]

        surface_n_inside_cubes = surf_cubes | inside_cubes
        edge_center_vertex_idx = torch.ones(((surface_n_inside_cubes).sum(), 13),
                                            dtype=torch.long, device=voxelgrid_vertices.device) * -1
        surf_cubes = surf_cubes[surface_n_inside_cubes]
        inside_cubes = inside_cubes[surface_n_inside_cubes]
        edge_center_vertex_idx[surf_cubes, :12] = vd_idx_map.reshape(-1, 12)
        edge_center_vertex_idx[inside_cubes, 12] = inside_cubes_center_idx

        all_edges = cube_idx[surface_n_inside_cubes][:, self.cube_edges].reshape(-1, 2)
        unique_edges, _idx_map, counts = torch.unique(
            all_edges, dim=0, return_inverse=True, return_counts=True)
        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 2
        mask = mask_edges[_idx_map]
        counts = counts[_idx_map]
        mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=self.device) * -1
        mapping[mask_edges] = torch.arange(mask_edges.sum(), device=self.device)
        idx_map = mapping[_idx_map]

        group_mask = (counts == 4) & mask
        group = idx_map.reshape(-1)[group_mask]
        edge_indices, indices = torch.sort(group)
        edge2cube = torch.arange(
            (_idx_map.shape[0] // 12), dtype=torch.long, device=self.device
        ).unsqueeze(1).expand(-1, 12).reshape(-1)[group_mask]
        edge_idx = torch.arange((12), dtype=torch.long, device=self.device).unsqueeze(
            0).expand(_idx_map.shape[0] // 12, -1).reshape(-1)[group_mask]
        # Identify the face shared by the adjacent cells.
        cube_idx_4 = edge2cube[indices].reshape(-1, 4)
        edge_dir = self.edge_dir_table[edge_idx[indices]].reshape(-1, 4)[..., 0]
        shared_faces_4x2 = self.dir_faces_table[edge_dir].reshape(-1)
        cube_idx_4x2 = cube_idx_4[:, self.adj_pairs].reshape(-1)
        # Identify an edge of the face with different signs and
        # select the mesh vertex corresponding to the identified edge.
        case_ids_expand = torch.ones((surface_n_inside_cubes).sum(), dtype=torch.long, device=voxelgrid_vertices.device) * 255
        case_ids_expand[surf_cubes] = case_ids
        cases = case_ids_expand[cube_idx_4x2]
        quad_edge = edge_center_vertex_idx[cube_idx_4x2, self.tet_table[cases, shared_faces_4x2]].reshape(-1, 2)
        mask = (quad_edge == -1).sum(-1) == 0
        inside_edge = mapping_inside_verts[unique_edges[mask_edges][edge_indices].reshape(-1)].reshape(-1, 2)
        tets_inside = torch.cat([quad_edge, inside_edge], -1)[mask]

        tets = torch.cat([tets_surface, tets_inside])
        vertices = torch.cat([vertices, inside_verts, inside_cubes_center])
        return vertices, tets
