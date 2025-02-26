# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import torch
from functools import partial

import warp as wp
import numpy as np
import torch
import warp.sparse as wps

from kaolin.physics.utils import finite_diff_jac
from kaolin.physics.utils.warp_utilities import _warp_csr_from_torch_dense
from kaolin.physics.simplicits import  weight_function_lbs

__all__ = [
    "sparse_lbs_matrix", 
    "sparse_collision_jacobian_matrix", 
    "sparse_mass_matrix",
    "sparse_dFdz_matrix_from_dense",
    'lumped_mass_matrix',
    'lbs_matrix',
    'jacobian_dF_dz',
    'jacobian_dx_dz'
]


@wp.kernel
def _get_lbs_triplets_wp_kernel(
    sim_weights: wp.array2d(dtype=wp.float32),
    sim_pts: wp.array(dtype=wp.vec3),
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    vals: wp.array3d(dtype=wp.float32),
):  # pragma: no cover
    # Get thread index
    p, k, i = wp.tid()
    idx = (p * sim_weights.shape[1] + k) * 3 + i

    weight = sim_weights[p, k]
    point = sim_pts[p]

    # For each point, we need to fill 3 rows (x,y,z, 1 coordinates)
    # in shape (3, 12)
    # [[x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, x, y, z, 1, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0, 0, 0, x, y, z, 1]]
    # repeated for each handle weight

    rows[idx] = p * 3 + i
    cols[idx] = k * 3 + i

    vals[idx, 0, 0] = weight * point[0]
    vals[idx, 0, 1] = weight * point[1]
    vals[idx, 0, 2] = weight * point[2]
    vals[idx, 0, 3] = weight

@wp.kernel
def _get_collision_jacobian_triplets_wp_kernel(
    indices: wp.array(dtype=wp.int32),
    pt_is_static: wp.array(dtype=wp.int32),
    sim_weights: wp.array2d(dtype=wp.float32),
    sim_pts: wp.array(dtype=wp.vec3),
    rows: wp.array(dtype=wp.int32),
    cols: wp.array(dtype=wp.int32),
    vals: wp.array3d(dtype=wp.float32),
    count: wp.array(dtype=wp.int32)
):  # pragma: no cover
    # Get thread index
    t, k, i = wp.tid() # valid point index, handle index, row index

    p = indices[t] # point index from global sim_weights
    
    # Skip the static objects
    if pt_is_static[p] == 1:
        return
    
    wp.atomic_add(count, 0, 1)
    
    weight = sim_weights[p, k]
    point = sim_pts[p]
    
    # Index of the triplet
    idx = (t * sim_weights.shape[1] + k) * 3 + i

    # For each point, we need to fill 3 rows (x,y,z, 1 coordinates)
    # in shape (3, 12)
    # [[x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, x, y, z, 1, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0, 0, 0, x, y, z, 1]]
    # repeated for each handle weight

    rows[idx] = t * 3 + i # row of the sparse matrix
    cols[idx] = k * 3 + i # column of the sparse matrix

    vals[idx, 0, 0] = weight * point[0]
    vals[idx, 0, 1] = weight * point[1]
    vals[idx, 0, 2] = weight * point[2]
    vals[idx, 0, 3] = weight


@wp.kernel
def _get_dFdz_triplets_wp_kernel(sim_weights: wp.array2d(dtype=wp.float32),
                        rows: wp.array(dtype=wp.int32),
                        cols: wp.array(dtype=wp.int32),
                        vals: wp.array(dtype=wp.float32),
                                 triplet_index: wp.array(dtype=wp.int32)):  # pragma: no cover

    # Get thread index
    i = wp.tid()

    dim = int(3)

    if i < sim_weights.shape[0]:
        # For each sample point, and each handle we need to fill the following matrix:
        # [[1, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0 ,0],
        #  [0, 1, 0, 0, 0, 0, 0, 0, 0 ,0, 0 ,0],
        #  [0, 0, 1, 0, 0, 0, 0, 0, 0 ,0, 0 ,0],
        #  [0, 0, 0, 0, 1, 0, 0, 0, 0 ,0, 0 ,0],
        #  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        #  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        #  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]

        for h in range(sim_weights.shape[1]):
            col = 12*h
            weight = sim_weights[i, h]

            for d in range(dim*dim):
                row = dim*dim*i + d
                # Skip the (dim+1)'th column (translations)
                if (col+1) % (dim+1) == 0:
                    col += 1
                idx = wp.atomic_add(triplet_index, 0, 1)
                rows[idx] = row
                cols[idx] = col
                vals[idx] = weight * wp.float32(1.0)

                col += 1


def sparse_lbs_matrix(sim_weights, sim_pts):
    r"""Creates a sparse LBS matrix from a set of points and corresponding weights.

    Args:
        sim_weights (wp.array2d(dtype=wp.float32)): Skinning weights.
        sim_pts (wp.array(dtype=wp.vec3)): Sample points.

    Returns:
        wp.sparse.bsr_matrix: Sparse LBS matrix of size :math:`(3 \text{num_samples}, 12 \text{num_handles})`
    """
    # Calculate number of non-zero entries
    # 3 entries per block. Each block is a 1x4 matrix. There are n x num_handles blocks
    nnz = sim_weights.shape[0] * sim_weights.shape[1] * 3

    # Create arrays for triplets
    rows = wp.empty(nnz, dtype=wp.int32)
    cols = wp.empty(nnz, dtype=wp.int32)
    vals = wp.zeros((nnz, 1, 4), dtype=wp.float32)

    # Launch kernel to build triplets
    wp.launch(
        kernel=_get_lbs_triplets_wp_kernel,
        dim=(*sim_weights.shape, 3),
        inputs=[
            sim_weights,
            sim_pts,
            rows,
            cols,
            vals,
        ]
    )

    nrow = sim_weights.shape[0]*3
    ncol = sim_weights.shape[1]*3
    return wps.bsr_from_triplets(nrow, ncol, rows, cols, vals)


def sparse_collision_jacobian_matrix(sim_weights, sim_pts, indices, cp_is_static):
    r"""Creates a sparse collision Jacobian matrix from the global weights and points of the scene. 
    Indices holds to the indices of the sim_pts and sim_weights to include in the matrix. 
    If index corresponds to a static object, the point is not included, but zeros are added instead.

    Args:
        sim_weights (wp.array2d(dtype=wp.float32)): Skinning weights.
        sim_pts (wp.array(dtype=wp.vec3)): Sample points.
        indices (wp.array(dtype=wp.int32)): Indices of the sim_pts and sim_weights to include in the matrix. 
        cp_is_static (wp.array(dtype=wp.int32)): 1 if the point is in a static object, 0 otherwise.
    
    Returns:
        wp.sparse.bsr_matrix: Sparse LBS matrix size :math:`(3 \text{num_indices}, 12 \text{num_handles})`
    """    
    # Calculate number of non-zero entries
    # 3 entries per block. Each block is a 1x4 matrix. There are num_pts x num_handles blocks 
    nnz = indices.shape[0] * sim_weights.shape[1] * 3
    nrow = indices.shape[0]*3
    ncol = sim_weights.shape[1]*3

    
    # Create arrays for triplets
    rows = wp.empty(nnz, dtype=wp.int32)
    cols = wp.empty(nnz, dtype=wp.int32)
    vals = wp.zeros((nnz, 1, 4), dtype=wp.float32)
    count = wp.zeros(1, dtype=wp.int32)

    # Launch kernel to build triplets
    wp.launch(
        kernel=_get_collision_jacobian_triplets_wp_kernel,
        dim=(indices.shape[0], sim_weights.shape[1], 3),
        inputs=[
            indices,
            cp_is_static,
            sim_weights,
            sim_pts,
            rows,
            cols,
            vals,
            count
        ]
    )
    final_nnz = int(count.numpy()[0])

    if final_nnz > 0:
        final_rows = wp.clone(rows[:final_nnz])
        final_cols = wp.clone(cols[:final_nnz])
        final_vals = wp.clone(vals[:final_nnz])
        return wps.bsr_from_triplets(nrow, ncol, final_rows, final_cols, final_vals)
    else:
        return wps.bsr_zeros(nrow, ncol, wp.types.matrix(shape=(1, 4), dtype=wp.float32))


def sparse_dFdz_matrix_from_dense(enriched_weights_fcn, pts):
    r"""Creates a sparse Jacobian matrix of the deformation gradient with respect to the sample points.

    Args:
        enriched_weights_fcn (function): Function that returns the skinning weights for a given point.
        pts (torch.Tensor): Sample points.

    Returns:
        wp.sparse.bsr_matrix: Sparse Jacobian matrix of size :math:`(9 \text{num_samples}, 12 \text{num_handles})`
    """
    weights = enriched_weights_fcn(pts)
    num_handles = weights.shape[1]
    z = torch.zeros(num_handles, 3, 4,
                    device=weights.device).reshape(-1, 1)

    # Get the dense Jacobian
    dense_dFdz = jacobian_dF_dz(
        enriched_weights_fcn, pts, z)

    return _warp_csr_from_torch_dense(dense_dFdz)


def _sparse_dFdz_matrix(sim_weights: np.ndarray):  # pragma: no cover
    r"""Creates a sparse Jacobian matrix of the deformation gradient with respect to the sample points.

    Args:
        sim_weights (wp.array2d(dtype=wp.float32)): Skinning weights.

    Returns:
        wp.sparse.bsr_matrix: Sparse Jacobian matrix of size :math:`(9 \text{num_samples}, 12 \text{num_handles})`
    """
    # TODO: UNUSED - only works for constant weights.
    # Debug if the dFdz_from_dense is becoming slow or remove this function.

    num_samples = sim_weights.shape[0]
    num_handles = sim_weights.shape[1]

    nnz = 9 * num_samples * num_handles

    rows = wp.zeros(nnz, dtype=wp.int32)
    cols = wp.zeros(nnz, dtype=wp.int32)
    vals = wp.zeros(nnz, dtype=wp.float32)
    offset = wp.zeros(1, dtype=wp.int32)

    wp.launch(
        kernel=_get_dFdz_triplets_wp_kernel,
        dim=num_samples,
        inputs=[wp.array(sim_weights, dtype=wp.float32),
                rows,
                cols,
                vals,
                offset]
    )

    num_rows = 9 * num_samples  # 9 per sample
    num_cols = 12*num_handles  # 12 per handle

    dFdz_csr_matrix = wps.bsr_zeros(num_rows, num_cols, block_type=wp.float32)
    wps.bsr_set_from_triplets(dFdz_csr_matrix, rows, cols, vals)

    return dFdz_csr_matrix


def sparse_mass_matrix(sim_rhos: np.ndarray):
    r"""Creates a sparse mass matrix from a set of densities.

    Args:
        sim_rhos (np.ndarray): Densities.

    Returns:
        wp.sparse.bsr_matrix: Sparse mass matrix of size :math:`(3 \text{num_samples}, 3 \text{num_samples})`
    """

    dim = 3  # dimension of the space

    # Create warp arrays and initialize mass matrix
    wp_sim_rhos_diag = wp.array(
        np.repeat(sim_rhos, dim), dtype=wp.float32)
    mass_matrix = wps.bsr_diag(wp_sim_rhos_diag)
    return mass_matrix


def lumped_mass_matrix(rhos, total_volume, dim=3):
    r"""Calculate the lumped mass matrix of an object sampled via points with spatially uniform sampling, and potentially spatially varying density

    Args:
        rhos (torch.Tensor): Point-wise vector of densities, of shape :math:`(\text{num_samples})`
        total_volume (float): Total volume of object in :math:`m^3`
        dim (int, optional): Spatial dimensions. Defaults to 3.

    Returns:
        torch.Tensor: Diagonal mass matrix of size :math:`(3 \text{num_samples}, 3 \text{num_samples})`
        torch.Tensor: Diagonal INVERSE mass matrix of size :math:`(3 \text{num_samples}, 3 \text{num_samples})`
    """
    # Assumes uniform sample pt distributions over the object
    # Does NOT assume uniform density
    vol_per_sample = total_volume / rhos.shape[0]
    pt_wise_mass = rhos * vol_per_sample
    return torch.diag(pt_wise_mass.repeat_interleave(dim)), torch.diag(1.0 / pt_wise_mass.repeat_interleave(dim))


def lbs_matrix(x0, w):
    r"""Encodes the lbs equation :math:`x_i = sum(w(x^0_i)_j * T_j * \begin{pmatrix}x^0_i\\1\end{pmatrix},\; \forall \; j=0...\|handles\|)` for all pts into matrix B such that flatten(x) = B(x0, w(x0))*flatten(T)

    Args:
        x0 (torch.Tensor): Input points, of shape :math:`(\text{num_samples}, 3)`
        w (torch.Tensor): Weights, of shape :math:`(\text{num_samples}, \text{num_handles})`

    Returns:
        torch.Tensor: Matrix that encodes the lbs transformation, given a set of vertices and corresponding weights, shape :math:`(3 \text{num_samples}, 12 \text{num_handles})`
    """
    num_samples = x0.shape[0]  # N
    num_handles = w.shape[1]  # H

    # Shape (N, 1)
    ones_column = torch.ones(x0.shape[0], 1).to(x0.device)

    # Shape (N, 4) ; x0 homogeneous coords
    x03 = torch.cat((x0, ones_column), dim=1)
    # Shape (3N, 12H) ; 3*num_samples x 4*3*num_handles
    # [[ x1, y1, z1, 1, ... x1, y1, z1, 1 ],
    #  [ x1, y1, z1, 1, ... x1, y1, z1, 1 ],
    #  [ x1, y1, z1, 1, ... x1, y1, z1, 1 ],
    #  [       |        ...        |      ],
    #  [ xN, yN, zN, 1, ... xN, yN, zN, 1 ],
    #  [ xN, yN, zN, 1, ... xN, yN, zN, 1 ],
    #  [ xN, yN, zN, 1, ... xN, yN, zN, 1 ]]
    x03reps = x03.repeat_interleave(3, dim=0).repeat((1, 3 * num_handles))
    # Shape (3N, 12H):
    # [[ w11, w11, w11, w11, w11 (12 times) ... w1H, w1H, w1H, w1H, w1H (12 times) ],
    #  [ w11, w11, w11, w11, w11 (12 times) ... w1H, w1H, w1H, w1H, w1H (12 times) ],
    #  [ w11, w11, w11, w11, w11 (12 times) ... w1H, w1H, w1H, w1H, w1H (12 times) ],
    #  [              |                   ...                     |                ],
    #  [ wN1, wN1, wN1, wN1, wN1 (12 times) ... wNH, wNH, wNH, wNH, N1H (12 times) ],
    #  [ wN1, wN1, wN1, wN1, wN1 (12 times) ... wNH, wNH, wNH, wNH, N1H (12 times) ],
    #  [ wN1, wN1, wN1, wN1, wN1 (12 times) ... wNH, wNH, wNH, wNH, N1H (12 times) ]]
    w_reps = w.repeat_interleave(12, dim=1).repeat_interleave(3, dim=0)
    # Shape (3N, 12H): Hadamard product (element-wise multiplication)
    w_x03reps = torch.mul(w_reps, x03reps)
    # Shape (3N, 3H) ; a (N,H) block matrix of 3x3 identity submats
    # [[ 1, 0, 0, 1, 0, 0, ... 1, 0, 0],
    #  [ 0, 1, 0, 0, 1, 0, ... 0, 1, 0],
    #  [ 0, 0, 1, 0, 0, 1, ... 0, 0, 1],
    #  [         |           ...   |  ],
    #  [ 1, 0, 0, 1, 0, 0, ... 1, 0, 0],
    #  [ 0, 1, 0, 0, 1, 0, ... 0, 1, 0],
    #  [ 0, 0, 1, 0, 0, 1, ... 0, 0, 1]]
    B_setup = torch.kron(
        torch.ones([num_samples, 1], device=x0.device),
        torch.eye(3, device=x0.device)
    ).repeat((1, num_handles))
    # Shape (3N, 12H) ; a (N,H) block matrix of 3x12 binary submats
    # [[ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 ... 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 ... 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    #  [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 ... 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    #  [                |                   ...                  |                ],
    #  [ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 ... 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 ... 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    #  [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 ... 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    B_mask = torch.repeat_interleave(B_setup, 4, dim=1)

    # Shape (3N, 12H) ; a (N,H) block matrix of (3,12) submats
    # [[ w11*x1, w11*y1, w11*z1, w11, 0,      0,      0,      0,  0,      0,      0,      0   .. w1H*x1, w1H*y1, ..]
    #  [ 0,      0,      0,      0,   w11*x1, w11*y1, w11*z1, w11 0,      0,      0,      0   .. 0,      0,      ..]
    #  [ 0,      0,      0,      0,   0,      0,      0,      0,  w11*x1, w11*y1, w11*z1, w11 .. 0,      0,      ..]
    #  [                                        |                                             ..        |          ]
    #  [wN1*xN,  wN1*yN, wN1*zN, wN1, 0,      0,      0,      0,  0,      0,      0,      0,  .. wNH*xN, wNH*yN  ..]
    #  [ 0,      0,      0,      0,   wN1*xN, wN1*yN, wN1*zN, w1N 0,      0,      0,      0   .. 0,      0,      ..]
    #  [ 0,      0,      0,      0,   0,      0,      0,      0,  wN1*xN, wN1*yN, wN1*zN, wN1 .. 0,      0,      ..]]
    B = torch.mul(B_mask, w_x03reps)
    return B


def _jacobian_dF_dz_const_handle(model, x0, z):  # pragma: no cover
    r"""Calculates dF/dz for the skinning weights model (network) and an extra dimension for the constant handle

    Args:
        model (nn.module): Simplicits object network without the constant handle
        x0 (torch.Tensor): Matrix of sample points, of shape :math:`(\text{num_samples}, 3)` 
        z (torch.Tensor): Vector of flattened transforms, of shape :math:`(12 \text{num_samples})`

    Returns:
        torch.Tensor: Jacobian matrix, of shape :math:`(9 \text{num_samples}, 12 \text{num_handles})`
    """
    num_samples = x0.shape[0]

    zeros_vec3_column = torch.zeros((num_samples, 1, 3), device=x0.device)
    ones_col = torch.ones((num_samples, 1), device=x0.device)
    weights = model(x0)
    weights = torch.cat((model(x0), ones_col), dim=1)
    grad_weights = finite_diff_jac(model, x=x0)
    grad_weights = torch.cat(
        [grad_weights.squeeze(), zeros_vec3_column], dim=1)
    x0_h = torch.cat((x0, torch.ones(
        x0.shape[0], 1, device=x0.device, dtype=x0.dtype)), dim=1).detach()

    def _compute_F(Ts):
        # W0_i, dW0_i: lazy, just avoiding neural net evaluations
        def defgrad(x0h, W0_i, dW0_i):
            T_w = torch.sum(W0_i.unsqueeze(dim=1).unsqueeze(dim=2) * Ts, dim=0)
            x_t = torch.tensordot(Ts, x0h.unsqueeze(dim=0), dims=([2], [1]))
            dT = torch.sum(torch.bmm(x_t, dW0_i.unsqueeze(dim=1)), dim=0)
            return dT + T_w.squeeze()[0:3, 0:3] + torch.eye(3, device=x0_h.device)

        # compute F for each sample point
        return torch.vmap(defgrad)(x0_h, weights, grad_weights)

    def _reshape_and_compute_F(z_):
        transforms = z_.reshape(-1, 3, 4)
        deformation_gradient = _compute_F(transforms)
        return deformation_gradient.reshape(9 * num_samples)

    dF_dz = torch.autograd.functional.jacobian(
        _reshape_and_compute_F, z).squeeze(-1)

    return dF_dz


def jacobian_dF_dz(model, x0, z):
    r"""Calculates jacobian dF/dz

    Args:
        model (nn.Module): Simplicits object network + constant handle
        x0 (torch.Tensor): Matrix of sample points, of shape :math:`(\text{num_samples}, 3)`
        z (torch.Tensor): Vector of flattened transforms, of shape :math:`(12 \text{num_samples})`

    Returns:
        torch.Tensor: Jacobian matrix, of shape :math:`(9 \text{num_samples}, 12 \text{num_handles})`
    """
    num_samples = x0.shape[0]

    def compute_defo_grad1(z):
        partial_weight_fcn_lbs = partial(
            weight_function_lbs, tfms=z.reshape(-1, 3, 4).unsqueeze(0), fcn=model)
        # N x 3 x 3 Deformation gradients
        defo_grads = finite_diff_jac(partial_weight_fcn_lbs, x0, eps=1e-7)
        return defo_grads

    dF_dz = torch.autograd.functional.jacobian(
        lambda x: compute_defo_grad1(x).reshape(9 * num_samples), z.flatten())
    return dF_dz


def jacobian_dx_dz(model, x0, z):
    r"""Calculates jacobian :math:`\frac{\partial x}{\partial z}`

    Args:
        model (nn.Module): Simplicits object network + constant handle
        x0 (torch.Tensor): Matrix of sample points, of shape :math:`(\text{num_samples}, 3)`
        z (torch.Tensor): Vector of flattened transforms, of shape :math:`(12 \text{num_samples})`

    Returns:
        torch.Tensor: Jacobian matrix, of shape :math:`(3 \text{num_samples}, 12 \text{num_handles})`
    """
    dx_dz = torch.autograd.functional.jacobian(lambda x: weight_function_lbs(
        x0, tfms=x.reshape(-1, 3, 4).unsqueeze(0), fcn=model).flatten(), z.flatten())
    return dx_dz
