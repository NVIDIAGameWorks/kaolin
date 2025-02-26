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
from typing import Any
import functools
import numpy as np
import warp as wp
import warp.sparse as wps
from warp.optim.linear import LinearOperator
from warp.fem.linalg import inverse_qr


import kaolin.physics.utils.torch_utilities as torch_utilities

__all__ = ["_wp_bsr_to_torch_bsr",
           "_bsr_to_torch",
           "_displacement_delta_kernel",
           "vec12",
           "mat1212",
           "mat312",
           "vec9",
           "mat99"]

vec12 = wp.types.vector(12, dtype=wp.float32)
mat1212 = wp.types.matrix(shape=(12, 12), dtype=wp.float32)
mat312 = wp.types.matrix(shape=(3, 12), dtype=wp.float32)
vec9 = wp.types.vector(9, dtype=wp.float32)
mat99 = wp.types.matrix(shape=(9, 9), dtype=wp.float32)


@wp.kernel
def _displacement_delta_kernel(
    dt: float,
    z: wp.array(dtype=float),
    z_prev: wp.array(dtype=float),
    z_dot: wp.array(dtype=float),
    delta_dz: wp.array(dtype=float),
):  # pragma: no cover
    r"""
    Computes the displacement delta :math:`\delta z = z - z_{prev} - dt \dot{z}` for variational integrator.
    
    Args:
        dt (float): The time step.
        z (wp.array(dtype=float)): The current position as a warp array of shape :math:`(n,)`.
        z_prev (wp.array(dtype=float)): The previous position as a warp array of shape :math:`(n,)`.
        z_dot (wp.array(dtype=float)): The velocity as a warp array of shape :math:`(n,)`.
        delta_dz (wp.array(dtype=float)): The output displacement delta as a warp array of shape :math:`(n,)`.
    """
    i = wp.tid()
    delta_dz[i] = (z[i] - z_prev[i]) - dt*z_dot[i]

@wp.kernel
def _bsr_mul_diag_wp_kernel(
    Bt_values: wp.array3d(dtype=float),
    Bt_columns: wp.array(dtype=int),
    C_values: wp.array(dtype=Any),
    Output_values: wp.array3d(dtype=float),
):  # pragma: no cover
    r"""
    Performs the reduction :math:`B^T C B` where :math:`B` is a sparse matrix and :math:`C` is a diagonal matrix.
    
    Args:
        Bt_values (wp.array3d(dtype=float)): The values of the sparse matrix :math:`B^T`.
        Bt_columns (wp.array(dtype=int)): The columns of the sparse matrix :math:`B^T`.
        C_values (wp.array(dtype=Any)): The values of the diagonal matrix :math:`C`.
        Output_values (wp.array3d(dtype=float)): The output values of the sparse matrix :math:`B^T C B`.
    """
    i, r = wp.tid()
    col = Bt_columns[i]
    C = C_values[col]
    Btr = Bt_values[i, r]
    Otr = Output_values[i, r]
    BtC = wp.vec3(Btr[0], Btr[1], Btr[2]) @ C
    for k in range(3):
        Otr[k] = BtC[k]


def _hessian_reduction(Jt, H, J):  # pragma: no cover
    r"""
    Performs a reduction :math:`J^T H J` where :math:`H` is a block-wise diagonal matrix and :math:`J` is a Jacobian matrix.
    
    Args:
        Jt (wps.BsrMatrix): The left Jacobian matrix of shape :math:`(m, n)`.
        H (wps.BsrMatrix): The Hessian matrix of shape :math:`(n, n)`.
        J (wps.BsrMatrix): The right Jacobian matrix of shape :math:`(n, m)`.
    
    Returns:
        wps.BsrMatrix: The reduced Hessian matrix of shape :math:`(m, m)`.
    """
    Output = wps.bsr_copy(Jt)
    wp.launch(_bsr_mul_diag_wp_kernel,
              dim=(Jt.nnz, Jt.block_shape[0]),
              inputs=[Jt.scalar_values, Jt.columns, H, Output.scalar_values]
              )
    return wps.bsr_mm(Output, J)


def _warp_csr_from_torch_dense(dense_mat):  # pragma: no cover
    r"""Converts a dense torch matrix to a sparse warp csr matrix.
    
    Args:
        dense_mat (torch.Tensor): Dense torch matrix of shape :math:`(m, n)`.
    
    Returns:
        wps.BsrMatrix: The converted sparse warp csr matrix of shape :math:`(m, n)`.
    """
    # Find indices of non-zero elements
    non_zero_indices = torch.nonzero(dense_mat, as_tuple=False)
    # Extract values at those indices
    non_zero_values = dense_mat[non_zero_indices[:,
                                                 0], non_zero_indices[:, 1]]
    # Combine indices and values into triplets
    rows = wp.from_torch(non_zero_indices[:, 0].int().to(device=dense_mat.device, dtype=torch.int))
    cols = wp.from_torch(non_zero_indices[:, 1].int().to(device=dense_mat.device, dtype=torch.int))
    vals = wp.from_torch(non_zero_values.to(device=dense_mat.device, dtype=torch.float32))

    # Create sparse matrix from triplets
    csr_matrix = wps.bsr_zeros(
        dense_mat.shape[0], dense_mat.shape[1], block_type=wp.float32, device=rows.device)
    wps.bsr_set_from_triplets(csr_matrix, rows, cols, vals)
    return csr_matrix


def _assemble_global_hessian(hess_list, wp_obj_to_z_map, z, block_size=3):  # pragma: no cover
    r"""
    Assembles the global Hessian matrix from a list of local Hessian matrices.
    
    Args:
        hess_list (list): The list of local Hessian matrices.
        wp_obj_to_z_map (dict): The mapping from object ids to z indices.
        z (torch.Tensor): The vector of flattened transforms.
        block_size (int): The block size of the Hessian matrices.
    
    Returns:
        wps.BsrMatrix: The assembled global Hessian matrix.
    """
    # From the list assemble the global hessian into wps.bsr_matrix
    # 1. For each tuple in the hess_list, get the obj.id of object i, and j.
    # 2. Map to the z indices of the object
    # 3. Place into the global hessian

    rows = []
    cols = []
    values = []

    for local_hess in hess_list:
        id_i, id_j, H_ij = local_hess

        z_i = wp.to_torch(wp_obj_to_z_map[id_i])[::block_size] // block_size
        z_j = wp.to_torch(wp_obj_to_z_map[id_j])[::block_size] // block_size

        # Create row and column indices for each element in H_ij
        row_indices = z_i.unsqueeze(1).expand(-1, len(z_j)).flatten()
        col_indices = z_j.unsqueeze(0).expand(len(z_i), -1).flatten()

        rows.append(row_indices)
        cols.append(col_indices)

        H_ij_block = (
            H_ij.reshape(
                len(z_i),
                block_size,
                len(z_j),
                block_size,
            )
            .permute((0, 2, 1, 3))
            .reshape(-1, block_size, block_size)
        )
        values.append(H_ij_block)

    # Concatenate all triplets
    rows = torch.cat(rows)
    cols = torch.cat(cols)
    values = torch.vstack(values)

    # Create block sparse tensor
    wp_H = wps.bsr_from_triplets(
        z.shape[0] // block_size,
        z.shape[0] // block_size,
        wp.from_torch(rows),
        wp.from_torch(cols),
        wp.from_torch(values),
    )

    return wp_H


def _wp_bsr_to_torch_bsr(mat):  # pragma: no cover
    r"""Converts a sparse warp BSR matrix (or CSR matrix) to a sparse torch matrix.

    Args:
        mat (wps.BsrMatrix): A sparse warp BSR matrix of shape :math:`(m, n)` with block shape :math:`(b, b)`.

    Returns:
        torch.sparse_bsr_tensor: A sparse torch matrix of shape :math:`(m, n)` with block shape :math:`(b, b)`.
    """
    mat.nnz_sync()

    if mat.block_shape == (1, 1):
        ctor = torch.sparse_csr_tensor
    else:
        ctor = torch.sparse_bsr_tensor
    torch_weights = ctor(
        crow_indices=wp.to_torch(mat.offsets[: mat.nrow + 1]),
        col_indices=wp.to_torch(mat.columns[: mat.nnz]),
        values=wp.to_torch(mat.values[: mat.nnz]),
        size=mat.shape,
    )
    return torch_weights


def _wp_bsr_to_wp_triplets(mat):  # pragma: no cover
    r"""Converts a sparse warp BSR matrix (or CSR matrix) to a sparse warp triplets.

    Args:
        mat (wps.BsrMatrix): A sparse warp BSR matrix

    Returns:
        tuple: (row_indices, col_indices, values) representing the triplets
    """

    # First convert to torch sparse tensor
    torch_mat = _wp_bsr_to_torch_bsr(mat)
    row_indices, col_indices, values = torch_utilities.torch_bsr_to_torch_triplets(
        torch_mat)

    # TODO: Be really careful with the types.
    # Causes all sorts of bugs that are hard to track down.
    row_indices = row_indices.int()
    col_indices = col_indices.int()
    values = values.float()
    return (wp.from_torch(row_indices, dtype=wp.dtype_from_torch(row_indices.dtype)), wp.from_torch(col_indices, dtype=wp.dtype_from_torch(col_indices.dtype)), wp.from_torch(values, dtype=wp.dtype_from_torch(values.dtype)))


def _bsr_to_torch(mat: wps.BsrMatrix):  # pragma: no cover
    r"""Converts a sparse warp BSR matrix (or CSR matrix) to a sparse torch matrix.

    Args:
        mat (wps.BsrMatrix): A sparse warp BSR matrix of shape :math:`(m, n)` with block shape :math:`(b, b)`.

    Returns:
        torch.sparse_bsr_tensor: A sparse torch matrix of shape :math:`(m, n)` with block shape :math:`(b, b)`.
    """
    mat.nnz_sync()
    if mat.block_shape == (1, 1):
        ctor = torch.sparse_csr_tensor
    else:
        ctor = torch.sparse_bsr_tensor

    torch_weights = ctor(
        crow_indices=wp.to_torch(mat.offsets)[: mat.nrow + 1],
        col_indices=wp.to_torch(mat.columns)[: mat.nnz],
        values=wp.to_torch(mat.values)[: mat.nnz],
        size=mat.shape,
    )
    return torch_weights


def _block_diagonalize(list_of_matrices):  # pragma: no cover
    r"""
    Block-diagonalizes a list of matrices.

    Args:
        list_of_matrices (list(wps.BsrMatrix)): A list of 2D warp BSR matrices

    Returns:
        wps.BsrMatrix: A large block-diagonal csr matrix
    """
    sum_rows = sum(m.nrow for m in list_of_matrices)
    sum_cols = sum(m.ncol for m in list_of_matrices)

    current_row = 0
    current_col = 0

    full_row_indices = []
    full_col_indices = []
    full_values = []
    for mat in list_of_matrices:
        nnz = mat.nnz_sync()
        if nnz > 0:
            row_indices = wp.to_torch(mat.uncompress_rows())[:nnz]
            col_indices = wp.to_torch(mat.columns)[:nnz]
            values = wp.to_torch(mat.values)[:nnz]
            full_row_indices.append(row_indices + current_row)
            full_col_indices.append(col_indices + current_col)
            full_values.append(values)

        current_row += mat.nrow
        current_col += mat.ncol

    block_diag_mat = wps.bsr_from_triplets(sum_rows, sum_cols,
                                           wp.from_torch(
                                               torch.cat(full_row_indices)),
                                           wp.from_torch(
                                               torch.cat(full_col_indices)),
                                           wp.from_torch(torch.cat(full_values, dim=0)))
    return block_diag_mat


TILE_SIZE = 72


def _build_preconditioner(lhs: wps.BsrMatrix, p_reg: float = 0.0001):  # pragma: no cover
    r"""
    Builds a preconditioner for the linear system.
    
    Args:
        lhs (wps.BsrMatrix): The left-hand side of the linear system.
        p_reg (float): The regularization parameter.
    
    Returns:
        wps.LinearOperator: The preconditioner.
    """
    tile_size = TILE_SIZE

    # pad lhs with zero blocks so we get an integer number of tiles
    blocks_per_tile = tile_size // lhs.block_shape[0]
    tile_count = (lhs.nrow + blocks_per_tile - 1) // blocks_per_tile

    padding_blocks = tile_count * blocks_per_tile - lhs.nrow
    if padding_blocks > 0:
        nnz = lhs.nnz_sync()
        lhs_padded = wps.bsr_zeros(
            tile_count * blocks_per_tile,
            tile_count * blocks_per_tile,
            block_type=lhs.dtype,
            device=lhs.device,
        )
        lhs_padded.columns = lhs.columns
        lhs_padded.values = lhs.values

        wp.copy(src=lhs.offsets, dest=lhs_padded.offsets, count=lhs.nrow+1)
        lhs_padded.offsets[lhs.nrow+1:lhs_padded.nrow+1].fill_(nnz)
        lhs_padded.copy_nnz_async(nnz)
    else:
        lhs_padded = lhs

    lhs_padded.nnz_sync()
    tile_type = wp.mat((tile_size, tile_size), dtype=float)
    P_coarse = wps.bsr_diag(
        rows_of_blocks=tile_count,
        block_type=tile_type,
    )
    wps.bsr_assign(src=lhs_padded, dest=P_coarse, masked=True)
    P_coarse.nnz_sync()

    tile_chol = _eval_tiled_dense_cholesky_batched_wp_kernel
    tile_solve = _solve_tiled_dense_cholesky_batched_wp_kernel

    BLOCK_DIM = 256

    wp.launch_tiled(
        tile_chol,
        dim=P_coarse.nrow,
        inputs=[P_coarse.scalar_values, p_reg],
        block_dim=BLOCK_DIM,
    )

    def P_inv_mv(x, y, z, alpha, beta):
        # for cg, y = z, alpha = 1, beta = 0
        x = wp.array(x, dtype=float)
        z = wp.array(z, dtype=float)
        wp.launch_tiled(
            tile_solve,
            dim=P_coarse.nrow,
            inputs=[P_coarse.scalar_values, x, z],
            block_dim=BLOCK_DIM,
        )

    return LinearOperator(
        lhs.shape, P_coarse.dtype, P_coarse.device, P_inv_mv
    )


@wp.kernel(enable_backward=False)
def _eval_tiled_dense_cholesky_batched_wp_kernel(mat: wp.array3d(dtype=float), reg: float):  # pragma: no cover
    r"""
    Evaluates the tiled dense Cholesky decomposition of a matrix.
    
    Args:
        mat (wp.array3d(dtype=float)): The matrix to decompose..
        reg (float): The regularization parameter.
    """
    block, _ = wp.tid()

    a = wp.tile_load(
        mat[block], shape=(TILE_SIZE, TILE_SIZE)
    )
    r = wp.tile_ones(dtype=float, shape=(TILE_SIZE)) * reg
    b = wp.tile_diag_add(a, r)
    l = wp.tile_cholesky(b)
    wp.tile_store(mat[block], l)


@wp.kernel(enable_backward=False)
def _solve_tiled_dense_cholesky_batched_wp_kernel(
        L: wp.array3d(dtype=float),
        X: wp.array1d(dtype=float),
        Y: wp.array1d(dtype=float)):  # pragma: no cover
    r"""
    Solves the tiled dense Cholesky decomposition of a matrix.
    
    Args:
        L (wp.array3d(dtype=float)): The lower triangular matrix.
        X (wp.array1d(dtype=float)): The right-hand side of the linear system.
        Y (wp.array1d(dtype=float)): The solution of the linear system.
    """
    block, _ = wp.tid()

    a = wp.tile_load(L[block], shape=(TILE_SIZE, TILE_SIZE))
    x = wp.tile_load(X, offset=block * TILE_SIZE, shape=TILE_SIZE)
    y = wp.tile_cholesky_solve(a, x)
    wp.tile_store(Y, y, offset=block * TILE_SIZE)
