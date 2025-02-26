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

import pytest

import torch
import numpy as np
import warp as wp
import warp.sparse as wps
import kaolin.physics as physics
from kaolin.physics.utils.warp_utilities import _bsr_to_torch, _wp_bsr_to_torch_bsr, _wp_bsr_to_wp_triplets, _block_diagonalize
from kaolin.utils.testing import check_allclose


@pytest.fixture(autouse=True)
def device(request):
    return 'cuda'

@pytest.fixture(autouse=True)
def dtype(request):
    return torch.float32

@pytest.fixture
def get_test_triplets(device, dtype):
    row_inds = torch.tensor([0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27,
                             27, 28, 28, 28, 28, 29, 29, 29, 29], device=device, dtype=torch.int32)

    col_inds = torch.tensor([0,  1,  2,  3,  4,  5,  6,  7,  9, 11, 13, 15, 17, 19, 21, 23,  0,  2,  8, 10,  4,  6,  8, 10,  0,  2,  4,  6,  0,  2,  8, 10,  4,  6,  8, 10,  0,  2,  4,  6,  0,  2,  8, 10,  4,  6,  8, 10,  0,  2,  4,  6,  0,  2,  8, 10,  4,  6,  8, 10,  0,  2,  4,  6,  0,  2,  8, 10,  4,  6,  8, 10, 12, 14, 16, 18, 12, 14, 20, 22, 16, 18, 20, 22, 12, 14, 16, 18, 12, 14, 20, 22, 16, 18, 20, 22, 12, 14, 16, 18, 12, 14, 20, 22, 16, 18, 20, 22, 12, 14, 16, 18, 12, 14, 20, 22, 16, 18, 20, 22, 12, 14, 16,
                            18, 12, 14, 20, 22, 16, 18, 20, 22], device=device, dtype=torch.int32)
    values = torch.tensor([-3.695e-01, -6.913e-01, -2.882e-01,  1.000e+01, -3.695e-01, -6.913e-01, -2.882e-01,  1.000e+01, -6.913e-01,  1.000e+01,  1.922e+00,  1.000e+01,  1.922e+00,  1.000e+01,  1.922e+00,  1.000e+01,  6.099e-02,  2.860e-01, -3.695e-01, -2.882e-01,  6.099e-02,  2.860e-01,  6.099e-02,  2.860e-01,  4.641e-02,  3.154e-01,  4.641e-02,  3.154e-01,  1.498e-01,  4.564e-01,  4.641e-02,  3.154e-01,  1.498e-01,  4.564e-01,  1.498e-01,  4.564e-01, -7.809e-02, -4.088e-01, -7.809e-02, -4.088e-01, -3.839e-01,
                           -9.966e-03, -7.809e-02, -4.088e-01, -3.839e-01, -9.966e-03, -3.839e-01, -9.966e-03,  1.099e-01,  3.649e-01,  1.099e-01,  3.649e-01,  2.747e-01,  4.139e-02,  1.099e-01,  3.649e-01,  2.747e-01,  4.139e-02,  2.747e-01,  4.139e-02,  2.900e-01, -
                           2.305e-01,  2.900e-01, -2.305e-01, -2.029e-01, -2.088e-01,  2.900e-01, -2.305e-01, -2.029e-01, -2.088e-01, -2.029e-01, -
                           2.088e-01, -4.090e-03, -3.095e-01, -4.090e-03, -3.095e-01, -
                           2.737e-01, -1.215e-01, -4.090e-03, -3.095e-01, -2.737e-01, -1.215e-01,
                           -2.737e-01, -1.215e-01,  9.103e-02, -2.244e-01,  9.103e-02, -2.244e-01, -1.178e-01,  1.500e-01,  9.103e-02, -2.244e-01, -1.178e-01,  1.500e-01, -1.178e-01,  1.500e-01,  2.900e-01, -2.374e-01,  2.900e-01, -2.374e-01,  8.281e-02, -2.414e-01,  2.900e-01, -
                           2.374e-01,  8.281e-02, -2.414e-01,  8.281e-02, -2.414e-01,  4.522e-01,  4.519e-01,  4.522e-01,  4.519e-01, -
                           4.993e-01,  2.742e-02,  4.522e-01,  4.519e-01, -4.993e-01,  2.742e-02, -
                           4.993e-01,  2.742e-02,  7.981e-02, -3.723e-01,  7.981e-02,
                           -3.723e-01, -3.755e-01,  3.598e-01,  7.981e-02, -3.723e-01, -3.755e-01,  3.598e-01, -3.755e-01,  3.598e-01], device=device, dtype=dtype)

    triplets = (row_inds, col_inds, values)
    return triplets



def test_block_diagonalize(get_test_triplets, device, dtype):
    num_points = 20
    num_handles = 2
    num_objects = 3
    wp_sparse_stacked_B = [physics.simplicits.precomputed.sparse_lbs_matrix(torch.rand(
        num_points, num_handles, device=device, dtype=dtype), torch.rand(num_points, 3, device=device, dtype=dtype)) for _ in range(num_objects)]

    dense_B = _wp_bsr_to_torch_bsr(_block_diagonalize(
        wp_sparse_stacked_B)).to_dense()

    # Test Block diagonalization
    torch_block_wise_B = torch.block_diag(*[_wp_bsr_to_torch_bsr(
        wp_sparse_stacked_B[i]).to_dense() for i in range(num_objects)])
    check_allclose(torch_block_wise_B, dense_B)


def test_warp_bsr_to_wp_triplets(get_test_triplets, device, dtype):

    triplets = get_test_triplets

    # Create torch sparse tensor from triplets
    coo_test = torch.sparse_coo_tensor(torch.stack(
        (triplets[0], triplets[1])), triplets[2], size=(60, 24))

    # Create warp sparse tensor from triplets
    warp_test = wps.bsr_zeros(60, 24, block_type=wp.float32)
    wps.bsr_set_from_triplets(dest=warp_test,
                              rows=wp.from_torch(triplets[0], dtype=wp.int32),
                              columns=wp.from_torch(
                                  triplets[1], dtype=wp.int32),
                              values=wp.from_torch(triplets[2], dtype=wp.float32))

    # Check that warp sets the correct sparse matrix
    check_allclose(coo_test.to_dense(),
                          _wp_bsr_to_torch_bsr(warp_test).to_dense())

    # Now convert back to triplets and check that they are the same as the original triplets
    wp_triplets = _wp_bsr_to_wp_triplets(warp_test)
    check_allclose(triplets[0], wp.to_torch(wp_triplets[0]))
    check_allclose(triplets[1], wp.to_torch(wp_triplets[1]))
    check_allclose(triplets[2], wp.to_torch(wp_triplets[2]))



def test_wp_bsr_to_torch_bsr(device, dtype):
    # Given a warp bsr sparse matrix, test that the conversion to torch bsr sparse matrix is correct
    num_handles = 2
    num_points = 20

    # Create warp sparse matrix from triplets
    n = num_points
    m = num_handles * 12

    # get random points in numpy
    sim_pts = torch.rand(n, 3, device=device, dtype=dtype)
    sim_weights = torch.rand(n, m, device=device, dtype=dtype)

    dense_mat = physics.simplicits.precomputed.lbs_matrix(
        sim_pts, sim_weights).to(dtype)

    # Convert to torch BSR
    warp_sparse_mat = physics.simplicits.precomputed.sparse_lbs_matrix(
        sim_weights, sim_pts)

    # Convert both to dense and compare
    torchified_mat = _wp_bsr_to_torch_bsr(warp_sparse_mat).to_dense().to(dtype)

    for i in range(n):
        for j in range(m):
            check_allclose(
                torchified_mat[i, j], dense_mat[i, j])

    # Also check that bsr_to_torch is correct
    # Test bsr to torch
    check_allclose(_bsr_to_torch(
        warp_sparse_mat).to_dense().to(dtype), dense_mat)
    check_allclose(_bsr_to_torch(
        warp_sparse_mat).to_dense().to(dtype), torchified_mat)
