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
from kaolin.physics.utils.torch_utilities import standard_transform_to_relative, create_projection_matrix, hess_reduction, torch_bsr_to_torch_triplets
from kaolin.utils.testing import check_allclose


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_standard_transform_to_relative(device, dtype):
    # Test 4x4 transform
    transform = torch.eye(4, device=device, dtype=dtype)
    transform[0, 3] = 1.0
    transform[1, 3] = 2.0
    transform[2, 3] = 3.0
    relative_transform = standard_transform_to_relative(transform)
    
    expected = torch.zeros((3, 4), device=device, dtype=dtype)
    expected[0, 3] = 1.0
    expected[1, 3] = 2.0 
    expected[2, 3] = 3.0
    check_allclose(relative_transform, expected)

    # Test 3x4 transform
    transform_3x4 = torch.zeros((3, 4), device=device, dtype=dtype)
    transform_3x4[:3, :3] = torch.eye(3, device=device, dtype=dtype)
    transform_3x4[0, 3] = 1.0
    transform_3x4[1, 3] = 2.0
    transform_3x4[2, 3] = 3.0
    relative_transform = standard_transform_to_relative(transform_3x4)
    check_allclose(relative_transform, expected)

    # Test invalid input shape raises error
    with pytest.raises(ValueError):
        transform_invalid = torch.eye(3, device=device, dtype=dtype)
        standard_transform_to_relative(transform_invalid)
        
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_create_projection_matrix(device, dtype):
    # Test basic case - removing single DOF
    num_dofs = 5
    kin_dofs = torch.tensor([2], device=device)
    P = create_projection_matrix(num_dofs, kin_dofs)
    
    expected = torch.tensor([[1., 0., 0., 0., 0.],
                           [0., 1., 0., 0., 0.],
                           [0., 0., 0., 1., 0.],
                           [0., 0., 0., 0., 1.]], device=device)
    check_allclose(P, expected)

    # Test removing multiple DOFs
    num_dofs = 6
    kin_dofs = torch.tensor([1, 3, 5], device=device)
    P = create_projection_matrix(num_dofs, kin_dofs)
    
    expected = torch.tensor([[1., 0., 0., 0., 0., 0.],
                           [0., 0., 1., 0., 0., 0.],
                           [0., 0., 0., 0., 1., 0.]], device=device)
    check_allclose(P, expected)

    # Test removing no DOFs
    num_dofs = 3
    kin_dofs = torch.tensor([], device=device, dtype=torch.int64)
    P = create_projection_matrix(num_dofs, kin_dofs)
    
    expected = torch.eye(3, device=device)
    check_allclose(P, expected)

    # Test removing all DOFs
    num_dofs = 4
    kin_dofs = torch.arange(num_dofs, device=device)
    P = create_projection_matrix(num_dofs, kin_dofs)

    expected = torch.empty((0, num_dofs), device=device)
    check_allclose(P, expected)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('block_size', [3, 9])
def test_hess_reduction(device, dtype, block_size):
    r"""hess_reduction computes Ja^T H Jb for block-diagonal H."""
    torch.manual_seed(0)
    n_blocks, n_dofs = 5, 8
    Ja = torch.randn(n_blocks * block_size, n_dofs, device=device, dtype=dtype)
    H = torch.randn(n_blocks, block_size, block_size, device=device, dtype=dtype)

    out = hess_reduction(Ja, H)
    assert out.shape == (n_dofs, n_dofs)

    # Compare against an explicit dense block-diagonal assembly.
    H_dense = torch.zeros(n_blocks * block_size, n_blocks * block_size,
                          device=device, dtype=dtype)
    for i in range(n_blocks):
        s = i * block_size
        H_dense[s:s + block_size, s:s + block_size] = H[i]
    check_allclose(out, Ja.T @ H_dense @ Ja, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_hess_reduction_two_sided(device, dtype):
    r"""Passing dense_Jb gives Ja^T H Jb rather than Ja^T H Ja."""
    torch.manual_seed(0)
    n_blocks, block_size, n_dofs = 4, 3, 6
    Ja = torch.randn(n_blocks * block_size, n_dofs, device=device, dtype=dtype)
    Jb = torch.randn(n_blocks * block_size, n_dofs, device=device, dtype=dtype)
    H = torch.randn(n_blocks, block_size, block_size, device=device, dtype=dtype)

    H_dense = torch.zeros(n_blocks * block_size, n_blocks * block_size,
                          device=device, dtype=dtype)
    for i in range(n_blocks):
        s = i * block_size
        H_dense[s:s + block_size, s:s + block_size] = H[i]

    check_allclose(hess_reduction(Ja, H, Jb), Ja.T @ H_dense @ Jb,
                   atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_hess_reduction_preallocated_matches(device, dtype):
    r"""The out=/hj_out= path must match the allocating path exactly.

    These kwargs exist so Hessian assembly is allocation-free under CUDA graph
    capture, where allocating inside a conditional graph node body is illegal.
    """
    torch.manual_seed(0)
    n_blocks, block_size, n_dofs = 5, 3, 8
    Ja = torch.randn(n_blocks * block_size, n_dofs, device=device, dtype=dtype)
    H = torch.randn(n_blocks, block_size, block_size, device=device, dtype=dtype)

    expected = hess_reduction(Ja, H)

    out = torch.zeros(n_dofs, n_dofs, device=device, dtype=dtype)
    hj_out = torch.zeros(n_blocks, block_size, n_dofs, device=device, dtype=dtype)
    returned = hess_reduction(Ja, H, out=out, hj_out=hj_out)

    check_allclose(out, expected)
    # Must return the same object it was handed, not a copy.
    assert returned is out

    # Reusing the buffers overwrites rather than accumulates -- the capturable path
    # relies on this across Newton iterations.
    hess_reduction(Ja, H, out=out, hj_out=hj_out)
    check_allclose(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="torch.cuda.memory_stats is the only sound instrument here")
def test_hess_reduction_out_is_allocation_free():
    r"""No allocation may occur when out= and hj_out= are supplied.

    Allocation inside a CUDA graph conditional-node body is illegal, so this property
    is what lets Hessian assembly be captured.

    Instrument choice matters. ``TorchDispatchMode`` pops the mode while running the op,
    so it never observes the ``at::empty`` an out-of-place kernel performs in C++ below
    the Python dispatch key -- it scores an implementation that ignores ``out=``/``hj_out=``
    and ends in ``out.copy_(...)`` as 0 allocations, i.e. it cannot fail. It is fine for
    dispatcher-visible allocations (it did correctly catch ``solve_ex``'s two
    ``new_empty`` calls) but blind to exactly the class that matters here.
    ``memory_stats`` counts every request through torch's allocator; verified to report
    0 for the correct implementation and 2 for that mutation.
    """
    torch.manual_seed(0)
    n_blocks, block_size, n_dofs = 4, 3, 6
    Ja = torch.randn(n_blocks * block_size, n_dofs, device='cuda')
    H = torch.randn(n_blocks, block_size, block_size, device='cuda')
    out = torch.zeros(n_dofs, n_dofs, device='cuda')
    hj_out = torch.zeros(n_blocks, block_size, n_dofs, device='cuda')

    hess_reduction(Ja, H, out=out, hj_out=hj_out)  # warm up any lazy init
    torch.cuda.synchronize()

    before = torch.cuda.memory_stats()['allocation.all.allocated']
    hess_reduction(Ja, H, out=out, hj_out=hj_out)
    torch.cuda.synchronize()
    n_alloc = torch.cuda.memory_stats()['allocation.all.allocated'] - before

    assert n_alloc == 0, \
        f"hess_reduction made {n_alloc} allocation(s) despite out=/hj_out="
