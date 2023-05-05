# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
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
import pytest
import os
import itertools

import torch

from kaolin.utils.testing import check_allclose
from kaolin.ops.spc import points_to_morton, morton_to_points, points_to_corners, \
                           coords_to_trilinear_coeffs, quantize_points, unbatched_query, \
                           scan_octrees, unbatched_points_to_octree, generate_points, \
                           unbatched_make_trinkets, unbatched_make_dual, unbatched_interpolate_trilinear

class TestPoints:
    @pytest.fixture(autouse=True)
    def points(self):
        return torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 1, 0]], device='cuda', dtype=torch.int16)

    @pytest.fixture(autouse=True)
    def morton(self):
        return torch.tensor([0, 1, 8, 9, 2], device='cuda', dtype=torch.long)

    def test_quantize_points(self):
        x = torch.tensor([
            [-1.1, -1.1, -1.1],
            [-1., -1., -1.],
            [0., 0., 0.],
            [0.1, 0.3, 0.6],
            [0.1, -1.1, 1.1],
            [0.1, -1., 1.],
            [1., 1., 1.],
            [1.1, 1.1, 1.1]], device='cuda', dtype=torch.float)

        points = quantize_points(x, 3)
        expected_points = torch.tensor([
            [0, 0, 0],
            [0, 0, 0],
            [4, 4, 4],
            [4, 5, 6],
            [4, 0, 7],
            [4, 0, 7],
            [7, 7, 7],
            [7, 7, 7]], device='cuda', dtype=torch.int16)

        assert torch.equal(points, expected_points)
    def test_points_to_morton(self, points, morton):
        assert torch.equal(points_to_morton(points), morton)

    def test_morton_to_points(self, morton, points):
        assert torch.equal(morton_to_points(morton), points)

    def test_points_to_corners(self, points):
        expected_corners = []
        for offset in itertools.product([0, 1], repeat=3):
            expected_corners.append(points + torch.tensor([offset], device='cuda', dtype=torch.int16))
        expected_corners = torch.stack(expected_corners, dim=-2)
        assert torch.equal(points_to_corners(points), expected_corners)

    def test_coords_to_trilinear_coeffs(self, points):
        w = torch.rand(points.shape, device='cuda')
        x = points + w
        expected_coeffs = torch.stack([
            (1 - w[:, 0]) * (1 - w[:, 1]) * (1 - w[:, 2]),
            (1 - w[:, 0]) * (1 - w[:, 1]) * w[:, 2],
            (1 - w[:, 0]) * w[:, 1] * (1 - w[:, 2]),
            (1 - w[:, 0]) * w[:, 1] * w[:, 2],
            w[:, 0] * (1 - w[:, 1]) * (1 - w[:, 2]),
            w[:, 0] * (1 - w[:, 1]) * w[:, 2],
            w[:, 0] * w[:, 1] * (1 - w[:, 2]),
            w[:, 0] * w[:, 1] * w[:, 2]
        ], dim=-1)

        level = 3
        coords = (x / (2 ** level)) * 2.0 - 1.0
        check_allclose(coords_to_trilinear_coeffs(coords, points, level), expected_coeffs, rtol=1e-4, atol=1e-4)

    def test_interpolate_trilinear_forward(self, points):
        w = torch.rand(points.shape, device='cuda')
        x = torch.cat([
            points + w,
            -torch.rand((4, 3), device='cuda')
        ], dim=0)

        level = 3

        octree = unbatched_points_to_octree(points, level)
        length = torch.tensor([len(octree)], dtype=torch.int32)
        _, pyramid, prefix = scan_octrees(octree, length)
        point_hierarchy = generate_points(octree, pyramid, prefix)

        pyramid = pyramid[0]
        point_hierarchy_dual, pyramid_dual = unbatched_make_dual(point_hierarchy, pyramid)
        trinkets, parents = unbatched_make_trinkets(point_hierarchy, pyramid, point_hierarchy_dual, pyramid_dual)

        coords = (x / (2 ** level)) * 2.0 - 1.0
        pidx = unbatched_query(octree, prefix, coords, level, with_parents=False)

        feats = torch.rand([pyramid_dual[0, level], 16], device='cuda')

        corner_feats = feats.index_select(0, trinkets[pidx].view(-1)).view(-1, 8, 16)
        coeffs = coords_to_trilinear_coeffs(coords, points, level)
        expected_results = (corner_feats * coeffs[..., None]).sum(-2)
        expected_results[points.shape[0]:] = 0.

        results = unbatched_interpolate_trilinear(
            coords[:, None], pidx.int(), point_hierarchy, trinkets, feats, level
        )[:, 0]

        check_allclose(results, expected_results, rtol=1e-5, atol=1e-5)

    def test_interpolate_trilinear_forward_dtypes(self, points):
        w = torch.rand(points.shape, device='cuda')
        x = torch.cat([
            points + w,
            -torch.rand((4, 3), device='cuda')
        ], dim=0)

        level = 3

        octree = unbatched_points_to_octree(points, level)
        length = torch.tensor([len(octree)], dtype=torch.int32)
        _, pyramid, prefix = scan_octrees(octree, length)
        point_hierarchy = generate_points(octree, pyramid, prefix)

        pyramid = pyramid[0]
        point_hierarchy_dual, pyramid_dual = unbatched_make_dual(point_hierarchy, pyramid)
        trinkets, parents = unbatched_make_trinkets(point_hierarchy, pyramid, point_hierarchy_dual, pyramid_dual)

        coords = (x / (2**level)) * 2.0 - 1.0
        pidx = unbatched_query(octree, prefix, coords, level, with_parents=False)

        feats = torch.rand([pyramid_dual[0, level], 16], device='cuda')

        results_float = unbatched_interpolate_trilinear(coords[:, None], pidx.int(), point_hierarchy, trinkets, feats, level)[:, 0]
        results_double = unbatched_interpolate_trilinear(coords[:, None], pidx.int(), point_hierarchy, trinkets, feats.double(), level)[:, 0]
        results_half = unbatched_interpolate_trilinear(coords[:, None], pidx.int(), point_hierarchy, trinkets, feats.half(), level)[:, 0]

        check_allclose(results_float, results_double.float(), rtol=1e-4, atol=1e-4)
        check_allclose(results_float.half(), results_half, rtol=1e-3, atol=1e-3)

    def test_interpolate_trilinear_backward(self, points):
        w = torch.rand(points.shape, device='cuda')
        x = torch.cat([
            points + w,
            -torch.rand((4, 3), device='cuda')
        ], dim=0)

        level = 3

        octree = unbatched_points_to_octree(points, level)
        length = torch.tensor([len(octree)], dtype=torch.int32)
        _, pyramid, prefix = scan_octrees(octree, length)
        point_hierarchy = generate_points(octree, pyramid, prefix)

        pyramid = pyramid[0]
        point_hierarchy_dual, pyramid_dual = unbatched_make_dual(point_hierarchy, pyramid)
        trinkets, parents = unbatched_make_trinkets(point_hierarchy, pyramid, point_hierarchy_dual, pyramid_dual)

        coords = (x / (2 ** level)) * 2.0 - 1.0
        pidx = unbatched_query(octree, prefix, coords, level, with_parents=False)

        feats = torch.rand([pyramid_dual[0, level], 16], device='cuda')
        feats.requires_grad_(True)
        if feats.grad is not None:
            feats.grad.detach()
            feats.grad.zero_()

        corner_feats = feats.index_select(0, trinkets[pidx].view(-1)).view(-1, 8, 16)
        coeffs = coords_to_trilinear_coeffs(coords, points, level)
        expected_results = (corner_feats * coeffs[..., None]).sum(-2)
        expected_results[points.shape[0]:] = 0.

        loss = expected_results.sum()
        loss.backward()
        expected_grad = feats.grad.clone()
        
        if feats.grad is not None:
            feats.grad.detach_()
            feats.grad.zero_()

        results = unbatched_interpolate_trilinear(
            coords[:, None], pidx.int(), point_hierarchy, trinkets, feats, level
        )[:, 0]
        loss = results.sum()
        loss.backward()
        grad = feats.grad.clone()

        check_allclose(grad, expected_grad, rtol=1e-5, atol=1e-5)

    def test_interpolate_trilinear_by_coords_backward(self, points):
        w = torch.rand(points.shape, device='cuda')
        x = torch.cat([
            points + w,
            -torch.rand((4, 3), device='cuda')
        ], dim=0)

        level = 3

        octree = unbatched_points_to_octree(points, level)
        length = torch.tensor([len(octree)], dtype=torch.int32)
        _, pyramid, prefix = scan_octrees(octree, length)
        point_hierarchy = generate_points(octree, pyramid, prefix)

        pyramid = pyramid[0]
        point_hierarchy_dual, pyramid_dual = unbatched_make_dual(point_hierarchy, pyramid)
        trinkets, parents = unbatched_make_trinkets(
            point_hierarchy, pyramid, point_hierarchy_dual, pyramid_dual)

        coords = (x / (2 ** level)) * 2.0 - 1.0
        pidx = unbatched_query(octree, prefix, coords, level, with_parents=False)
        feats = torch.rand([pyramid_dual[0, level], 16], device='cuda')

        # w is the relative position inside a cell
        w = w.detach()
        w.requires_grad_(True)
        if w.grad is not None:
            w.grad.detach()
            w.grad.zero_()

        # (5, 8, 16)
        corner_feats = feats.index_select(0, trinkets[pidx].view(-1)).view(-1, 8, 16)
        corner_feats[points.shape[0]:] = 0.

        # (5, 8)
        expected_coeffs = torch.cat([torch.stack([
                (1 - w[:, 0]) * (1 - w[:, 1]) * (1 - w[:, 2]),
                (1 - w[:, 0]) * (1 - w[:, 1]) * w[:, 2],
                (1 - w[:, 0]) * w[:, 1] * (1 - w[:, 2]),
                (1 - w[:, 0]) * w[:, 1] * w[:, 2],
                w[:, 0] * (1 - w[:, 1]) * (1 - w[:, 2]),
                w[:, 0] * (1 - w[:, 1]) * w[:, 2],
                w[:, 0] * w[:, 1] * (1 - w[:, 2]),
                w[:, 0] * w[:, 1] * w[:, 2]
            ], dim=-1),
            torch.zeros((4, 8), device='cuda', dtype=torch.float)
        ], dim=0)
        expected_coeffs = expected_coeffs.requires_grad_(True)  # prevents element0 error
        expected_results = (corner_feats * expected_coeffs[..., None]).sum(1)
        expected_results[points.shape[0]:] = 0.

        loss = expected_results.sum()
        loss.backward()
        expected_grad = torch.zeros_like(x)
        expected_grad[:points.shape[0]] = w.grad.clone()

        coords.requires_grad_(True)
        if coords.grad is not None:
            coords.grad.detach()
            coords.grad.zero_()
        results = unbatched_interpolate_trilinear(
            coords[:, None], pidx.int(), point_hierarchy, trinkets, feats, level)
        loss = results[:, 0].sum()
        loss.backward()
        coords_grad = coords.grad.clone()

        assert torch.allclose(coords_grad, expected_grad, rtol=1e-4, atol=1e-3)

    def test_interpolate_trilinear_by_coords_toggleable(self, points):
        # Test that features only grad does not generate coords grad
        w = torch.rand(points.shape, device='cuda')
        x = torch.cat([
            points + w,
            -torch.rand((4, 3), device='cuda')
        ], dim=0)


        level = 3

        octree = unbatched_points_to_octree(points, level)
        length = torch.tensor([len(octree)], dtype=torch.int32)
        _, pyramid, prefix = scan_octrees(octree, length)
        point_hierarchy = generate_points(octree, pyramid, prefix)

        pyramid = pyramid[0]
        point_hierarchy_dual, pyramid_dual = unbatched_make_dual(point_hierarchy, pyramid)
        trinkets, parents = unbatched_make_trinkets(point_hierarchy, pyramid, point_hierarchy_dual, pyramid_dual)

        coords = (x / (2 ** level)) * 2.0 - 1.0
        pidx = unbatched_query(octree, prefix, coords, level, with_parents=False)
        feats = torch.rand([pyramid_dual[0, level], 16], device='cuda')

        feats.requires_grad_(True)
        coords.requires_grad_(False)
        results = unbatched_interpolate_trilinear(coords[:, None], pidx.int(), point_hierarchy, trinkets, feats, level)
        loss = results[:, 0].sum()
        loss.backward()

        assert coords.grad is None
