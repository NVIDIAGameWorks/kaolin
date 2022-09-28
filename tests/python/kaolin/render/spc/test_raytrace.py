# Copyright (c) 2021,22 NVIDIA CORPORATION & AFFILIATES.
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

from kaolin.ops.spc import scan_octrees, generate_points, bits_to_uint8

from kaolin.render.spc import unbatched_raytrace, mark_pack_boundaries

class TestRaytrace:
    @pytest.fixture(autouse=True)
    def octree(self):
        bits_t = torch.tensor([
            [0, 0, 0, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1], [ 0, 0, 0, 0, 0, 0, 0, 0]],
            device='cuda', dtype=torch.float)
        return bits_to_uint8(torch.flip(bits_t, dims=(-1,)))

    @pytest.fixture(autouse=True)
    def length(self, octree):
        return torch.tensor([len(octree)], dtype=torch.int)

    @pytest.fixture(autouse=True)
    def max_level_pyramids_exsum(self, octree, length):
        return scan_octrees(octree, length)

    @pytest.fixture(autouse=True)
    def pyramid(self, max_level_pyramids_exsum):
        return max_level_pyramids_exsum[1].squeeze(0)

    @pytest.fixture(autouse=True)
    def exsum(self, max_level_pyramids_exsum):
        return max_level_pyramids_exsum[2]

    @pytest.fixture(autouse=True)
    def point_hierarchy(self, octree, pyramid, exsum):
        return generate_points(octree, pyramid.unsqueeze(0), exsum)

    def _generate_rays_origin (self, height, width, camera_dist):
        """Make simple orthographic rays"""
        camera_dist = torch.tensor(camera_dist, dtype=torch.float, device='cuda')
        camera_dist = camera_dist.repeat(height, width)
        ii, jj = torch.meshgrid(
            torch.arange(height, dtype=torch.float, device='cuda'),
            torch.arange(width, dtype=torch.float, device='cuda'))
        ii = (ii * 2. / height) - (height - 1.) / height
        jj = (jj * 2. / width) - (width - 1.) / width
        return torch.stack([ii, jj, camera_dist], dim=-1).reshape(-1, 3)

    def test_raytrace_positive(self, octree, point_hierarchy, pyramid, exsum):
        height = 4
        width = 4
        direction = torch.tensor([[0., 0., 1.]], dtype=torch.float,
                                 device='cuda').repeat(height * width , 1)
        origin = self._generate_rays_origin(height, width, -3)
        ridx, pidx = unbatched_raytrace(
            octree, point_hierarchy, pyramid, exsum, origin, direction, 2, return_depth=False)

        expected_nuggets = torch.tensor([
            [ 0,  5],
            [ 0,  6],
            [ 0, 13],
            [ 0, 14],
            [ 1,  7],
            [ 1,  8],
            [ 2, 15],
            [ 4,  9],
            [ 4, 10],
            [ 5, 11],
            [ 5, 12]], device='cuda', dtype=torch.int)
        assert torch.equal(ridx, expected_nuggets[...,0])
        assert torch.equal(pidx, expected_nuggets[...,1])

    def test_raytrace_negative(self, octree, point_hierarchy, pyramid, exsum):
        height = 4
        width = 4
        direction = torch.tensor([[0., 0., -1.]], dtype=torch.float,
                                 device='cuda').repeat(height * width , 1)
        origin = self._generate_rays_origin(height, width, 3)
        ridx, pidx = unbatched_raytrace(
            octree, point_hierarchy, pyramid, exsum, origin, direction, 2, return_depth=False)

        expected_nuggets = torch.tensor([
            [ 0, 14],
            [ 0, 13],
            [ 0,  6],
            [ 0,  5],
            [ 1,  8],
            [ 1,  7],
            [ 2, 15],
            [ 4, 10],
            [ 4,  9],
            [ 5, 12],
            [ 5, 11]], device='cuda', dtype=torch.int)
        assert torch.equal(ridx, expected_nuggets[...,0])
        assert torch.equal(pidx, expected_nuggets[...,1])

    def test_raytrace_none(self, octree, point_hierarchy, pyramid, exsum):
        height = 4
        width = 4
        direction = torch.tensor([[0., 0., 1.]], dtype=torch.float,
                                 device='cuda').repeat(height * width , 1)
        origin = self._generate_rays_origin(height, width, 3)
        ridx, pidx, depth = unbatched_raytrace(
            octree, point_hierarchy, pyramid, exsum, origin, direction, 2, return_depth=True, with_exit=True)

        expected_nuggets = torch.zeros((0, 2), device='cuda', dtype=torch.int)
        expected_depths = torch.zeros((0, 2), device='cuda', dtype=torch.float) 
        assert torch.equal(ridx, expected_nuggets[...,0])
        assert torch.equal(pidx, expected_nuggets[...,1])
        assert torch.equal(depth, expected_depths)

    def test_raytrace_coarser(self, octree, point_hierarchy, pyramid, exsum):
        height = 4
        width = 4
        direction = torch.tensor([[0., 0., 1.]], dtype=torch.float,
                                 device='cuda').repeat(height * width , 1)
        origin = self._generate_rays_origin(height, width, -3)
        ridx, pidx = unbatched_raytrace(
            octree, point_hierarchy, pyramid, exsum, origin, direction, 1, return_depth=False)

        expected_nuggets = torch.tensor([
            [ 0,  1],
            [ 0,  2],
            [ 1,  1],
            [ 1,  2],
            [ 2,  3],
            [ 3,  3],
            [ 4,  1],
            [ 4,  2],
            [ 5,  1],
            [ 5,  2],
            [ 6,  3],
            [ 7,  3],
            [ 8,  4],
            [ 9,  4],
            [12,  4],
            [13,  4]], device='cuda', dtype=torch.int)
        assert torch.equal(ridx, expected_nuggets[...,0])
        assert torch.equal(pidx, expected_nuggets[...,1])

    def test_raytrace_with_depth(self, octree, point_hierarchy, pyramid, exsum):
        height = 4
        width = 4
        direction = torch.tensor([[0., 0., -1.]], dtype=torch.float,
                                 device='cuda').repeat(height * width , 1)
        origin = self._generate_rays_origin(height, width, 3)
        ridx, pidx, depth = unbatched_raytrace(
            octree, point_hierarchy, pyramid, exsum, origin, direction, 2, return_depth=True)

        expected_nuggets = torch.tensor([
            [ 0, 14],
            [ 0, 13],
            [ 0,  6],
            [ 0,  5],
            [ 1,  8],
            [ 1,  7],
            [ 2, 15],
            [ 4, 10],
            [ 4,  9],
            [ 5, 12],
            [ 5, 11]], device='cuda', dtype=torch.int)
        assert torch.equal(ridx, expected_nuggets[...,0])
        assert torch.equal(pidx, expected_nuggets[...,1])

        expected_depth = torch.tensor([
            [2.0],
            [2.5],
            [3.0],
            [3.5],
            [3.0],
            [3.5],
            [3.5],
            [3.0],
            [3.5],
            [3.0],
            [3.5]], device='cuda', dtype=torch.float)
        assert torch.equal(depth, expected_depth)

    def test_raytrace_with_depth_with_exit(self, octree, point_hierarchy, pyramid, exsum):
        height = 4
        width = 4
        direction = torch.tensor([[0., 0., -1.]], dtype=torch.float,
                                 device='cuda').repeat(height * width , 1)
        origin = self._generate_rays_origin(height, width, 3)
        ridx, pidx, depth = unbatched_raytrace(
            octree, point_hierarchy, pyramid, exsum, origin, direction, 2, return_depth=True, with_exit=True)

        expected_nuggets = torch.tensor([
            [ 0, 14],
            [ 0, 13],
            [ 0,  6],
            [ 0,  5],
            [ 1,  8],
            [ 1,  7],
            [ 2, 15],
            [ 4, 10],
            [ 4,  9],
            [ 5, 12],
            [ 5, 11]], device='cuda', dtype=torch.int)
        assert torch.equal(ridx, expected_nuggets[...,0])
        assert torch.equal(pidx, expected_nuggets[...,1])
        
        expected_depth = torch.tensor([
            [2.0, 2.5],
            [2.5, 3.0],
            [3.0, 3.5],
            [3.5, 4.0],
            [3.0, 3.5],
            [3.5, 4.0],
            [3.5, 4.0],
            [3.0, 3.5],
            [3.5, 4.0],
            [3.0, 3.5],
            [3.5, 4.0]], device='cuda', dtype=torch.float)
        
        assert torch.equal(depth, expected_depth)

    @pytest.mark.parametrize('return_depth,with_exit', [(False, False), (True, False), (True, True)])
    def test_raytrace_inside(self, octree, point_hierarchy, pyramid, exsum, return_depth, with_exit):
        height = 4
        width = 4
        direction = torch.tensor([[0., 0., -1.]], dtype=torch.float,
                                 device='cuda').repeat(height * width , 1)
        origin = self._generate_rays_origin(height, width, 0.9)
        outputs = unbatched_raytrace(
            octree, point_hierarchy, pyramid, exsum, origin, direction, 2,
            return_depth=return_depth, with_exit=with_exit)

        ridx = outputs[0]
        pidx = outputs[1]

        expected_nuggets = torch.tensor([
            [ 0, 13],
            [ 0,  6],
            [ 0,  5],
            [ 1,  8],
            [ 1,  7],
            [ 2, 15],
            [ 4, 10],
            [ 4,  9],
            [ 5, 12],
            [ 5, 11]], device='cuda', dtype=torch.int)
        assert torch.equal(ridx, expected_nuggets[...,0])
        assert torch.equal(pidx, expected_nuggets[...,1])
        if return_depth:
            depth = outputs[2]
            if with_exit:
                expected_depth = torch.tensor([
                    [0.4, 0.9],
                    [0.9, 1.4],
                    [1.4, 1.9],
                    [0.9, 1.4],
                    [1.4, 1.9],
                    [1.4, 1.9],
                    [0.9, 1.4],
                    [1.4, 1.9],
                    [0.9, 1.4],
                    [1.4, 1.9]], device='cuda', dtype=torch.float)
            else:
                expected_depth = torch.tensor([
                    [0.4],
                    [0.9],
                    [1.4],
                    [0.9],
                    [1.4],
                    [1.4],
                    [0.9],
                    [1.4],
                    [0.9],
                    [1.4]], device='cuda', dtype=torch.float)
            assert torch.allclose(depth, expected_depth)

    def test_ambiguous_raytrace(self):
        # TODO(ttakikawa):
        # Since 0.10.0, the behaviour of raytracing exactly between voxels 
        # has been changed from no hits at all to hitting all adjacent voxels.
        # This has numerical ramifications because it may cause instability / error 
        # in the estimation of optical thickness in the volume rendering process 
        # among other issues. However, we have found that this doesn't lead to any 
        # obvious visual errors, whereas the no hit case causes speckle noise.
        # We will eventually do a more thorough analysis of the numerical consideration of this
        # behaviour, but for now we choose to prevent obvious visual errors.

        octree = torch.tensor([255], dtype=torch.uint8, device='cuda')
        length = torch.tensor([1], dtype=torch.int32)
        max_level, pyramids, exsum = scan_octrees(octree, length)
        point_hierarchy = generate_points(octree, pyramids, exsum)
        origin = torch.tensor([
            [0., 0., 3.],
            [3., 3., 3.]], dtype=torch.float, device='cuda')
        direction = torch.tensor([
            [0., 0., -1.],
            [-1. / 3., -1. / 3., -1. / 3.]], dtype=torch.float, device='cuda')
        ridx, pidx, depth = unbatched_raytrace(
            octree, point_hierarchy, pyramids[0], exsum, origin, direction, 1, return_depth=True)
        expected_nuggets = torch.tensor([
            [0, 2],
            [0, 1],
            [0, 4],
            [0, 6],
            [0, 3],
            [0, 5],
            [0, 8],
            [0, 7],
            [1, 8], 
            [1, 1]], device='cuda', dtype=torch.int)
        assert torch.equal(ridx, expected_nuggets[...,0])
        assert torch.equal(pidx, expected_nuggets[...,1])

    def test_mark_first_positive(self, octree, point_hierarchy, pyramid, exsum):
        height = 4
        width = 4
        direction = torch.tensor([[0., 0., 1.]], dtype=torch.float,
                                 device='cuda').repeat(height * width , 1)
        origin = self._generate_rays_origin(height, width, -3)
        ridx, pidx = unbatched_raytrace(
            octree, point_hierarchy, pyramid, exsum, origin, direction, 2, return_depth=False)
        first_hits = mark_pack_boundaries(ridx)
        expected_first_hits = torch.tensor([1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                                           device='cuda', dtype=torch.bool)
        assert torch.equal(first_hits, expected_first_hits)

    def test_mark_first_negative(self, octree, point_hierarchy, pyramid, exsum):
        height = 4
        width = 4
        direction = torch.tensor([[0., 0., -1.]], dtype=torch.float,
                                 device='cuda').repeat(height * width , 1)
        origin = self._generate_rays_origin(height, width, 3)
        ridx, pidx = unbatched_raytrace(
            octree, point_hierarchy, pyramid, exsum, origin, direction, 2, return_depth=False)
        first_hits = mark_pack_boundaries(ridx)
        expected_first_hits = torch.tensor([1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                                           device='cuda', dtype=torch.bool)
        assert torch.equal(first_hits, expected_first_hits)

