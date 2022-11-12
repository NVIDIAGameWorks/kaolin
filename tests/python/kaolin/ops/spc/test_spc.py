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

import torch
from kaolin.ops.spc.uint8 import bits_to_uint8, uint8_bits_sum, uint8_to_bits
from kaolin.ops.random import random_spc_octrees
from kaolin.rep import Spc

from kaolin.ops.spc import scan_octrees, generate_points, to_dense, feature_grids_to_spc
from kaolin.ops.spc import unbatched_query, unbatched_points_to_octree
from kaolin.ops.spc import unbatched_get_level_points, unbatched_make_dual, unbatched_make_trinkets
from kaolin.ops.spc import points_to_corners

from kaolin.utils.testing import FLOAT_TYPES, with_seed, check_tensor

@pytest.mark.parametrize('device', ['cuda'])
class TestSimpleBase:
    @pytest.fixture(autouse=True)
    def octrees(self, device):
        bits_t = torch.tensor([
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0],

            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1],  [0, 1, 0, 1, 0, 1, 0, 1]],
            device='cuda', dtype=torch.float)
        return bits_to_uint8(torch.flip(bits_t, dims=(-1,)))

    @pytest.fixture(autouse=True)
    def lengths(self):
        return torch.tensor([6, 5], dtype=torch.int)

    def test_scan_octrees(self, octrees, lengths):
        expected_pyramids = torch.tensor(
            [[[1, 2, 3, 3, 0], [0, 1, 3, 6, 9]],
             [[1, 1, 3, 13, 0], [0, 1, 2, 5, 18]]], dtype=torch.int32)
        expected_exsum = torch.tensor(
            [0, 2, 4, 5, 6, 7, 8, 0, 1, 4, 5, 13, 17],
            dtype=torch.int32, device='cuda')
        max_level, pyramids, exsum = scan_octrees(octrees, lengths)
        assert max_level == 3
        assert torch.equal(pyramids, expected_pyramids)
        assert torch.equal(exsum, expected_exsum)

    def test_generate_points(self, octrees, lengths):
        max_level, pyramids, exsum = scan_octrees(octrees, lengths)
        expected_point_hierarchies = torch.tensor([
            [0, 0, 0],
            [0, 0, 0], [1, 0, 0],
            [0, 0, 1], [0, 1, 0], [3, 0, 1],
            [1, 1, 3], [1, 3, 1], [6, 1, 3],

            [0, 0, 0],
            [1, 1, 1],
            [3, 2, 2], [3, 2, 3], [3, 3, 2],
            [7, 4, 5], [6, 4, 6], [6, 4, 7], [6, 5, 6], [6, 5, 7], [7, 4, 6], \
                [7, 4, 7], [7, 5, 6], [7, 5, 7], [6, 6, 4], [6, 7, 4], \
                [7, 6, 4], [7, 7, 4]
            ], device='cuda', dtype=torch.int16)

        point_hierarchies = generate_points(octrees, pyramids, exsum)

        assert torch.equal(point_hierarchies, expected_point_hierarchies)


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('max_level', [1, 4])
@pytest.mark.parametrize('batch_size', [1, 3])
class TestBase:
    @pytest.fixture(autouse=True)
    def octrees_and_lengths(self, batch_size, max_level, device):
        return random_spc_octrees(batch_size, max_level, device)

    @pytest.fixture(autouse=True)
    def octrees(self, octrees_and_lengths):
        return octrees_and_lengths[0]

    @pytest.fixture(autouse=True)
    def lengths(self, octrees_and_lengths):
        return octrees_and_lengths[1]

    def test_scan_octrees(self, octrees, lengths, max_level):
        # Naive implementation
        num_childrens_per_node = uint8_bits_sum(octrees).cpu()
        octree_start_idx = 0
        num_childrens_per_level = []
        levels_first_idx = []
        expected_exsum = torch.zeros((num_childrens_per_node.shape[0] +
                                      lengths.shape[0], ),
                                      dtype=torch.int32)
        for bs, length in enumerate(lengths):
            cur_num_childrens_per_node = \
                num_childrens_per_node[octree_start_idx:octree_start_idx + length]
            num_childrens_per_level.append([1])
            levels_first_idx.append([0])
            for i in range(max_level):
                cur_idx = levels_first_idx[-1][-1]
                cur_num_childrens = num_childrens_per_level[-1][-1]
                num_childrens_per_level[-1].append(int(torch.sum(
                    cur_num_childrens_per_node[cur_idx:cur_idx + cur_num_childrens])))
                levels_first_idx[-1].append(cur_idx + cur_num_childrens)
            levels_first_idx[-1].append(levels_first_idx[-1][-1] +
                                        num_childrens_per_level[-1][-1])
            num_childrens_per_level[-1].append(0);
            # + bs + 1 because torch.cumsum is inclusive
            expected_exsum[octree_start_idx + bs + 1:octree_start_idx + bs + 1 + length] = \
                torch.cumsum(cur_num_childrens_per_node, dim=0)
            octree_start_idx += length
        num_childrens_per_level = torch.tensor(num_childrens_per_level, dtype=torch.int32)
        levels_first_idx = torch.tensor(levels_first_idx, dtype=torch.int32)
        expected_pyramids = torch.stack([num_childrens_per_level, levels_first_idx], dim=1)
        expected_exsum = expected_exsum.cuda()

        out_level, pyramids, exsum = scan_octrees(octrees, lengths)

        assert out_level == max_level
        assert torch.equal(pyramids, expected_pyramids)
        assert torch.equal(exsum, expected_exsum)

    def test_generate_points(self, octrees, lengths, max_level):
        out_level, pyramids, exsum = scan_octrees(octrees, lengths)
        point_hierarchies = generate_points(octrees, pyramids, exsum)
        expected_point_hierarchies = []
        bits_t = uint8_to_bits(octrees).reshape(-1, 2, 2, 2).cpu()
        octree_first_idx = 0
        for bs, length in enumerate(lengths):
            expected_point_hierarchies.append(torch.tensor([[0, 0, 0]], dtype=torch.long))
            cur_bits_t = bits_t[octree_first_idx:octree_first_idx + length]
            offsets = torch.tensor([[0,0,0]], dtype=torch.int32)
            for i in range(max_level):
                next_offset = []
                cur_level_num_nodes = pyramids[bs, 0, i]
                level_first_idx = pyramids[bs, 1, i]
                for cur_level_node_idx in range(cur_level_num_nodes):
                    node_bits = cur_bits_t[level_first_idx + cur_level_node_idx]
                    offset = offsets[cur_level_node_idx]
                    point_coords = torch.nonzero(node_bits, as_tuple=False) + offset.unsqueeze(0)
                    expected_point_hierarchies.append(point_coords)
                    next_offset.append(point_coords * 2)
                offsets = torch.cat(next_offset, dim=0)
            octree_first_idx += length
        expected_point_hierarchies = torch.cat(expected_point_hierarchies,
                                               dim=0).cuda().short()
        assert torch.equal(point_hierarchies, expected_point_hierarchies)


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('max_level', [4, 6, 1])
@pytest.mark.parametrize('batch_size', [1])
class TestTrinkets:
    @pytest.fixture(autouse=True)
    def octrees_and_lengths(self, batch_size, max_level, device):
        return random_spc_octrees(batch_size, max_level, device)

    @pytest.fixture(autouse=True)
    def octrees(self, octrees_and_lengths):
        return octrees_and_lengths[0]

    @pytest.fixture(autouse=True)
    def lengths(self, octrees_and_lengths):
        return octrees_and_lengths[1]

    def test_unbatched_make_trinkets(self, octrees, lengths, max_level):
        out_level, pyramid, exsum = scan_octrees(octrees, lengths)
        point_hierarchy = generate_points(octrees, pyramid, exsum)
        pyramid = pyramid[0]
        point_hierarchy_dual, pyramid_dual = unbatched_make_dual(point_hierarchy, pyramid)
        trinkets, parents = unbatched_make_trinkets(point_hierarchy, pyramid, point_hierarchy_dual, pyramid_dual)
        
        for i in range(0, max_level+1):
            _idx = pyramid_dual[1, i] + unbatched_get_level_points(trinkets, pyramid, i)
            pts = point_hierarchy_dual.index_select(0, _idx.view(-1)).view(-1, 8, 3)
            expected_pts = points_to_corners(unbatched_get_level_points(point_hierarchy, pyramid, i))
            assert torch.equal(pts, expected_pts)

        assert parents[0] == -1

        for i in range(1, max_level+1):
            parent = point_hierarchy.index_select(0, unbatched_get_level_points(parents, pyramid, i))
        assert torch.equal(parent, torch.div(unbatched_get_level_points(point_hierarchy, pyramid, i), 2, rounding_mode='trunc'))


class TestQuery:
    def test_query(self):
        points = torch.tensor(
            [[3,2,0],
             [3,1,1],
             [0,0,0],
             [3,3,3]], device='cuda', dtype=torch.short)
        level = 2
        resolution = 2**level
        octree = unbatched_points_to_octree(points, level)
        length = torch.tensor([len(octree)], dtype=torch.int32)
        _, pyramid, prefix = scan_octrees(octree, length)

        query_points = torch.tensor(
            [[3,2,0],
             [3,1,1],
             [0,0,0],
             [3,3,3],
             [2,2,2],
             [1,1,1]], device='cuda', dtype=torch.short)
        query_coords_float = (2.0 * (query_points.float() / resolution) - 1.0)
        query_coords_int = query_points

        point_hierarchy = generate_points(octree, pyramid, prefix)

        results_float = unbatched_query(octree, prefix, query_coords_float, 2)
        results_int = unbatched_query(octree, prefix, query_coords_int, 2)

        expected_results = torch.tensor(
            [7,6,5,8,-1,-1], dtype=torch.long, device='cuda')

        assert torch.equal(expected_results, results_float)
        assert torch.equal(expected_results, results_int)
        assert torch.equal(point_hierarchy[results_float[:-2]], query_points[:-2])
        assert torch.equal(point_hierarchy[results_int[:-2]], query_points[:-2])

    def test_query_flooredge(self):
        points = torch.tensor(
            [[0,0,0]], device='cuda', dtype=torch.short)
        level = 1
        octree = unbatched_points_to_octree(points, level)
        length = torch.tensor([len(octree)], dtype=torch.int32)
        _, pyramid, prefix = scan_octrees(octree, length)
        query_coords = torch.tensor(
            [[-3.0,-3.0,-3.0],
             [-2.5,-2.5,-2.5],
             [2.5,2.5,2.5],
             [3.0,3.0,3.0],
             [0.0,0.0,0.0],
             [0.5,0.5,0.5]], device='cuda', dtype=torch.float)
        results = unbatched_query(octree, prefix, query_coords, 0)
        expected_results = torch.tensor(
            [-1,-1,-1,-1,0,0], dtype=torch.long, device='cuda')
        assert torch.equal(expected_results, results)

    def test_query_multiscale(self):
        points = torch.tensor(
            [[3,2,0],
             [3,1,1],
             [0,0,0],
             [3,3,3]], device='cuda', dtype=torch.short)
        level = 3
        resolution = 2**level
        octree = unbatched_points_to_octree(points, level)
        length = torch.tensor([len(octree)], dtype=torch.int32)
        _, pyramid, prefix = scan_octrees(octree, length)

        query_points = torch.tensor(
            [[3,2,0],
             [3,1,1],
             [0,0,0],
             [0,4,4],
             [3,3,3],
             [2,2,2],
             [1,1,1],
             [16,16,16]], device='cuda', dtype=torch.short)
        query_coords_float = (2.0 * (query_points.float() / resolution) - 1.0)
        query_coords_int = query_points

        point_hierarchy = generate_points(octree, pyramid, prefix)

        expected_results0 = unbatched_query(octree, prefix, query_coords_float, 0)
        expected_results1 = unbatched_query(octree, prefix, query_coords_float, 1)
        expected_results2 = unbatched_query(octree, prefix, query_coords_float, 2)
        expected_results3 = unbatched_query(octree, prefix, query_coords_float, 3)

        results03_float = unbatched_query(octree, prefix, query_coords_float, level, with_parents=True)
        results02_float = unbatched_query(octree, prefix, query_coords_float, level-1, with_parents=True)

        assert torch.equal(expected_results0, results03_float[:,0])
        assert torch.equal(expected_results1, results03_float[:,1])
        assert torch.equal(expected_results2, results03_float[:,2])
        assert torch.equal(expected_results3, results03_float[:,3])
        
        assert torch.equal(expected_results0, results02_float[:,0])
        assert torch.equal(expected_results1, results02_float[:,1])
        assert torch.equal(expected_results2, results02_float[:,2])
        
        expected_results3 = unbatched_query(octree, prefix, query_coords_int, 3)

        results03_int = unbatched_query(octree, prefix, query_coords_int, level, with_parents=True)

        assert torch.equal(expected_results3, results03_int[:,3])

class TestToDense:
    @pytest.mark.parametrize('with_spc_to_dict', [False, True])
    def test_simple(self, with_spc_to_dict):
        bits_t = torch.tensor([
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0],

            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1],  [0, 1, 0, 1, 0, 1, 0, 1]],
            device='cuda', dtype=torch.float)
        octrees = bits_to_uint8(torch.flip(bits_t, dims=(-1,)))
        lengths = torch.tensor([6, 5], dtype=torch.int)
        max_level, pyramids, exsum = scan_octrees(octrees, lengths)
        point_hierarchies = generate_points(octrees, pyramids, exsum)
        coalescent_features = torch.tensor([
            1., 2., 3.,
            4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.
        ], device='cuda', dtype=torch.float).reshape(-1, 1)

        feat_idx = torch.tensor([
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 6, 7, 6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 7, 7],
            [1, 3, 1, 4, 4, 4, 5, 5, 4, 4, 5, 5, 6, 7, 6, 7],
            [3, 1, 3, 5, 6, 7, 6, 7, 6, 7, 6, 7, 4, 4, 4, 4]
        ], dtype=torch.long)

        expected_feature_grids = torch.zeros((2, 1, 8, 8, 8), dtype=torch.float, device='cuda')
        expected_feature_grids[feat_idx[0], :, feat_idx[1], feat_idx[2], feat_idx[3]] = coalescent_features
        if with_spc_to_dict:
            feature_grids = to_dense(**Spc(octrees, lengths).to_dict(),
                                     input=coalescent_features)
        else:
            feature_grids = to_dense(point_hierarchies, pyramids, coalescent_features, max_level)

        assert torch.equal(feature_grids, expected_feature_grids)

    @pytest.mark.parametrize('max_level', [1, 4])
    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('feature_dim', [1, 4])
    def test_to_dense(self, batch_size, max_level, feature_dim):
        octrees, lengths = random_spc_octrees(batch_size, max_level, 'cuda')

        max_level, pyramids, exsum = scan_octrees(octrees, lengths)
        point_hierarchies = generate_points(octrees, pyramids, exsum)
        in_num_nodes = torch.sum(pyramids[:, 0, -2])
        coalescent_features = torch.rand((in_num_nodes, feature_dim), device='cuda',
                                         requires_grad=True)
        expected_size = 2 ** max_level
        feat_idx = []
        bs_start_idx = 0
        for bs in range(batch_size):
            start_idx = pyramids[bs, 1, -2] + bs_start_idx
            num_points = pyramids[bs, 0, -2]
            feat_idx.append(torch.nn.functional.pad(
                point_hierarchies[start_idx:start_idx + num_points],
                (1, 0), value=bs))
            bs_start_idx += pyramids[bs, 1, -1]
        feat_idx = torch.cat(feat_idx, dim=0).permute(1, 0).long()
        expected_feature_grids = torch.zeros((batch_size, feature_dim, expected_size,
                                              expected_size, expected_size), device='cuda')
        expected_feature_grids[feat_idx[0], :, feat_idx[1], feat_idx[2], feat_idx[3]] = coalescent_features

        # test forward
        feature_grids = to_dense(point_hierarchies, pyramids, coalescent_features, max_level)
        assert torch.equal(expected_feature_grids, feature_grids)

        grad_out = torch.rand_like(feature_grids)
        feature_grids.backward(grad_out)
        octrees, lengths, coalescent_expected_grad = feature_grids_to_spc(
            grad_out, torch.any(feature_grids != 0, dim=1))
        assert torch.equal(coalescent_features.grad, coalescent_expected_grad)

@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('batch_size', [1, 5])
@pytest.mark.parametrize('feature_dim', [1, 3])
@pytest.mark.parametrize('height,width,depth,threshold',
                         [(2, 2, 2, 0.1), (113, 251, 251, 0.9)])
@pytest.mark.parametrize('dtype', [torch.float])
class TestCycleConversionsFeatureGrids:
    @pytest.fixture(autouse=True)
    def expected_out_size(self, height, width, depth):
        max_level = math.ceil(math.log2(max(height, width, depth)))
        return 2 ** max_level

    @pytest.fixture(autouse=True)
    def sparsity_masks(self, batch_size, height, width, depth,
                       threshold, device):
        # We want the array to be quite sparse so even at high level
        # (near the root) there is sparsity
        return torch.rand(batch_size, height, width, depth,
                          device=device) > threshold

    @pytest.fixture(autouse=True)
    def feature_grids(self, batch_size, feature_dim, height,
                      width, depth, dtype, device):
        return torch.rand((
            batch_size,
            feature_dim,
            height,
            width,
            depth,
        ), dtype=dtype, device=device)

    @pytest.fixture(autouse=True)
    def sparse_feature_grids(self, feature_grids, sparsity_masks):
        return feature_grids * sparsity_masks.unsqueeze(1)

    @pytest.fixture(autouse=True)
    def expected_out_feature_grids(self, sparse_feature_grids, batch_size,
                                   feature_dim, height, width, depth,
                                   expected_out_size):
        out = torch.zeros((batch_size, feature_dim, expected_out_size,
                           expected_out_size, expected_out_size),
                          device='cuda',
                          dtype=sparse_feature_grids.dtype)
        out[:, :, :height, :width, :depth] = sparse_feature_grids
        return out

    def test_feature_grids_to_spc(self, sparse_feature_grids,
                                  expected_out_feature_grids,
                                  device):
        octrees, lengths, features = feature_grids_to_spc(
            sparse_feature_grids)
        assert octrees.device.type == device
        assert features.device.type == device
        octrees = octrees.cuda()
        features = features.cuda()
        max_level, pyramids, exsum = scan_octrees(octrees, lengths)
        point_hierarchies = generate_points(octrees, pyramids, exsum)
        out_feature_grids = to_dense(point_hierarchies, pyramids, features, max_level)

        assert torch.equal(out_feature_grids, expected_out_feature_grids)

    def test_feature_grids_to_spc_with_masks(self, feature_grids, sparsity_masks,
                                             expected_out_feature_grids, device):
        octrees, lengths, features = feature_grids_to_spc(feature_grids,
                                                          sparsity_masks)
        assert octrees.device.type == device
        assert features.device.type == device
        octrees = octrees.cuda()
        features = features.cuda()
        max_level, pyramids, exsum = scan_octrees(octrees, lengths)
        point_hierarchies = generate_points(octrees, pyramids, exsum)
        out_feature_grids = to_dense(point_hierarchies, pyramids, features, max_level)

        assert torch.equal(out_feature_grids, expected_out_feature_grids)

    def test_zeros(self, batch_size, feature_dim,
                   height, width, depth, dtype, device):
        feature_grids = torch.zeros((batch_size, feature_dim, height, width, depth),
                                    dtype=dtype, device=device)
        octrees, lengths, features = feature_grids_to_spc(feature_grids)
        assert torch.equal(octrees, torch.zeros((batch_size), dtype=torch.uint8,
                                                device=device))
        assert torch.equal(lengths, torch.ones((batch_size), dtype=torch.int,
                                               device='cpu'))
        assert torch.equal(features, torch.empty((0, feature_dim), dtype=dtype,
                                                 device=device))

    def test_ones(self, batch_size, feature_dim,
                  height, width, depth, dtype, device):
        feature_grids = torch.ones((batch_size, feature_dim, height, width, depth),
                                    dtype=dtype, device=device)
        octrees, lengths, features = feature_grids_to_spc(feature_grids)
        assert octrees.device.type == device
        assert features.device.type == device
        octrees = octrees.cuda()
        features = features.cuda()
        max_level, pyramids, exsum = scan_octrees(octrees, lengths)
        point_hierarchies = generate_points(octrees, pyramids, exsum)
        out_feature_grids = to_dense(point_hierarchies, pyramids, features, max_level)
        assert torch.all(out_feature_grids[:, :, :height, :width, :depth] == 1)
        assert torch.all(out_feature_grids[:, :, height:] == 0)
        assert torch.all(out_feature_grids[:, :, :, width:] == 0)
        assert torch.all(out_feature_grids[..., depth:] == 0)
