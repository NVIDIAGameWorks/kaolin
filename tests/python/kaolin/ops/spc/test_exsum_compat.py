# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

import kaolin
from kaolin.ops.spc import (scan_octrees, generate_points, unbatched_query,
                            unbatched_points_to_octree, bits_to_uint8)
from kaolin.ops.spc.exsum_compat import (ensure_current_exsum, current_to_legacy,
                                          octree_byte_lengths_from_pyramid)
from kaolin.rep import Spc


@pytest.fixture
def batched_octrees():
    bits_t = torch.tensor([
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]],
        device='cuda', dtype=torch.float)
    return bits_to_uint8(torch.flip(bits_t, dims=(-1,)))


@pytest.fixture
def batched_lengths():
    return torch.tensor([6, 5], dtype=torch.int)


@pytest.fixture
def single_octree():
    points = torch.tensor([[3, 2, 0], [3, 1, 1], [3, 3, 3]],
                          device='cuda', dtype=torch.short)
    return unbatched_points_to_octree(points, 2)


class TestScanOctrees:
    def test_default_is_current_layout(self, batched_octrees, batched_lengths):
        _, _, exsum = scan_octrees(batched_octrees, batched_lengths)
        assert exsum.shape[0] == batched_octrees.shape[0]  # num_bytes, not +batch

    def test_legacy_opt_in_warns_and_pads(self, batched_octrees, batched_lengths):
        _, _, current = scan_octrees(batched_octrees, batched_lengths)
        with pytest.warns(DeprecationWarning):
            _, _, legacy = scan_octrees(batched_octrees, batched_lengths,
                                        legacy_exsum=True)
        assert legacy.shape[0] == batched_octrees.shape[0] + batched_lengths.shape[0]
        # legacy block per octree == cat([0], current_block)
        expected = current_to_legacy(current, batched_lengths)
        assert torch.equal(legacy, expected)


class TestConversionHelpers:
    def test_roundtrip_batched(self, batched_octrees, batched_lengths):
        _, _, current = scan_octrees(batched_octrees, batched_lengths)
        legacy = current_to_legacy(current, batched_lengths)
        with pytest.warns(DeprecationWarning):
            back = ensure_current_exsum(legacy, batched_lengths, "test")
        assert torch.equal(back, current)

    def test_current_passthrough_no_warning(self, batched_octrees, batched_lengths):
        _, _, current = scan_octrees(batched_octrees, batched_lengths)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning -> failure
            out = ensure_current_exsum(current, batched_lengths, "test")
        assert out is current

    def test_byte_lengths_from_pyramid(self, batched_octrees, batched_lengths):
        _, pyramid, _ = scan_octrees(batched_octrees, batched_lengths)
        byte_lengths = octree_byte_lengths_from_pyramid(pyramid)
        assert torch.equal(byte_lengths.cpu(), batched_lengths.long())

    def test_bad_length_raises(self, batched_octrees, batched_lengths):
        _, _, current = scan_octrees(batched_octrees, batched_lengths)
        bogus = torch.cat([current, current.new_zeros(3)])
        with pytest.raises(ValueError):
            ensure_current_exsum(bogus, batched_lengths, "test")


class TestConsumersAcceptLegacy:
    def test_generate_points_legacy_matches_current(self, batched_octrees, batched_lengths):
        _, pyramid, current = scan_octrees(batched_octrees, batched_lengths)
        legacy = current_to_legacy(current, batched_lengths)
        ph_current = generate_points(batched_octrees, pyramid, current)
        with pytest.warns(DeprecationWarning):
            ph_legacy = generate_points(batched_octrees, pyramid, legacy)
        assert torch.equal(ph_current, ph_legacy)

    def test_unbatched_query_legacy_matches_current(self, single_octree):
        length = torch.tensor([len(single_octree)], dtype=torch.int32)
        _, _, current = scan_octrees(single_octree, length)
        legacy = current_to_legacy(current, length)
        coords = torch.tensor([[3, 2, 0]], device='cuda', dtype=torch.short)
        out_current = unbatched_query(single_octree, current, coords, 2)
        with pytest.warns(DeprecationWarning):
            out_legacy = unbatched_query(single_octree, legacy, coords, 2)
        assert torch.equal(out_current, out_legacy)

    def test_unbatched_raytrace_legacy_matches_current(self, single_octree):
        length = torch.tensor([len(single_octree)], dtype=torch.int32)
        level, pyramid, current = scan_octrees(single_octree, length)
        point_hierarchy = generate_points(single_octree, pyramid, current)
        legacy = current_to_legacy(current, length)
        origin = torch.tensor([[0., 0., -3.]], device='cuda')
        direction = torch.tensor([[0., 0., 1.]], device='cuda')
        ray_current = kaolin.render.spc.unbatched_raytrace(
            single_octree, point_hierarchy, pyramid[0], current, origin, direction, level)
        with pytest.warns(DeprecationWarning):
            ray_legacy = kaolin.render.spc.unbatched_raytrace(
                single_octree, point_hierarchy, pyramid[0], legacy, origin, direction, level)
        assert torch.equal(ray_current[1], ray_legacy[1])


class TestSpcAcceptsLegacy:
    def test_spc_legacy_normalized_and_warns(self, batched_octrees, batched_lengths):
        _, _, current = scan_octrees(batched_octrees, batched_lengths)
        legacy = current_to_legacy(current, batched_lengths)
        with pytest.warns(DeprecationWarning):
            spc = Spc(batched_octrees, batched_lengths, exsum=legacy)
        assert torch.equal(spc._exsum, current)

    def test_spc_current_no_warning(self, batched_octrees, batched_lengths):
        _, _, current = scan_octrees(batched_octrees, batched_lengths)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            spc = Spc(batched_octrees, batched_lengths, exsum=current)
        assert torch.equal(spc._exsum, current)
