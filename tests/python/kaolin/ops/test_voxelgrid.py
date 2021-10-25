# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import torch
import random
from kaolin.ops import voxelgrid as vg
from kaolin.utils.testing import BOOL_TYPES, FLOAT_TYPES, INT_TYPES


@pytest.mark.parametrize('device,dtype', FLOAT_TYPES + BOOL_TYPES)
class TestDownsample:

    def test_scale_val_1(self, device, dtype):
        # The scale should be smaller or equal to the size of the input.
        with pytest.raises(ValueError,
                           match="Downsample ratio must be less than voxelgrids "
                           "shape of 6 at index 2, but got 7."):
            voxelgrids = torch.ones([2, 6, 6, 6], device=device, dtype=dtype)
            vg.downsample(voxelgrids, [1, 2, 7])

    def test_scale_val_2(self, device, dtype):
        # Every element in the scale should be greater or equal to one.
        with pytest.raises(ValueError,
                           match="Downsample ratio must be at least 1 "
                           "along every dimension but got -1 at "
                           "index 0."):
            voxelgrids = torch.ones([2, 6, 6, 6], device=device, dtype=dtype)
            vg.downsample(voxelgrids, [-1, 3, 2])

    def test_voxelgrids_dim(self, device, dtype):
        # The dimension of voxelgrids should be 4 (batched).
        with pytest.raises(ValueError,
                           match="Expected voxelgrids to have 4 dimensions "
                                 "but got 3 dimensions."):
            voxelgrids = torch.ones([6, 6, 6], device=device, dtype=dtype)
            vg.downsample(voxelgrids, [2, 2, 2])

    def test_scale_dim(self, device, dtype):
        # The dimension of scale should be 3 if it is a list.
        with pytest.raises(ValueError,
                           match="Expected scale to have 3 dimensions "
                                 "but got 2 dimensions."):
            voxelgrids = torch.ones([2, 6, 6, 6], device=device, dtype=dtype)
            vg.downsample(voxelgrids, [2, 2])

        with pytest.raises(TypeError,
                           match="Expected scale to be type list or int "
                                 "but got <class 'str'>."):
            voxelgrids = torch.ones([2, 6, 6, 6], device=device, dtype=dtype)
            vg.downsample(voxelgrids, "2")

    def test_output_size(self, device, dtype):
        # The size of the output should be input.shape / scale
        voxelgrids = torch.ones([3, 6, 6, 6], device=device, dtype=dtype)
        output = vg.downsample(voxelgrids, [1, 2, 3])
        assert (output.shape == torch.Size([3, 6, 3, 2]))

    def test_output_batch(self, device, dtype):
        if dtype == torch.bool:
            pytest.skip("This test won't work for torch.bool.")

        # The size of the batched input shoud be correct.
        # For example, if the input size is [2, 6, 6, 6],
        # Scale is [3, 3, 3], the output size should be [2, 2, 2, 2]
        # Also, test the function is numerically correct
        voxelgrid1 = torch.ones([4, 4, 4], device=device, dtype=dtype)
        voxelgrid2 = torch.ones((4, 4, 4), device=device, dtype=dtype)
        voxelgrid2[1, :2] = 0.8
        voxelgrid2[1, 2:] = 0.4
        voxelgrid2[3] = 0
        batched_voxelgrids = torch.stack((voxelgrid1, voxelgrid2))
        output = vg.downsample(batched_voxelgrids, [2, 2, 2])

        expected1 = torch.ones((2, 2, 2), device=device, dtype=dtype)
        expected2 = torch.tensor([[[0.9, 0.9],
                                   [0.7, 0.7]],

                                  [[0.5000, 0.5000],
                                   [0.5000, 0.5000]]], device=device, dtype=dtype)
        expected = torch.stack((expected1, expected2))
        assert torch.allclose(output, expected)


    def test_bool_input(self, device, dtype):
        if dtype != torch.bool:
            pytest.skip("This test is only for torch.bool.")

        voxelgrids = torch.ones((2, 4, 4, 4), device=device, dtype=dtype)
        voxelgrids[:, :, 1, :] = 0
        voxelgrids[:, :, 3, :] = 0

        output = vg.downsample(voxelgrids, 2)

        expected_dtype = torch.half if device == "cuda" else torch.float
        expected = torch.ones((2, 2, 2, 2), device=device, dtype=expected_dtype) * 0.5
        assert torch.equal(output, expected)


@pytest.mark.parametrize('device,dtype', FLOAT_TYPES + BOOL_TYPES)
@pytest.mark.parametrize('mode', ['wide', 'thin'])
class TestExtractSurface:

    def test_valid_mode(self, device, dtype, mode):
        voxelgrids = torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
        with pytest.raises(ValueError, match='mode "this is not a valid mode" is not supported.'):
            vg.extract_surface(voxelgrids, mode="this is not a valid mode")

    def test_input_size(self, device, dtype, mode):
        voxelgrids = torch.ones((3, 3, 3), device=device, dtype=dtype)
        with pytest.raises(ValueError, 
                           match="Expected voxelgrids to have 4 dimensions " 
                                 "but got 3 dimensions."):
            vg.extract_surface(voxelgrids, mode=mode)

    def test_output_value(self, device, dtype, mode):
        voxelgrids = torch.ones((2, 4, 4, 4), device=device, dtype=dtype)
        voxelgrids[0, 0, 0, 0] = 0  # Remove a voxel on a corner
        voxelgrids[1, 1, 0, 0] = 0  # Remove a voxel on an edge
        expected = voxelgrids.clone().bool()

        surface = vg.extract_surface(voxelgrids, mode=mode)

        expected[0, 1:3, 1:3, 1:3] = 0
        expected[1, 1:3, 1:3, 1:3] = 0
        if mode == 'wide':
            expected[0, 1, 1, 1] = 1
            expected[1, 1:3, 1, 1] = 1

        assert torch.equal(surface, expected)


@pytest.mark.parametrize('device,dtype', FLOAT_TYPES + BOOL_TYPES)
class TestExtractOdms:

    def test_handmade_input(self, device, dtype):
        # The input is hand-made.
        voxelgrid1 = torch.tensor([[[1, 0, 0],
                                    [0, 1, 1],
                                    [0, 1, 1]],

                                   [[1, 0, 0],
                                    [0, 1, 1],
                                    [0, 0, 1]],

                                   [[0, 1, 0],
                                    [1, 1, 0],
                                    [0, 0, 1]]], device=device, dtype=dtype)
        voxelgrid2 = voxelgrid1.transpose(0, 1)

        expected1 = torch.tensor([[[2, 0, 0],
                                   [2, 0, 0],
                                   [1, 1, 0]],

                                  [[0, 1, 1],
                                   [0, 1, 2],
                                   [1, 0, 2]],

                                  [[2, 0, 0],
                                   [2, 1, 0],
                                   [1, 1, 0]],

                                  [[0, 1, 1],
                                   [0, 1, 1],
                                   [1, 0, 2]],

                                  [[1, 0, 3],
                                   [0, 0, 1],
                                   [3, 2, 0]],

                                  [[0, 2, 3],
                                   [2, 0, 0],
                                   [3, 0, 0]]], device=device, dtype=torch.long)

        expected2 = torch.tensor([[[2, 2, 1],
                                   [0, 0, 1],
                                   [0, 0, 0]],

                                  [[0, 0, 1],
                                   [1, 1, 0],
                                   [1, 2, 2]],

                                  [[1, 0, 3],
                                   [0, 0, 1],
                                   [3, 2, 0]],

                                  [[0, 2, 3],
                                   [2, 0, 0],
                                   [3, 0, 0]],

                                  [[2, 0, 0],
                                   [2, 1, 0],
                                   [1, 1, 0]],

                                  [[0, 1, 1],
                                   [0, 1, 1],
                                   [1, 0, 2]]], device=device, dtype=torch.long)

        voxelgrids = torch.stack([voxelgrid1, voxelgrid2])
        expected = torch.stack([expected1, expected2])
        output = vg.extract_odms(voxelgrids)

        assert torch.equal(output, expected)
        assert torch.equal(output, expected)

@pytest.mark.parametrize('device', ['cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
class TestFill:

    def test_complex_value(self, device, dtype):
        # The center of the voxelgrids shoud be filled. Other places unchanged
        voxelgrid1 = torch.zeros((5, 5, 5), dtype=dtype, device=device)
        voxelgrid1[1:4, 1:4, 1:4] = 1
        voxelgrid1[2, 2, 2] = 0

        voxelgrid2 = torch.ones((5, 5, 5), dtype=dtype, device=device)
        voxelgrid2[1:4, 1:4, 1:4] = 0

        # With 0 in the middle, but not enclosed.
        voxelgrid3 = torch.zeros((5, 5, 5), dtype=dtype, device=device)
        voxelgrid3[1:4, 1:4, 1:4] = 1
        voxelgrid3[2, 2, 2:4] = 0

        batch_voxelgrids = torch.stack((voxelgrid1, voxelgrid2, voxelgrid3))


        output = vg.fill(batch_voxelgrids)

        # Only center is changed for batch sample 1.
        expected1 = voxelgrid1
        expected1[2, 2, 2] = 1

        # The batch sample 2 should be all ones.
        expected2 = torch.ones((5, 5, 5), dtype=dtype, device=device)

        # The batch sample 3 should be unchanged.
        expected3 = voxelgrid3

        expected = torch.stack((expected1, expected2, expected3)).type(torch.bool)

        assert torch.equal(output, expected)

    def test_voxelgrids_dim(self, device, dtype):
        # The dimension of voxelgrids should be 4 (batched).
        with pytest.raises(ValueError,
                           match="Expected voxelgrids to have 4 dimensions "
                                 "but got 3 dimensions."):
            voxelgrids = torch.ones([6, 6, 6], device=device, dtype=dtype)
            vg.fill(voxelgrids)

@pytest.mark.parametrize('device,dtype', INT_TYPES)
class TestProjectOdms:

    def test_batch_match(self, device, dtype):
        # The batch size of voxelgrids and odms must match.
        with pytest.raises(ValueError,
                           match="Expected voxelgrids and odms' batch size to be the same, "
                                 "but got 2 for odms and 3 for voxelgrid."):
            voxelgrids = torch.ones((3, 3, 3, 3), device=device, dtype=dtype)
            odms = torch.ones((2, 6, 3, 3), device=device, dtype=dtype)
            vg.project_odms(odms, voxelgrids)

    def test_dimension_match(self, device, dtype):
        # The dimension of voxelgrids and odms must match
        with pytest.raises(ValueError,
                           match="Expected voxelgrids and odms' dimension size to be the same, "
                                 "but got 3 for odms and 4 for voxelgrid."):
            voxelgrids = torch.ones((2, 4, 4, 4), device=device, dtype=dtype)
            odms = torch.ones((2, 6, 3, 3), device=device, dtype=dtype)
            vg.project_odms(odms, voxelgrids)

    def test_empty_filled_odms(self, device, dtype):
        # If the input is an empty odms, the output should be a filled voxel grid
        odms1 = torch.zeros((6, 3, 3), device=device, dtype=dtype)

        # If the input odms is a filled odms. the output should be an empty voxel grid
        odms2 = torch.ones((6, 3, 3), device=device, dtype=dtype) * 3

        odms = torch.stack((odms1, odms2))
        voxelgrids = vg.project_odms(odms)

        assert torch.equal(voxelgrids[0], torch.ones((3, 3, 3), device=device, dtype=torch.bool))
        assert torch.equal(voxelgrids[1], torch.zeros((3, 3, 3), device=device, dtype=torch.bool))

    def test_handmade_input_vote1(self, device, dtype):
        # The input is hand-made.
        odms = torch.tensor([[[[2, 0, 0],
                               [2, 0, 0],
                               [1, 1, 0]],

                              [[0, 1, 1],
                               [0, 1, 2],
                               [1, 0, 2]],

                              [[2, 0, 0],
                               [2, 1, 0],
                               [1, 1, 0]],

                              [[0, 1, 1],
                               [0, 1, 1],
                               [1, 0, 2]],

                              [[1, 0, 3],
                               [0, 0, 1],
                               [3, 2, 0]],

                              [[0, 2, 3],
                               [2, 0, 0],
                               [3, 0, 0]]]], device=device, dtype=dtype)

        expected = torch.tensor([[[[1, 0, 0],
                                   [0, 1, 1],
                                   [0, 1, 1]],

                                  [[1, 0, 0],
                                   [0, 1, 1],
                                   [0, 0, 1]],

                                  [[0, 1, 0],
                                   [1, 1, 0],
                                   [0, 0, 1]]]], device=device, dtype=torch.bool)

        output = vg.project_odms(odms)
        assert torch.equal(output, expected)

    def test_handmade_input_vote4(self, device, dtype):
        # The input is hand-made.
        odms = torch.tensor([[[[2, 0, 0],
                               [2, 0, 0],
                               [1, 1, 0]],

                              [[0, 1, 1],
                               [0, 1, 2],
                               [1, 0, 2]],

                              [[2, 0, 0],
                               [2, 1, 0],
                               [1, 1, 0]],

                              [[0, 1, 1],
                               [0, 1, 1],
                               [1, 0, 2]],

                              [[1, 0, 3],
                               [0, 0, 1],
                               [3, 2, 0]],

                              [[0, 2, 3],
                               [2, 0, 0],
                               [3, 0, 0]]]], device=device, dtype=dtype)

        expected_votes = torch.tensor([[[[1, 1, 0],
                                         [1, 1, 1],
                                         [0, 1, 1]],

                                        [[1, 1, 0],
                                         [1, 1, 1],
                                         [0, 1, 1]],

                                        [[1, 1, 0],
                                         [1, 1, 1],
                                         [0, 1, 1]]]], device=device, dtype=torch.bool)

        output_votes = vg.project_odms(odms, votes=4)
        assert torch.equal(output_votes, expected_votes)
