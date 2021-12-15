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

import torch

from kaolin.metrics import tetmesh


class TestTetMeshMetrics:

    def test_tetrahedron_volume(self):
        tetrahedrons = torch.tensor([[[[0.5000, 0.5000, 0.4500],
                                       [0.4500, 0.5000, 0.5000],
                                       [0.4750, 0.4500, 0.4500],
                                       [0.5000, 0.5000, 0.5000]]]])
        assert torch.allclose(tetmesh.tetrahedron_volume(tetrahedrons), torch.tensor([[-2.0833e-05]]))

    def test_amips(self):
        tetrahedrons = torch.tensor([[[
                                      [1.7000, 2.3000, 4.4500],
                                      [3.4800, 0.2000, 5.3000],
                                      [4.9000, 9.4500, 6.4500],
                                      [6.2000, 8.5000, 7.1000]],
                                      [[-1.3750, 1.4500, 3.2500],
                                       [4.9000, 1.8000, 2.7000],
                                       [3.6000, 1.9000, 2.3000],
                                       [1.5500, 1.3500, 2.9000]]],
                                     [[[1.7000, 2.3000, 4.4500],
                                       [3.4800, 0.2000, 5.3000],
                                       [4.9000, 9.4500, 6.4500],
                                       [6.2000, 8.5000, 7.1000]],
                                      [[-1.3750, 1.4500, 3.2500],
                                       [4.9000, 1.8000, 2.7000],
                                       [3.6000, 1.9000, 2.3000],
                                       [1.5500, 1.3500, 2.9000]]]])
        inverse_offset_matrix = torch.tensor([[[[-1.1561, -1.1512, -1.9049],
                                                [1.5138, 1.0108, 3.4302],
                                                [1.6538, 1.0346, 4.2223]],
                                               [[2.9020, -1.0995, -1.8744],
                                                [1.1554, 1.1519, 1.7780],
                                                [-0.0766, 1.6350, 1.1064]]],
                                              [[[-0.9969, 1.4321, -0.3075],
                                                [-1.3414, 1.5795, -1.6571],
                                                [-0.1775, -0.4349, 1.1772]],
                                               [[-1.1077, -1.2441, 1.8037],
                                                [-0.5722, 0.1755, -2.4364],
                                                [-0.5263, 1.5765, 1.5607]]]])
        torch.allclose(tetmesh.amips(tetrahedrons, inverse_offset_matrix), torch.tensor([[13042.3408], [2376.2517]]))

    def test_equivolume(self):
        tetrahedrons = torch.tensor([[[[0.5000, 0.5000, 0.7500],
                                       [0.4500, 0.8000, 0.6000],
                                       [0.4750, 0.4500, 0.2500],
                                       [0.5000, 0.3000, 0.3000]],
                                      [[0.4750, 0.4500, 0.2500],
                                       [0.5000, 0.9000, 0.3000],
                                       [0.4500, 0.4000, 0.9000],
                                       [0.4500, 0.4500, 0.7000]]],
                                     [[[0.7000, 0.3000, 0.4500],
                                       [0.4800, 0.2000, 0.3000],
                                       [0.9000, 0.4500, 0.4500],
                                       [0.2000, 0.5000, 0.1000]],
                                      [[0.3750, 0.4500, 0.2500],
                                       [0.9000, 0.8000, 0.7000],
                                       [0.6000, 0.9000, 0.3000],
                                       [0.5500, 0.3500, 0.9000]]]])
        assert torch.allclose(tetmesh.equivolume(tetrahedrons, pow=4), torch.tensor([[2.2898e-15], [2.9661e-10]]))
