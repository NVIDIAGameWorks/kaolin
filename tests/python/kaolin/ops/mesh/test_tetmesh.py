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

import pytest
import torch

from kaolin.ops.mesh import tetmesh


class TestTetMeshOps:

    def test_validate_tetrahedrons_wrong_ndim(self):
        wrong_ndim_tet = torch.randn(size=(2, 2))
        with pytest.raises(Exception):
            tetmesh._validate_tetrahedrons(wrong_ndim_tet)

    def test_validate_tetrahedrons_wrong_third_dimension(self):
        wrong_third_dim_tet = torch.randn(size=(2, 2, 3))
        with pytest.raises(Exception):
            tetmesh._validate_tetrahedrons(wrong_third_dim_tet)

    def test_validate_tetrahedrons_wrong_fourth_dimension(self):
        wrong_fourth_dim_tet = torch.randn(size=(2, 2, 4, 2))
        with pytest.raises(Exception):
            tetmesh._validate_tetrahedrons(wrong_fourth_dim_tet)

    def test_inverse_vertices_offset(self):
        tetrahedrons = torch.tensor([[[[-0.0500, 0.0000, 0.0500],
                                       [-0.0250, -0.0500, 0.0000],
                                       [0.0000, 0.0000, 0.0500],
                                       [0.5000, 0.5000, 0.4500]]]])
        oracle = torch.tensor([[[[0.0000, 20.0000, 0.0000],
                                 [79.9999, -149.9999, 10.0000],
                                 [-99.9999, 159.9998, -10.0000]]]])
        torch.allclose(tetmesh.inverse_vertices_offset(tetrahedrons), oracle)
