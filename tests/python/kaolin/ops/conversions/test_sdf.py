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
import sys

import torch

from kaolin.ops.conversions import sdf


class TestSdfToVoxelgrids:

    def sphere(self, points, center=0, radius=0.5):
        return torch.sum((points - center) ** 2, 1) ** 0.5 - radius

    def two_spheres(self, points):
        dis1 = self.sphere(points, 0.1, 0.4)
        dis2 = self.sphere(points, -0.1, 0.4)
        dis = torch.zeros_like(dis1)
        mask = (dis1 > 0) & (dis2 > 0)
        dis[mask] = torch.min(dis1[mask], dis2[mask])
        mask = (dis1 < 0) ^ (dis2 < 0)
        dis[mask] = torch.max(-torch.abs(dis1[mask]), -torch.abs(dis2[mask]))
        mask = (dis1 < 0) & (dis2 < 0)
        dis[mask] = torch.min(torch.abs(dis1[mask]), torch.abs(dis2[mask]))
        return dis

    def sdf_to_voxelgrids_naive(self, sdf, res):
        outputs = []
        for i_batch in range(len(sdf)):
            output = torch.ones((res, res, res))
            grid_pts = torch.nonzero(output).float() / (res - 1) - 0.5
            outputs.append((sdf[i_batch](grid_pts) <= 0).float().reshape(output.shape))
        return torch.stack(outputs)

    def test_sdf_type(self):
        with pytest.raises(TypeError,
                           match=r"Expected sdf to be list "
                                 r"but got <class 'int'>."):
            sdf.sdf_to_voxelgrids(0)
    def test_each_sdf_type(self):
        with pytest.raises(TypeError,
                           match=r"Expected sdf\[0\] to be callable "
                                 r"but got <class 'int'>."):
            sdf.sdf_to_voxelgrids([0])
    def test_bbox_center_type(self):
        with pytest.raises(TypeError,
                           match=r"Expected bbox_center to be int or float "
                                 r"but got <class 'str'>."):
            sdf.sdf_to_voxelgrids([self.sphere], bbox_center=' ')

    def test_bbox_dim_type(self):
        with pytest.raises(TypeError,
                           match=r"Expected bbox_dim to be int or float "
                                 r"but got <class 'str'>."):
            sdf.sdf_to_voxelgrids([self.sphere], bbox_dim=' ')

    def test_init_res_type(self):
        with pytest.raises(TypeError,
                           match=r"Expected init_res to be int "
                                 r"but got <class 'float'>."):
            sdf.sdf_to_voxelgrids([self.sphere], init_res=0.5)

    def test_upsampling_steps_type(self):
        with pytest.raises(TypeError,
                           match=r"Expected upsampling_steps to be int "
                                 r"but got <class 'float'>."):
            sdf.sdf_to_voxelgrids([self.sphere], upsampling_steps=0.5)

    @pytest.mark.parametrize('init_res', [4, 8, 32])
    @pytest.mark.parametrize('upsampling_steps', [0, 2, 4])
    def test_sphere(self, init_res, upsampling_steps):
        final_res = init_res * 2 ** upsampling_steps + 1
        assert(torch.equal(sdf.sdf_to_voxelgrids([self.sphere], init_res=init_res, upsampling_steps=upsampling_steps), 
                           self.sdf_to_voxelgrids_naive([self.sphere], final_res)))

    @pytest.mark.parametrize('init_res', [4, 8, 32])
    @pytest.mark.parametrize('upsampling_steps', [0, 2, 4])
    def test_two_spheres(self, init_res, upsampling_steps):
        final_res = init_res * 2 ** upsampling_steps + 1
        assert(torch.equal(sdf.sdf_to_voxelgrids([self.two_spheres], init_res=init_res, upsampling_steps=upsampling_steps), 
                           self.sdf_to_voxelgrids_naive([self.two_spheres], final_res)))
