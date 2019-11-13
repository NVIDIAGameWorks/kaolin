# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#
# 
# Multi-View Silhouette and Depth Decomposition for High Resolution 3D Object Representation components
#
# Copyright (c) 2019 Edward Smith
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional

import torch

from kaolin import helpers


class VoxelGrid(object):
    """Base class to hold (regular) voxel grids. """

    def __init__(self, voxels: Optional[torch.Tensor] = None,
                 copy: Optional[bool] = False):
        r"""Initialize a voxel grid, given a tensor of voxel `features`.

        Args:
            voxels (torch.Tensor, optional): Tensor containing voxel features
                (shape: Any shape that has >= 3 dims).
            copy (bool, optional): Whether or not to create a deep copy of the
                Tensor(s) used to initialize class member(s).

        Note:
            By default, the created VoxelGrid object stores a reference to the
            input `voxels` tensor. To create a deep copy of the voxels, set the
            `copy` argument to `True`.

        """
        super(VoxelGrid, self).__init__()
        if voxels is None:
            self.voxels = None
        else:
            helpers._assert_tensor(voxels)
            helpers._assert_dim_ge(voxels, 3)
            self.voxels = voxels.clone() if copy else voxels
