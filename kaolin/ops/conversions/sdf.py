# Copyright (c) 2019,21 NVIDIA CORPORATION & AFFILIATES.
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
#
#
# Occupancy Networks
#
# Copyright 2019 Lars Mescheder, Michael Oechsle, Michael Niemeyer, Andreas Geiger, Sebastian Nowozin
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

import torch
import numpy as np

from . import mise

__all__ = ['sdf_to_voxelgrids']

def sdf_to_voxelgrids(sdf, bbox_center=0., bbox_dim=1., init_res=32, upsampling_steps=0):
    r"""Converts SDFs to voxelgrids. 

    For each SDF returns a voxel grid with resolution 
    :math:`init\_res * 2 ^ {upsampling\_steps} + 1` 
    (so the underlying voxel resolution is
    :math:`init\_res * 2 ^ {upsampling\_steps}`)
    where each grid point holds a binary value 
    determined by the sign of the SDF at the location
    of the grid point after normalizing the voxel grid
    to the bounding box defined by bbox_center and bbox_dim.

    This solution is largely borrowed from "Multiresolution IsoSurface Extraction (MISE)" 
    proposed in the CVPR 2019 paper "Occupancy Networks: Learning 3D Reconstruction in Function Space": 
    https://arxiv.org/abs/1906.02739. Instead of evaluating SDF values of all grid points at high 
    resolution, this function incrementally builds an octree and only evaluate dense grid points 
    around the surface.

    Args:
        sdf (list[callable]):
            A list of callable that takes 3D coordinates as a :class:`torch.Tensor`, of shape
            :math:`(\text{num_points}, 3)` and output the N corresponding SDF values
            as a :class:`torch.Tensor`, of shape :math:`(\text{num_points})`.
        bbox_center (optional, float):
            Center of the surface's bounding box. Default: 0.
        bbox_dim (optional, float):
            Largest dimension of the surface's bounding box. Default: 1.
        init_res (optional, int):
            The initial resolution of the voxelgrids, should be
            large enough to properly define the surface. Default: 32.
        upsampling_steps (optional, int):
            Number of times the initial resolution will be doubled. Default: 0.

    Returns:
        (torch.Tensor):
            Binary voxelgrids, of shape
            :math:`(\text{batch_size}, \text{init_res} * 2 ^ \text{upsampling_steps} + 1)`.

    Example:
        >>> def sphere(points):
        ...     return torch.sum(points ** 2, 1) ** 0.5 - 0.5
        >>> sdf_to_voxelgrids([sphere], init_res=4)
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                 [[0., 0., 0., 0., 0.],
                  [0., 1., 1., 1., 0.],
                  [0., 1., 1., 1., 0.],
                  [0., 1., 1., 1., 0.],
                  [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                 [[0., 0., 1., 0., 0.],
                  [0., 1., 1., 1., 0.],
                  [1., 1., 1., 1., 1.],
                  [0., 1., 1., 1., 0.],
                  [0., 0., 1., 0., 0.]],
        <BLANKLINE>
                 [[0., 0., 0., 0., 0.],
                  [0., 1., 1., 1., 0.],
                  [0., 1., 1., 1., 0.],
                  [0., 1., 1., 1., 0.],
                  [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                 [[0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.]]]])

    """
    if not isinstance(bbox_center, (int, float)):
        raise TypeError(f"Expected bbox_center to be int or float "
                        f"but got {type(bbox_center)}.")
    if not isinstance(bbox_dim, (int, float)):
        raise TypeError(f"Expected bbox_dim to be int or float "
                        f"but got {type(bbox_dim)}.")
    if not isinstance(init_res, int):
        raise TypeError(f"Expected init_res to be int "
                        f"but got {type(init_res)}.")
    if not isinstance(upsampling_steps, int):
        raise TypeError(f"Expected upsampling_steps to be int "
                        f"but got {type(upsampling_steps)}.")
    if not isinstance(sdf, list):
        raise TypeError(f"Expected sdf to be list "
                        f"but got {type(sdf)}.")
    voxels = []
    for i_batch in range(len(sdf)):
        if not callable(sdf[i_batch]):
            raise TypeError(f"Expected sdf[{i_batch}] to be callable "
                            f"but got {type(sdf[i_batch])}.")
        mesh_extractor = mise.MISE(
            init_res, upsampling_steps, .5)

        points = mesh_extractor.query()
        while points.shape[0] != 0:
            # Query points
            pointsf = torch.FloatTensor(points)
            # Normalize to bounding box
            pointsf = pointsf / (mesh_extractor.resolution)
            pointsf = bbox_dim * (pointsf - 0.5 + bbox_center)
            values = sdf[i_batch](pointsf) <= 0
            values = values.data.cpu().numpy().astype(np.float64)
            mesh_extractor.update(points, values)
            points = mesh_extractor.query()

        voxels.append(torch.FloatTensor(mesh_extractor.to_dense()))

    return torch.stack(voxels)
