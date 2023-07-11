# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

from __future__ import annotations
import torch


def center_points(points: torch.FloatTensor, normalize: bool = False, eps=1e-6):
    r"""Returns points centered at the origin for every pointcloud. If `normalize` is
    set, will also normalize each point cloud spearately to the range of [-0.5, 0.5].
    Note that each point cloud is centered individually.

    Args:
        points (torch.FloatTensor): point clouds of shape :math:`(\text{batch_size}, \text{num_points}, 3)`,
         (other channel numbers supported).
        normalize (bool): if true, will also normalize each point cloud to be in the range [-0.5, 0.5]
        eps (float): eps to use to avoid division by zero when normalizing

    Return:
        (torch.FloatTensor) modified points with same shape, device and dtype as input
    """
    assert len(points.shape) == 3, f'Points have unexpected shape {points.shape}'

    vmin = points.min(dim=1, keepdim=True)[0]
    vmax = points.max(dim=1, keepdim=True)[0]
    vmid = (vmin + vmax) / 2
    res = points - vmid
    if normalize:
        den = (vmax - vmin).max(dim=-1, keepdim=True)[0].clip(min=eps)
        res = res / den
    return res
