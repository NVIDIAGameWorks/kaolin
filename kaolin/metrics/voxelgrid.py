# Copyright (c) 2019,20 NVIDIA CORPORATION & AFFILIATES.
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


def iou(pred, gt):
    r"""Computes IoU across two voxelgrids

    Arguments:
        pred (torch.Tensor): predicted (binary) voxelgrids, of shape
                             :math:`(\text{batch_size}, \text{X}, \text{Y}, \text{Z})`.
        gt (torch.Tensor): ground-truth (binary) voxelgrids, of shape
                           :math:`(\text{batch_size}, \text{X}, \text{Y}, \text{Z})`.

    Returns:
        (torch.FloatTensor): the intersection over union value.
    Example:
        >>> pred = torch.tensor([[[[0., 0.],
        ...                        [1., 1.]],
        ...                       [[1., 1.],
        ...                        [1., 1.]]]])
        >>> gt = torch.ones((1,2,2,2))
        >>> iou(pred, gt)
        tensor([0.7500])
    """
    if pred.shape != gt.shape:
        raise ValueError(
            f"Expected predicted voxelgrids and ground truth voxelgrids to have "
            f"the same shape, but got {pred.shape} for predicted and {gt.shape} for ground truth.")

    pred = pred.bool()
    gt = gt.bool()

    intersection = torch.sum(torch.logical_and(pred, gt), dim=(1, 2, 3)).float()
    union = torch.sum(torch.logical_or(pred, gt), dim=(1, 2, 3)).float()

    return intersection / union
