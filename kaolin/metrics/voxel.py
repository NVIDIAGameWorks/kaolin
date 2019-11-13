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

import torch
import kaolin as kal


def iou(pred, gt, thresh=.5, reduction='mean'):
    r""" Computes IoU across two voxel grids

    Arguments:
            pred (torch.Tensor): predicted (binary) voxel grid
            gt (torch.Tensor): ground-truth (binary) voxel grid
            thresh (float): value to threshold the prediction with

    Returns:
            iou (torch.Tensor): the intersection over union value

    Example:
            >>> pred = torch.rand(32, 32, 32)
            >>> gt = torch.rand(32, 32, 32) *2. // 1
            >>> loss = iou(pred, gt)
            >>> loss
            tensor(0.3338)
    """
    pred = pred.clone()
    pred[pred <= thresh] = 0
    pred[pred > thresh] = 1

    pred = pred.view(-1).byte()
    gt = gt.view(-1).byte()
    assert pred.shape == gt.shape, 'pred and gt must have the same shape'

    iou = torch.sum(torch.mul(pred, gt).float()) / \
        torch.sum((pred + gt).clamp(min=0, max=1).float())

    return iou
