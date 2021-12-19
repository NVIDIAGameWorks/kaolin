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

def mask_iou(lhs_mask, rhs_mask):
    r"""Compute the Intersection over Union of two segmentation masks.

    Args:
        lhs_mask (torch.FloatTensor):
            A segmentation mask, of shape
            :math:`(\text{batch_size}, \text{height}, \text{width})`.
        rhs_mask (torch.FloatTensor):
            A segmentation mask, of shape
            :math:`(\text{batch_size}, \text{height}, \text{width})`.

    Returns:
        (torch.FloatTensor): The IoU loss, as a torch scalar.
    """
    batch_size, height, width = lhs_mask.shape
    assert rhs_mask.shape == lhs_mask.shape
    sil_mul = lhs_mask * rhs_mask
    sil_add = lhs_mask + rhs_mask
    iou_up = torch.sum(sil_mul.reshape(batch_size, -1), dim=1)
    iou_down = torch.sum((sil_add - sil_mul).reshape(batch_size, -1), dim=1)
    iou_neg = iou_up / (iou_down + 1e-10)
    mask_loss = 1.0 - torch.mean(iou_neg)
    return mask_loss
