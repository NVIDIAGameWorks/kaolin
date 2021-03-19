# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
from kaolin import _C


class _SidedDistanceFunction(torch.autograd.Function):
    """torch.autograd.Function for sided_distance.

        Refer to :func:`sided_distance`.
    """

    @staticmethod
    def forward(ctx, p1, p2):
        p1_dtype = p1.dtype
        p1_device = p1.device

        p1 = p1.contiguous()
        p2 = p2.contiguous()

        batchsize = p1.shape[0]
        n = p1.shape[1]

        dist = torch.zeros(batchsize, n, device=p1_device, dtype=p1_dtype, requires_grad=True)
        idx = torch.zeros(batchsize, n, device=p1_device, dtype=torch.long)

        _C.metrics.sided_distance_forward_cuda(p1, p2, dist, idx)

        ctx.save_for_backward(p1, p2, idx)
        ctx.mark_non_differentiable(idx)

        return dist, idx

    @staticmethod
    def backward(ctx, grad_output_dist, grad_output_idx):

        grad_output_dist = grad_output_dist.contiguous()

        p1, p2, idx = ctx.saved_tensors

        grad_p1 = torch.zeros_like(p1)
        grad_p2 = torch.zeros_like(p2)

        _C.metrics.sided_distance_backward_cuda(grad_output_dist, p1, p2, idx, grad_p1, grad_p2)

        return grad_p1, grad_p2


def sided_distance(p1, p2):
    r"""For every point in p1 find the indices and euclidean 
    distances of the closest point in p2.

    Args:
        p1 (torch.Tensor) : pointclouds of shape (B, N, 3)
        p2 (torch.Tensor) : pointclouds of shape (B, M, 3)

    Returns:
        (torch.Tensor, torch.Tensor): the indices and distances from points in p1 to the
        corresponding closest points in p2, both have shape of
        (B, N)

    Example:
        >>> p1 = torch.tensor([[[5.9336, 4.9742, 8.1047]],
        ...                    [[4.1939, 3.3612, 9.5407]]], device='cuda', dtype=torch.float)
        >>> p2 = torch.tensor([[[1.6998, 0.7719, 2.9987],
        ...                     [0.1812, 8.9342, 10.0285]],
        ...                    [[10.0184, 0.3928, 5.2545],
        ...                     [4.2934, 11.2127, 4.5247]]], device='cuda', dtype=torch.float)
        >>> distance, idx = sided_distance(p1, p2)
        >>> distance
        tensor([[52.4727],
                [61.1077]], device='cuda:0')
        >>> idx
        tensor([[1],
                [0]], device='cuda:0')
    """
    dist, idx = _SidedDistanceFunction.apply(p1, p2)

    return dist, idx

def chamfer_distance(p1, p2, w1=1., w2=1.):
    r"""Computes the chamfer distance between two pointclouds

    Args:
        p1 (torch.Tensor): pointclouds of shape (B, N, 3)
        p2 (torch.Tensor): pointclouds of shape (B, M, 3)
        w1 (float): weighting of forward direction. Default: 1.
        w2 (float): weighting of backward direction. Default: 1.

    Returns:
        (torch.Tensor): chamfer distance between two pointclouds p1 and p2 of shape (B)
    Example:
        >>> p1 = torch.tensor([[[8.8977, 4.1709, 1.2839],
        ...                     [8.5640, 7.7767, 9.4214]],
        ...                    [[0.5431, 6.4495, 11.4914],
        ...                     [3.2126, 8.0865, 3.1018]]], device='cuda', dtype=torch.float)
        >>> p2 = torch.tensor([[[6.9340, 6.1152, 3.4435],
        ...                     [0.1032, 9.8181, 11.3350]],
        ...                    [[11.4006, 2.2154, 7.9589],
        ...                     [4.2586, 1.4133, 7.2606]]], device='cuda', dtype=torch.float)
        >>> chamfer_distance(p1, p2)
        tensor([ 72.5838, 151.0809], device='cuda:0')
    """
    dist_to_p2 = sided_distance(p1, p2)[0].mean(dim=-1)
    dist_to_p1 = sided_distance(p2, p1)[0].mean(dim=-1)

    if (w1 == 1 and w2 == 1):
        distance = dist_to_p2 + dist_to_p1
    else:
        distance = w1 * dist_to_p2 + w2 * dist_to_p1

    return distance

def f_score(gt_points, pred_points, radius=0.01, eps=1e-8):
    r"""Computes the f-score of two sets of points, with a hit defined by two point existing within a defined radius of each other

    Args:
        gt_points (torch.Tensor): ground truth pointclouds of shape (B, N, 3)
        pred_points (torch.Tensor): predicted points pointclouds of shape (B, M, 3)
        radius (float): radius from a point to define a hit
                        Default: 0.01
        eps (float): epsilon used to calculate f score.

    Returns:
        (torch.Tensor): computed f-score tensor of shape (B), which has the same dtype as input pred_points.

    Example:
        >>> p1 = torch.tensor([[[8.8977, 4.1709, 1.2839],
        ...                     [8.5640, 7.7767, 9.4214]],
        ...                    [[0.5431, 6.4495, 11.4914],
        ...                     [3.2126, 8.0865, 3.1018]]], device='cuda', dtype=torch.float)
        >>> p2 = torch.tensor([[[9.4863, 4.2249, 0.1712],
        ...                     [8.1783, 8.5310, 8.5119]],
        ...                    [[-0.0020699, 6.4429, 12.3],
        ...                     [3.8386, 8.3585, 4.7662]]], device='cuda', dtype=torch.float)
        >>> f_score(p1, p2, radius=1)
        tensor([0.5000, 0.0000], device='cuda:0')
        >>> f_score(p1, p2, radius=1.5)
        tensor([1.0000, 0.5000], device='cuda:0')
    """
    pred_distances = torch.sqrt(sided_distance(gt_points, pred_points)[0])
    gt_distances = torch.sqrt(sided_distance(pred_points, gt_points)[0])

    data_type = gt_points.dtype

    fn = torch.sum(pred_distances > radius, dim=0).type(data_type)
    fp = torch.sum(gt_distances > radius, dim=0).type(data_type)
    tp = (gt_distances.shape[1] - fp).type(data_type)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f_score = 2 * (precision * recall) / (precision + recall + eps)
    return f_score

def _sided_distance(p1, p2):
    """
    Pytorch version of sided distances for testing.
    """

    batch_size = p1.shape[0]

    dists = (p1.reshape(batch_size, -1, 1, 3) - p2.reshape(batch_size, 1, -1, 3)) ** 2
    dists = torch.sum(dists, dim=-1)

    dist = torch.min(dists, dim=-1)
    return dist.values
