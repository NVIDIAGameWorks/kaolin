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
from kaolin.nnsearch import nnsearch
import kaolin.cuda.sided_distance as sd
from scipy.spatial import cKDTree as Tree
import numpy as np


class SidedDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S1, S2):

        batchsize, n, _ = S1.size()
        S1 = S1.contiguous()
        S2 = S2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        dist1 = dist1.cuda()
        idx1 = idx1.cuda()
        try:
            sd.forward(S1, S2, dist1, idx1)
        except BaseException:
            sd.forward_cuda(S1, S2, dist1, idx1)

        return idx1.long()


class SidedDistance(torch.nn.Module):
    r"""For every point in set1 find the indecies of the closest point in set2

    Args:
            set1 (torch.cuda.Tensor) : set of pointclouds of shape B x N x 3
            set2 (torch.cuda.Tensor) : set of pointclouds of shape B x M x 3

    Returns:
            torch.cuda.Tensor: indecies of the closest points in set2

    Example:
            >>> A = torch.rand(2,300,3)
            >>> B = torch.rand(2,200,3)
            >>> sided_minimum_dist = SidedDistance()
            >>> indices = sided_minimum_dist(A,B)
            >>> indices.shape
            torch.Size([2, 300])


    """

    def forward(self, S1: torch.Tensor, S2: torch.Tensor):
        assert len(S1.shape) == 3
        assert len(S2.shape) == 3
        return SidedDistanceFunction.apply(S1, S2).detach()


def chamfer_distance(S1: torch.Tensor, S2: torch.Tensor,
                     w1: float = 1., w2: float = 1.):
    r"""Computes the chamfer distance between two point clouds

    Args:
            S1 (torch.Tensor): point cloud
            S2 (torch.Tensor): point cloud
            w1: (float): weighting of forward direction
            w2: (float): weighting of backward direction

    Returns:
            torch.Tensor: chamfer distance between two point clouds S1 and S2

    Example:
            >>> A = torch.rand(300,3)
            >>> B = torch.rand(200,3)
            >>> >>> chamfer_distance(A,B)
            tensor(0.1868)

    """

    assert (S1.dim() == S2.dim()), 'S1 and S2 must have the same dimesionality'
    assert (S1.dim() == 2), 'the dimensions of the input must be 2 '

    dist_to_S2 = directed_distance(S1, S2)
    dist_to_S1 = directed_distance(S2, S1)

    distance = w1 * dist_to_S2 + w2 * dist_to_S1

    return distance


def directed_distance(S1: torch.Tensor, S2: torch.Tensor, mean: bool = True):
    r"""Computes the average distance from point cloud S1 to point cloud S2

    Args:
            S1 (torch.Tensor): point cloud
            S2 (torch.Tensor): point cloud
            mean (bool): if the distances should be reduced to the average

    Returns:
            torch.Tensor: ditance from point cloud S1 to point cloud S2

    Args:

    Example:
            >>> A = torch.rand(300,3)
            >>> B = torch.rand(200,3)
            >>> >>> directed_distance(A,B)
            tensor(0.1868)

    """

    if S1.is_cuda and S2.is_cuda:
        sided_minimum_dist = SidedDistance()
        closest_index_in_S2 = sided_minimum_dist(
            S1.unsqueeze(0), S2.unsqueeze(0))[0]
        closest_S2 = torch.index_select(S2, 0, closest_index_in_S2)

    else:
        from time import time
        closest_index_in_S2 = nnsearch(S1, S2)
        closest_S2 = S2[closest_index_in_S2]

    dist_to_S2 = (((S1 - closest_S2)**2).sum(dim=-1))
    if mean:
        dist_to_S2 = dist_to_S2.mean()

    return dist_to_S2


def iou(points1: torch.Tensor, points2: torch.Tensor, thresh=.5):
    r""" Computes the intersection over union values for two sets of points

    Args:
            points1 (torch.Tensor): first points
            points2 (torch.Tensor): second points
    Returns:
            iou (torch.Tensor) : IoU scores for the two sets of points

    Examples:
            >>> points1 = torch.rand( 1000)
            >>> points2 = torch.rand( 1000)
            >>> loss = iou(points1, points2)
            tensor(0.3400)

    """
    points1[points1 <= thresh] = 0
    points1[points1 > thresh] = 1

    points2[points2 <= thresh] = 0
    points2[points2 > thresh] = 1

    points1 = points1.view(-1).byte()
    points2 = points2.view(-1).byte()

    assert points1.shape == points2.shape, 'points1 and points2 must have the same shape'

    iou = torch.sum(torch.mul(points1, points2).float()) / \
        torch.sum((points1 + points2).clamp(min=0, max=1).float())

    return iou


def f_score(gt_points: torch.Tensor, pred_points: torch.Tensor,
            radius: float = 0.01, extend=False):
    r""" Computes the f-score of two sets of points, with a hit defined by two point existing withing a defined radius of each other

    Args:
            gt_points (torch.Tensor): ground truth points
            pred_points (torch.Tensor): predicted points points
            radius (float): radisu from a point to define a hit
            extend (bool): if the alternate f-score definition should be applied

    Returns:
            (float): computed f-score

    Example:
            >>> points1 = torch.rand(1000)
            >>> points2 = torch.rand(1000)
            >>> loss = f_score(points1, points2)
            >>> loss
            tensor(0.0070)

    """

    pred_distances = torch.sqrt(directed_distance(
        gt_points, pred_points, mean=False))
    gt_distances = torch.sqrt(directed_distance(
        pred_points, gt_points, mean=False))

    if extend:
        fp = (gt_distances > radius).float().sum()
        tp = (gt_distances <= radius).float().sum()
        precision = tp / (tp + fp)
        tp = (pred_distances <= radius).float().sum()
        fn = (pred_distances > radius).float().sum()
        recall = tp / (tp + fn)

    else:
        fn = torch.sum(pred_distances > radius)
        fp = torch.sum(gt_distances > radius).float()
        tp = torch.sum(gt_distances <= radius).float()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

    f_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f_score
