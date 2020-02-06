# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import numpy as np
import torch

from kaolin.rep import PointCloud
from kaolin import helpers


# Tiny eps
EPS = 1e-6


def scale(cloud: Union[torch.Tensor, PointCloud],
          scf: Union[float, int, torch.Tensor],
          inplace: Optional[bool] = True):
    """Scales the input pointcloud by a scaling factor.

    Args:
        cloud (torch.Tensor or kaolin.rep.PointCloud): pointcloud (ndims >= 2).
        scf (float, int, or torch.Tensor): scaling factor (scalar, or tensor).
            All elements of scf must be positive.
        inplace (bool, optional): Bool to make the transform in-place.

    Returns:
        (torch.Tensor): scaled pointcloud of the same shape as input.

    Shape:
        - cloud: :math:`(B x N x D)` (or) :math:`(N x D)`, where :math:`(B)`
            is the batchsize, :math:`(N)` is the number of points per cloud,
            and :math:`(D)` is the dimensionality of each cloud.
        - scf: :math:`(1)` or :math:`(B)`.

    Example:
        >>> points = torch.rand(1000,3)
        >>> points2 = scale(points, torch.FloatTensor([3]))

    """

    if isinstance(cloud, np.ndarray):
        cloud = torch.from_numpy(cloud)

    if isinstance(scf, np.ndarray):
        scf = torch.from_numpy(scf)

    if isinstance(cloud, PointCloud):
        cloud = cloud.points

    if isinstance(scf, int) or isinstance(scf, float):
        scf = torch.Tensor([scf]).to(cloud.device)

    helpers._assert_tensor(cloud)
    helpers._assert_tensor(scf)
    helpers._assert_dim_ge(cloud, 2)
    helpers._assert_gt(scf, 0.)

    if not inplace:
        cloud = cloud.clone()

    return scf * cloud


def rotate(cloud: Union[torch.Tensor, PointCloud], rotmat: torch.Tensor,
           inplace: Optional[bool] = True):
    """Rotates the the input pointcloud by a rotation matrix.

    Args:
        cloud (Tensor or np.array): pointcloud (ndims = 2 or 3)
        rotmat (Tensor or np.array): rotation matrix (3 x 3, 1 per cloud).
        inplace (bool, optional): Bool to make the transform in-place.

    Returns:
        cloud_rot (Tensor): rotated pointcloud of the same shape as input

    Shape:
        - cloud: :math:`(B x N x 3)` (or) :math:`(N x 3)`, where :math:`(B)`
            is the batchsize, :math:`(N)` is the number of points per cloud,
            and :math:`(3)` is the dimensionality of each cloud.
        - rotmat: :math:`(3, 3)` or :math:`(B, 3, 3)`.

    Example:
        >>> points = torch.rand(1000,3)
        >>> r_mat = torch.rand(3,3)
        >>> points2 = rotate(points, r_mat)

    """
    if isinstance(cloud, np.ndarray):
        cloud = torch.from_numpy(cloud)
    if isinstance(cloud, PointCloud):
        cloud = cloud.points
    if isinstance(rotmat, np.ndarray):
        rotmat = torch.from_numpy(rotmat)

    helpers._assert_tensor(cloud)
    helpers._assert_tensor(rotmat)
    helpers._assert_dim_ge(cloud, 2)
    helpers._assert_dim_ge(rotmat, 2)
    # Rotation matrix must have last two dimensions of shape 3.
    helpers._assert_shape_eq(rotmat, (3, 3), dim=-1)
    helpers._assert_shape_eq(rotmat, (3, 3), dim=-2)

    if not inplace:
        cloud = cloud.clone()

    if rotmat.dim() == 2 and cloud.dim() == 2:
        cloud = torch.mm(rotmat, cloud.transpose(0, 1)).transpose(0, 1)
    else:
        if rotmat.dim() == 2:
            rotmat = rotmat.expand(cloud.shape[0], 3, 3)
        cloud = torch.bmm(rotmat, cloud.transpose(1, 2)).transpose(1, 2)

    return cloud


def realign(src: Union[torch.Tensor, PointCloud],
            tgt: Union[torch.Tensor, PointCloud],
            inplace: Optional[bool] = True):
    r""" Aligns a pointcloud `src` to be in the same (axis-aligned) bounding
    box as that of pointcloud `tgt`.

    Args:
        src (torch.Tensor or PointCloud) : Source pointcloud to be transformed
            (shape: :math:`\cdots \times N \times D`, where :math:`N` is the
            number of points in the pointcloud, and :math:`D` is the
            dimensionality of each point in the cloud).
        tgt (torch.Tensor or PointCloud) : Target pointcloud to which `src`is
            to be transformed (The `src` cloud is transformed to the
            axis-aligned bounding box that the target cloud maps to). This
            cloud must have the same number of dimensions :math:`D` as in the
            source cloud. (shape: :math:`\cdots \times \cdots \times D`).
        inplace (bool, optional): Bool to make the transform in-place.

    Returns:
        (torch.Tensor): Pointcloud `src` realigned to fit in the (axis-aligned)
            bounding box of the `tgt` cloud.

    Example:
        >>> tgt = torch.rand(1000)
        >>> src = (tgt * 100) + 3
        >>> src_realigned = realign(src, tgt)

    """
    if isinstance(src, PointCloud):
        src = src.points
    if isinstance(tgt, PointCloud):
        tgt = tgt.points
    helpers._assert_tensor(src)
    helpers._assert_tensor(tgt)
    helpers._assert_dim_ge(src, 2)
    helpers._assert_dim_ge(tgt, 2)
    helpers._assert_shape_eq(src, tgt.shape, dim=-1)

    if not inplace:
        src = src.clone()

    # Compute the relative scaling factor and scale the src cloud.
    src_min, _ = src.min(-2, keepdim=True)
    src_max, _ = src.max(-2, keepdim=True)
    tgt_min, _ = tgt.min(-2, keepdim=True)
    tgt_max, _ = tgt.max(-2, keepdim=True)

    src = ((src - src_min) / (src_max - src_min + EPS)) * (tgt_max - tgt_min) + tgt_min
    return src


def normalize(cloud: Union[torch.Tensor, PointCloud],
              inplace: Optional[bool] = True):
    r"""Returns a normalized pointcloud with zero-mean and unit standard
    deviation. For batched clouds, each cloud is independently normalized.

    Args:
        cloud (torch.Tensor or PointCloud): Input pointcloud to be normalized
            (shape: :math:`B \times \cdots \times N \times D`, where :math:`B`
            is the batchsize (optional), :math:`N` is the number of points in
            the cloud, and :math:`D` is the dimensionality of the cloud.
        inplace (bool, optional): Bool to make the transform in-place.

    Returns:
        (torch.Tensor or PointCloud): The normalized pointcloud.

    """

    if isinstance(cloud, np.ndarray):
        cloud = torch.from_numpy(cloud)

    helpers._assert_tensor(cloud)
    helpers._assert_dim_ge(cloud, 2)
    if not inplace:
        cloud = cloud.clone()

    cloud = (cloud - cloud.mean(-2).unsqueeze(-2))\
            / (cloud.std(-2).unsqueeze(-2) + EPS)

    return cloud



if __name__ == '__main__':

    device = 'cpu'
    
    # Test: realign
    src = torch.randn(2, 3, 4, 3).to(device)
    tgt = torch.rand(2, 3, 4, 3).to(device)
    src_ = realign(src, tgt)
    # After transformation, src_.mean(-2) should be equal to tgt.mean(-2)
    print(src_.mean(-2))
    print(tgt.mean(-2))

    # Test: normalize
    a = torch.rand(2, 10, 3)
    a_ = normalize(a)
    print(a_.mean(-2))
    print(a_.std(-2))
