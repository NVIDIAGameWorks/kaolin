# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Kornia components Copyright (c) 2019 Kornia project authors
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

"""
Projective geometry utility functions
"""

import torch

from kaolin.mathutils import *

# Borrows from Kornia.
# https://github.com/kornia/kornia/blob/master/kornia/geometry/camera/perspective.py
def project_points(pts: torch.Tensor, intrinsics: torch.Tensor,
                   extrinsics: torch.Tensor = None):
    r"""Projects a set of 3D points onto a 2D image, given the
    camera parameters (intrinsics and extrinsics).

    Args:
        pts (torch.Tensor): 3D points to be projected onto the image
            (shape: :math:`\cdots \times N \times 3` or :math:`\cdots
            \times N \times 4`).
        intrinsics (torch.Tensor): camera intrinsic matrix/matrices
            (shape: :math:`B \times 4 \times 4` or :math:`4 \times 4`).
        extrinsics (torch.Tensor): camera extrinsic matrix/matrices
            (shape: :math:`B \times 4 \times 4` or :math:`4 \times 4`).

    Note:
        If pts is not of dim 2, then it is treated as a minibatch. For
        each point set in the minibatch (of size :math:`B`), one can
        apply the same pair of intrinsics or extrinsics (size :math:`4
        \times 4`), or choose a different intrinsic-extrinsic pair. In
        the latter case, the passed intrinsics/extrinsics must be of
        shape (:math:`B \times 4 \times 4`).

    Returns:
        (torch.Tensor): pixel coordinates of the input 3D points.

    Examples:
        >>> pts = torch.rand(5, 3)
        tensor([[0.6411, 0.4996, 0.7689],
                [0.2288, 0.9391, 0.2062],
                [0.4991, 0.4673, 0.6192],
                [0.0397, 0.3477, 0.4895],
                [0.9219, 0.4121, 0.8046]])
        >>> intrinsics = torch.FloatTensor([[720, 0, 120, 0],
                                            [0, 720, 90, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])
        >>> img_pts = kal.vision.project_points(pts, intrinsics)
        tensor([[ 720.3091,  557.8361],
                [ 919.0596, 3369.7185],
                [ 700.4019,  633.3636],
                [ 178.3637,  601.4078],
                [ 945.0151,  458.7479]])
        >>> img_pts.shape
        torch.Size([5, 2])

        >>> # `project_points()` also takes in batched inputs
        >>> pts = torch.rand(10, 5, 3)
        >>> # Applies the same intrinsics to all samples in the batch
        >>> img_pts = kal.vision.project_points(pts, intrinsics)
        torch.Size([10, 5, 2])

        >>> # Optionally, can use a per-sample intrinsic, for each
        >>> # example in the minibatch.
        >>> intrinsics_a = intrinsics.repeat(5, 1, 1)
        >>> intrinsics_b = torch.eye(4).repeat(5, 1, 1)
        >>> # Use `intrinsics_a` for the first 5 samples and
        >>> # `intrinsics_b` for the last 5
        >>> intrinsics = torch.cat((intrinsics_a, intrinsics_b), dim=0)
        >>> img_pts = kal.vision.project_points(pts, intrinsics)
        >>> img_pts.shape
        torch.Size([10, 5, 2])

        >>> # Can also use a per sample extrinsics matrix
        >>> pts = torch.rand(10, 5, 3)
        >>> extrinsics = torch.eye(4).repeat(10, 1, 1)
        >>> img_pts = kal.vision.project_points(pts, intrinsics, extrinsics)

    """
    if not torch.is_tensor(pts):
        raise TypeError('Expected input pts to be of type torch.Tensor. '
                        'Got {0} instead.'.format(type(pts)))
    if not torch.is_tensor(intrinsics):
        raise TypeError('Expected input intrinsics to be of type '
                        'torch.Tensor. Got {0} instead'.format(type(intrinsics)))
    if extrinsics is not None:
        if not torch.is_tensor(extrinsics):
            raise TypeError('Expected input extrinsics to be of type '
                            'torch.Tensor. Got{0} instead.'.format(type(extrinsics)))

    if pts.dim() < 2:
        raise ValueError('Expected input pts to have at least 2 dims. '
                         'Got only {0}'.format(pts.dim()))
    if pts.shape[-1] not in [3, 4]:
        raise ValueError('Last dim of input pts must be of shape '
                         '3 or 4. Got {0} instead.'.format(pts.shape[-1]))

    # Infer the batchsize of pts (assume it is 1, to begin with)
    batchsize = 1

    if pts.dim() > 2:
        batchsize = pts.shape[0]
    # if pts.dim() == 2:
    #     pts = pts.unsqueeze(0)

    # If extrinsics is None, set to identity
    if extrinsics is None:
        extrinsics = torch.eye(4).to(pts.device)

    if intrinsics.shape[-2:] != (4, 4):
        raise ValueError('Expected intrinsics to be of shape (4, 4). '
                         'Got {0} instead.'.format(intrinsics.shape))
    if extrinsics.shape[-2:] != (4, 4):
        raise ValueError('Expected extrinsics to be of shape (4, 4). '
                         'Got {0} instead.'.format(extrinsics.shape))

    if intrinsics.dim() > 2:
        if intrinsics.shape[0] != batchsize and intrinsics.shape[0] != 1:
            raise ValueError('Dimension 0 of intrinsics must be either '
                             'equal to 1, or equal to the batch size of input pts. '
                             'Got {0} instead.'.format(intrinsics.shape[0]))
    if extrinsics.dim() > 2:
        if extrinsics.shape[0] != batchsize and extrinsics.shape[0] != 1:
            raise ValueError('Dimension 0 of extrinsics must be either '
                             'equal to 1, or equal to the batch size of input pts. '
                             'Got {0} instead.'.format(extrinsics.shape[0]))
        if intrinsics.shape[0] != extrinsics.shape[0]:
            raise ValueError('Inputs intrinsics and extrinsics must '
                             'have same shape at dim 0. Got {0} and {1}.'.format(
                                 intrinsics.shape[0], extrinsics.shape[0]))

    # Determine whether or not to homogenize pts
    if pts.shape[-1] == 3:
        pts = homogenize_points(pts)

    # Perform projection
    pts = transform3d(pts, torch.matmul(intrinsics, extrinsics))
    x = pts[..., 0]
    y = pts[..., 1]
    z = pts[..., 2]
    u = x / torch.where(z == 0, torch.ones_like(z), z)
    v = y / torch.where(z == 0, torch.ones_like(z), z)

    return torch.stack([u, v], dim=-1)


# Borrows from Kornia.
# https://github.com/kornia/kornia/blob/master/kornia/geometry/camera/perspective.py
def unproject_points(pts: torch.Tensor, depth: torch.Tensor,
                     intrinsics: torch.Tensor):
    r"""Unprojects (back-projects) a set of points from a 2D image
    to 3D camera coordinates, given depths and the intrinsics.

    Args:
        pts (torch.Tensor): 2D points to be 'un'projected to 3D.
            (shape: :math:`\cdots \times N \times 2` or
            :math:`\cdots \times 3`).
        depth (torch.Tensor): Depth for each point in pts (shape:
            :math:`\cdots \times N \times 1`).
        intrinsics (torch.Tensor): Camera intrinsics (shape: :math:`
            \cdots \times 4 \times 4`).

    Returns:
        (torch.Tensor): Camera coordinates of the input points.
            (shape: :math:`\cdots \times 3`)

    Examples:
        >>> img_pts = torch.rand(5, 2)
        tensor([[0.6591, 0.8643],
                [0.4913, 0.8048],
                [0.2129, 0.2338],
                [0.9604, 0.2347],
                [0.5779, 0.9745]])
        >>> depths = torch.rand(5) + 1.
        tensor([1.4135, 1.0138, 1.6001, 1.6868, 1.0867])
        >>> intrinsics = torch.FloatTensor([[720, 0, 120, 0],
                                            [0, 720, 90, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])
        >>> cam_pts = kal.vision.unproject_points(img_pts, depths, intrinsics)
        tensor([[-0.2343, -0.1750,  1.4135],
                [-0.1683, -0.1256,  1.0138],
                [-0.2662, -0.1995,  1.6001],
                [-0.2789, -0.2103,  1.6868],
                [-0.1802, -0.1344,  1.0867]])
        >>> cam_pts.shape
        torch.Size([5, 3])

        >>> # Also works for batched inputs
        >>> img_pts = torch.rand(10, 5, 2)
        >>> depths = torch.rand(10, 5) + 1.
        >>> cam_pts = kal.vision.unproject_points(img_pts, depths, intrinsics)
        >>> cam_pts.shape
        torch.Size([10, 5, 3])

        >>> # Just like for `project_points()`, can use a per-sample intrinsics
        >>> # matrix.
        >>> intrinsics_a = intrinsics.repeat(5, 1, 1)
        >>> intrinsics_b = torch.eye(4).repeat(5, 1, 1)
        >>> # Use `intrinsics_a` for the first 5 samples and
        >>> # `intrinsics_b` for the last 5
        >>> intrinsics = torch.cat((intrinsics_a, intrinsics_b), dim=0)
        >>> cam_pts = kal.vision.project_points(img_pts, depths, intrinsics)
        >>> cam_pts.shape
        torch.Size([10, 5, 3])

    """
    if not torch.is_tensor(pts):
        raise TypeError('Expected input pts to be of type torch.Tensor. '
                        'Got {0} instead.'.format(type(pts)))
    if not torch.is_tensor(depth):
        raise TypeError('Expected input depth to be of type torch.Tensor. '
                        'Got {0} instead.'.format(type(depth)))
    if not torch.is_tensor(intrinsics):
        raise TypeError('Expected input intrinsics to be of type '
                        'torch.Tensor. Got {0} instead'.format(
                            type(intrinsics)))

    if pts.dim() < 2:
        raise ValueError('Expected input pts to have at least 2 dims. '
                         'Got only {0}'.format(pts.dim()))
    if pts.shape[-1] not in [2, 3]:
        raise ValueError('Last dim of input pts must be of shape '
                         '2 or 3. Got {0} instead.'.format(pts.shape[-1]))

    if depth.shape[-1] != 1:
        if depth.dim() == pts.dim() - 1:
            # If the dim of depth differs from the dim of pts by just 1,
            # try appending an additional dimension to make it work.
            # Else, raise a ValueError.
            depth = depth.unsqueeze(-1)
        else:
            raise ValueError('Input depth must have shape 1 in the last '
                             'dimension. Got {0} instead.'.format(depth.shape[-1]))
    if depth.shape[:-1] != pts.shape[:-1]:
        raise ValueError('Inputs pts and depth must have matching shapes '
                         'except at the last dimension. Got {0} and {1} respectively.'
                         ''.format(pts.shape, depth.shape))

    # Homogenize pts if needed
    if pts.shape[-1] == 2:
        # If pts is 2D, homogenize twice (as we need to
        # apply a 4 x 4 intrinsics inverse matrix)
        pts = homogenize_points(pts)
        pts = homogenize_points(pts)
    elif pts.shape[-1] == 3:
        pts = homogenize_points(pts)

    return transform3d(pts, intrinsics.inverse()) * depth
