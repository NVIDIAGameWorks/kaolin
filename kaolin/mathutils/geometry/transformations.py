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

import numpy as np
import torch

# Borrows from kornia
# https://github.com/arraiyopensource/kornia

def rotx(theta, enc='rad'):
    r"""Returns a 3D rotation matrix about the X-axis

    Returns the 3 x 3 rotation matrix :math:`R` that rotates a 3D point by an angle
    theta about the X-axis of the canonical Cartesian axes :math:`\left[ e_1 e_2 e_3 \right]`.

    Note:

    .. math::
        e_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, e_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, e_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}

    Generated matrix:

    .. math::

        R = \begin{bmatrix}
                1 & 0 & 0 \\
                0 & cos(\theta) & -sin(\theta) \\
                0 & sin(\theta) & cos(\theta)
            \end{bmatrix}

    Args:
        theta (Tensor or np.array): degree of rotation (assumes radians
            by default)
        enc (str, choices=['rad', 'deg']): whether the angle is specified in
            degrees ('deg') or radians ('rad'). Default: 'rad'.

    Returns:
        Tensor: one 3 x 3 rotation matrix, for each input entry in theta

    Shape:
        - Input: :math:`(B)` (or) :math:`(B, 1)` (:math:`B` is the batchsize)
        - Output: :math:`(B, 3, 3)`

    Examples:
        >>> # Create a random batch of angles of rotation
        >>> theta = torch.randn(10, 1)
        >>> # Get a 10 x 3 x 3 rotation matrix, one 3 x 3 matrix for each element
        >>> rx = kaolin.mathutils.rotx(theta)
        >>> # Alternatively, use rotations specified in degrees
        >>> theta = 180 * torch.randn(10, 1)
        >>> rx = kaolin.mathutils.rotx(theta, enc='deg')

    """

    if isinstance(theta, np.ndarray) or isinstance(
            theta, float) or isinstance(theta, int):
        theta = torch.from_numpy(theta)

    # Used an f-string here, so can maybe support only Python 3.6+
    if not torch.is_tensor(theta):
        raise TypeError('Expected type torch.Tensor for argument theta. \
            Got {} instead'.format(type(theta)))

    # Check that the passed tensor is 1D (or 1D-like)
    if theta.dim() != 1:
        assert theta.dim() == 2, 'Invalid shape. Exceeds two dimensions.'
        if theta.dim() == 2:
            assert (theta.shape[0] == 1 or theta.shape[
                    1] == 1), 'Must be 1D-like.'
        theta = theta.view(theta.numel())

    # Raise a NotImplementedError if the input encoding is something other
    # than 'rad'
    if enc != 'rad':
        raise NotImplementedError

    # Compute the rotation matrix
    c = torch.cos(theta)
    s = torch.sin(theta)
    rx = torch.zeros(theta.numel(), 3, 3)
    rx[:, 0, 0] = torch.ones(theta.numel())
    rx[:, 1, 1] = rx[:, 2, 2] = c
    rx[:, 1, 2] = -s
    rx[:, 2, 1] = s

    return rx


def roty(theta, enc='rad'):
    r"""Returns a 3D rotation matrix about the Y-axis

    Returns the 3 x 3 rotation matrix that rotates a 3D point by an angle
    theta about the Y-axis of the canonical Cartesian axes [e1 e2 e3].

    Note:

    .. math::
        e_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, e_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, e_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}

    Generated matrix:

    .. math::
        R = \begin{bmatrix}
                cos(\theta) & 0 & sin(\theta) \\
                0 & 1 & 0 \\
                -sin(\theta) & 0 & cos(\theta)
            \end{bmatrix}

    Args:
        theta (Tensor or np.array): degree of rotation (assumes radians
            by default)
        enc (str, choices=['rad', 'deg']): whether the angle is specified in
            degrees ('deg') or radians ('rad'). Default: 'rad'.

    Returns:
        Tensor: one 3 x 3 rotation matrix, for each input entry in theta

    Shape:
        - Input: :math:`(B)` (or) :math:`(B, 1)` (:math:`B` is the batchsize)
        - Output: :math:`(B, 3, 3)`

    Examples:
        >>> # Create a random batch of angles of rotation
        >>> theta = torch.randn(10, 1)
        >>> # Get a 10 x 3 x 3 rotation matrix, one 3 x 3 matrix for each element
        >>> ry = kaolin.mathutils.roty(theta)
        >>> # Alternatively, use rotations specified in degrees
        >>> theta = 180 * torch.randn(10, 1)
        >>> ry = kaolin.mathutils.roty(theta, enc='deg')

    """

    if isinstance(theta, np.ndarray) or isinstance(
            theta, float) or isinstance(theta, int):
        theta = torch.from_numpy(theta)

    # Used an f-string here, so can maybe support only Python 3.6+
    if not torch.is_tensor(theta):
        raise TypeError('Expected type torch.Tensor for argument theta. \
            Got {} instead'.format(type(theta)))

    # Check that the passed tensor is 1D (or 1D-like)
    if theta.dim() != 1:
        assert theta.dim() == 2, 'Invalid shape. Exceeds two dimensions.'
        if theta.dim() == 2:
            assert (theta.shape[0] == 1 or theta.shape[
                    1] == 1), 'Must be 1D-like.'
        theta = theta.view(theta.numel())

    # Raise a NotImplementedError if the input encoding is something other
    # than 'rad'
    if enc != 'rad':
        raise NotImplementedError

    # Compute the rotation matrix
    c = torch.cos(theta)
    s = torch.sin(theta)
    ry = torch.zeros(theta.numel(), 3, 3)
    ry[:, 1, 1] = torch.ones(theta.numel())
    ry[:, 0, 0] = ry[:, 2, 2] = c
    ry[:, 2, 0] = -s
    ry[:, 0, 2] = s

    return ry


def rotz(theta, enc='rad'):
    r"""Returns a 3D rotation matrix about the Z-axis

    Returns the 3 x 3 rotation matrix that rotates a 3D point by an angle
    theta about the Z-axis of the canonical Cartesian axes [e1 e2 e3].

    Note:

    .. math::
        e_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, e_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, e_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}

    Generated matrix:

    .. math::
        R = \begin{bmatrix}
                cos(\theta) & -sin(\theta) & 0  \\
                sin(\theta) & cos(\theta) & 0 \\
                0 & 0 & 1
            \end{bmatrix}

    Args:
        theta (Tensor or np.array): degree of rotation (assumes radians
            by default)
        enc (str, choices=['rad', 'deg']): whether the angle is specified in
            degrees ('deg') or radians ('rad'). Default: 'rad'.

    Returns:
        Tensor: one 3 x 3 rotation matrix, for each input entry in theta

    Shape:
        - Input: :math:`(B)` (or) :math:`(B, 1)` (:math:`B` is the batchsize)
        - Output: :math:`(B, 3, 3)`

    Examples:
        >>> # Create a random batch of angles of rotation
        >>> theta = torch.randn(10, 1)
        >>> # Get a 10 x 3 x 3 rotation matrix, one 3 x 3 matrix for each element
        >>> rz = kaolin.mathutils.rotz(theta)
        >>> # Alternatively, use rotations specified in degrees
        >>> theta = 180 * torch.randn(10, 1)
        >>> rz = kaolin.mathutils.rotz(theta, enc='deg')

    """

    if isinstance(theta, np.ndarray) or isinstance(
            theta, float) or isinstance(theta, int):
        theta = torch.from_numpy(theta)

    # Used an f-string here, so can maybe support only Python 3.6+
    if not torch.is_tensor(theta):
        raise TypeError('Expected type torch.Tensor for argument theta. \
            Got {} instead'.format(type(theta)))

    # Check that the passed tensor is 1D (or 1D-like)
    if theta.dim() != 1:
        assert theta.dim() == 2, 'Invalid shape. Exceeds two dimensions.'
        if theta.dim() == 2:
            assert (theta.shape[0] == 1 or theta.shape[
                    1] == 1), 'Must be 1D-like.'
        theta = theta.view(theta.numel())

    # Raise a NotImplementedError if the input encoding is something other
    # than 'rad'
    if enc != 'rad':
        raise NotImplementedError

    # Compute the rotation matrix
    c = torch.cos(theta)
    s = torch.sin(theta)
    rz = torch.zeros(theta.numel(), 3, 3)
    rz[:, 2, 2] = torch.ones(theta.numel())
    rz[:, 0, 0] = rz[:, 1, 1] = c
    rz[:, 0, 1] = -s
    rz[:, 1, 0] = s

    return rz


# Borrows from Kornia.
# https://github.com/kornia/kornia/blob/master/kornia/geometry/conversions.py
def homogenize_points(pts: torch.Tensor):
    r"""Converts a set of points to homogeneous coordinates.

    Args:
        pts (torch.Tensor): Tensor containing points to be homogenized.

    Returns:
        (torch.Tensor): Homogeneous coordinates for pts.

    Shape:
        pts: :math:`\cdots \times 2` or :math:`\cdots \times 3`

    Example:
        >>> pts = torch.randn(2, 5, 3)
        tensor([[[ 0.0897, -0.1876,  0.1637],
                 [-0.1026, -0.4994,  0.8622],
                 [-1.2909,  0.2678, -1.8021],
                 [-0.2500,  0.3505,  0.9121],
                 [ 0.0580,  1.4497, -0.7224]],

                [[ 0.8102, -0.2467,  0.1951],
                 [ 0.4059, -1.9658,  0.1850],
                 [ 1.5487, -0.8154, -0.5592],
                 [ 0.2269, -0.4137,  0.7187],
                 [-1.1810, -2.3412, -0.4925]]])

        >>> homo_pts = homogenize_points(pts)
        tensor([[[ 0.0897, -0.1876,  0.1637,  1.0000],
                 [-0.1026, -0.4994,  0.8622,  1.0000],
                 [-1.2909,  0.2678, -1.8021,  1.0000],
                 [-0.2500,  0.3505,  0.9121,  1.0000],
                 [ 0.0580,  1.4497, -0.7224,  1.0000]],

                [[ 0.8102, -0.2467,  0.1951,  1.0000],
                 [ 0.4059, -1.9658,  0.1850,  1.0000],
                 [ 1.5487, -0.8154, -0.5592,  1.0000],
                 [ 0.2269, -0.4137,  0.7187,  1.0000],
                 [-1.1810, -2.3412, -0.4925,  1.0000]]])

        >>> homo_pts.shape
        torch.Size([2, 5, 4])

    """

    if not torch.is_tensor(pts):
        raise TypeError('Expected input of type torch.Tensor. '
                        'Got {0} instead.'.format(type(pts)))
    if pts.dim() < 2:
        raise ValueError('Input tensors must have at least 2 dims. '
                         'Got {0} instead.'.format(pts.dim()))

    pad = torch.nn.ConstantPad1d((0, 1), 1.)
    return pad(pts)


# Borrows from Kornia.
# https://github.com/kornia/kornia/blob/master/kornia/geometry/conversions.py
def unhomogenize_points(pts: torch.Tensor):
    r"""Convert a set of points from homogeneous coordinates
    (i.e., projective space) to Euclidean space.

    Usually, for each point :math:`(x, y, z, w)` for the 3D case,
    `unhomogenize_points` returns :math:`\left(\frac{x}{w}, \frac{y}{w},
    \frac{z}{w} \right)`. For the special case where :math:`w` is zero,
    `unhomogenize_points` returns :math:`(x, y, z)`, following OpenCV's
    convention.

    Args:
        pts (torch.Tensor): Tensor containing points to be unhomogenized.

    Shape:
        pts: :math:`\cdots \times 3` or :math:`\cdots \times 4` (usually).

    Returns:
        (torch.Tensor): Unhomogenized coordinates for `pts`.

    Examples:
        >>> homo_pts = torch.randn(2, 5, 4)
        tensor([[[ 0.0897, -0.1876,  0.1637,  1.0000],
                 [-0.1026, -0.4994,  0.8622,  1.0000],
                 [-1.2909,  0.2678, -1.8021,  1.0000],
                 [-0.2500,  0.3505,  0.9121,  1.0000],
                 [ 0.0580,  1.4497, -0.7224,  1.0000]],

                [[ 0.8102, -0.2467,  0.1951,  1.0000],
                 [ 0.4059, -1.9658,  0.1850,  1.0000],
                 [ 1.5487, -0.8154, -0.5592,  1.0000],
                 [ 0.2269, -0.4137,  0.7187,  1.0000],
                 [-1.1810, -2.3412, -0.4925,  1.0000]]])

        >>> unhomo_pts = kal.math.unhomogenize_points(homo_pts)
        tensor([[[ 0.0897, -0.1876,  0.1637],
                 [-0.1026, -0.4994,  0.8622],
                 [-1.2909,  0.2678, -1.8021],
                 [-0.2500,  0.3505,  0.9121],
                 [ 0.0580,  1.4497, -0.7224]],

                [[ 0.8102, -0.2467,  0.1951],
                 [ 0.4059, -1.9658,  0.1850],
                 [ 1.5487, -0.8154, -0.5592],
                 [ 0.2269, -0.4137,  0.7187],
                 [-1.1810, -2.3412, -0.4925]]])

        >>> unhomo_pts = kal.math.unhomogenize_points(unhomo_pts)
        tensor([[[  0.5482,  -1.1463],
                 [ -0.1190,  -0.5792],
                 [  0.7163,  -0.1486],
                 [ -0.2741,   0.3843],
                 [ -0.0803,  -2.0066]],

                [[  4.1518,  -1.2645],
                 [  2.1938, -10.6255],
                 [ -2.7696,   1.4582],
                 [  0.3157,  -0.5756],
                 [  2.3977,   4.7533]]])

    """

    if not torch.is_tensor(pts):
        raise TypeError('Expected input of type torch.Tensor. '
                        'Got {0} instead.'.format(type(pts)))
    if pts.dim() < 2:
        raise ValueError('Input tensors must have at least 2 dims. '
                         'Got {0} instead.'.format(pts.dim()))

    # Get points with the last coordinate (scale) as 0 (points at inf).
    w = pts[..., -1:]
    # Determine the scale factor each point needs to be multiplied by.
    # For points at infinity, use a scale factor 1.
    eps = 1e-6
    scale = torch.where(torch.abs(w) > eps, 1. / w, torch.ones_like(w))

    return scale * pts[..., :-1]


# Borrows from Kornia.
# https://github.com/kornia/kornia/blob/master/kornia/geometry/linalg.py
def transform3d(pts: torch.Tensor, tform: torch.Tensor) -> torch.Tensor:
    r"""Transform a set of points `pts` using a general 3D transform
    `tform`.

    Args:
        pts (torch.Tensor): Points to be transformed (shape: :math:`\cdots
            \times 4`)
        tform (torch.Tensor): A 3D projective transformation matrix.
            (shape: :math:`4 \times 4`)

    Returns
        (torch.Tensor): Transformed points.

    """

    if not torch.is_tensor(pts):
        raise TypeError('Expected input pts to be of type torch.Tensor. '
                        'Got {0} instead.'.format(type(pts)))
    if not torch.is_tensor(tform):
        raise TypeError('Expected input tform to be of type torch.Tensor. '
                        'Got {0} instead.'.format(type(tform)))

    if pts.dim() < 2:
        raise ValueError('Input pts must have at least 2 dimensions. '
                         'Got only {0}.'.format(pts.dim()))
    if pts.shape[-1] != 4:
        raise ValueError('Input pts must have shape 4 in its last dimension. '
                         'Got {0} instead.'.format(pts.shape[-1]))
    if tform.dim() < 2 or tform.shape[-1] != 4 or tform.shape[-2] != 4:
        raise ValueError('Input tform must have at least 2 dimensions '
                         'and the last two dims must be of shape 4. Got {0} dimensions and '
                         'shape {1}.'.format(tform.dim(), tform.shape))

    # Unsqueezing at dim -3 (to handle arbitrary batchsizes)
    # For a 2D tensor, unsqueeze(-3) is equivalent to unsqueeze(0)
    # tform is ordered as (B, 4, 4)
    # pts is ordered as (B, N, 3), where B: batchsize, N: num points
    pts_tformed_homo = torch.matmul(tform.unsqueeze(-3), pts.unsqueeze(-1))
    pts_tformed = unhomogenize_points(pts_tformed_homo.squeeze(-1))

    return pts_tformed[..., :3]


# Borrows from Kornia.
# https://github.com/kornia/kornia/blob/master/kornia/geometry/linalg.py
def invert_rigid_transform_3d(tform: torch.Tensor):
    r"""Invert a 3D rigid body (SE(3)) transform.

    Args:
        tform (torch.Tensor): SE(3) transformation matrix (shape:
            :math:`\cdots \times 4 \times 4`)

    Returns:
        inv_tform (torch.Tensor): Inverse transformation matrix (shape:
            :math:`\cdots \times 4 \times 4`)
    """

    if not torch.is_tensor(tform):
        raise TypeError('Expected input tform to be of type torch.Tensor. '
                        'Got {0} instead.'.format(type(tform)))
    if tform.shape[-2, :] != (4, 4):
        raise ValueError('Input tform must be of shape (..., 4, 4). '
                         'Got {0} instead.'.format(tform.shape))

    # Unpack translation and rotation components
    rot = tform[..., :3, :3]
    trans = tform[..., :3, :3]

    # Compute the inverse
    inv_rot = torch.transpose(rot, -1, -2)
    inv_trans = torch.matmul(-inv_rot, trans)

    # Pack the inverse rotation and translation components
    inv_trans = torch.zeros_like(tform)
    inv_trans[..., :3, :3] = inv_rot
    inv_trans[..., :3, 3] = inv_trans
    inv_trans[..., -1, -1] = 1.

    return inv_trans


# Borrows from Kornia.
# https://github.com/kornia/kornia/blob/master/kornia/geometry/linalg.py
def compose_transforms_3d(tforms):
    r"""Compose (concatenate) a series of 3D transformation matrices.

    Args:
        tforms (tuple, list): Iterable containing the transforms
            to be composed (Each transform must be of type torch.Tensor)
            (shape: :math:`\cdots \times 4 \times 4`).

    Returns:
        cat (torch.Tensor): Concatenated transform. (shape:
            :math:`\cdots \times 4 \times 4`)

    """

    if len(tforms) == 1:
        raise ValueError('Expected at least 2 transforms to compose. '
                         'Got only 1.')
    cat = None
    for idx, tform in enumerate(tforms):
        if not torch.is_tensor(tform):
            raise TypeError('Expected elements of tforms to be of type '
                            'torch.Tensor. Got {0} at index {1}'.format(
                                type(tform), idx))
        if tform.shape[-2, :] != (4, 4):
            raise TypeError('Expected elements of tforms to be of shape '
                            '(..., 4, 4). Got {0} at index {1}'.format(
                                tform.shape, idx))
        if idx == 0:
            cat = tform
        else:
            cat = torch.matmul(cat, tform)

    return cat


def compute_camera_params(azimuth: float, elevation: float, distance: float):

    theta = np.deg2rad(azimuth)
    phi = np.deg2rad(elevation)

    camY = distance * np.sin(phi)
    temp = distance * np.cos(phi)
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    axisZ = cam_pos.copy()
    axisY = np.array([0, 1, 0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([axisX, axisY, axisZ])
    l2 = np.atleast_1d(np.linalg.norm(cam_mat, 2, 1))
    l2[l2 == 0] = 1
    cam_mat = cam_mat / np.expand_dims(l2, 1)

    return torch.FloatTensor(cam_mat), torch.FloatTensor(cam_pos)
