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

from typing import Iterable, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from kaolin.rep import Mesh, TriangleMesh, QuadMesh
from kaolin import helpers


# Tiny eps
EPS = 1e-6


def sample_triangle_mesh(vertices: torch.Tensor, faces: torch.Tensor,
                         num_samples: int, eps: float = 1e-10):
    r""" Uniformly samples the surface of a mesh.

    Args:
        vertices (torch.Tensor): Vertices of the mesh (shape:
            :math:`N \times 3`, where :math:`N` is the number of vertices)
        faces (torch.LongTensor): Faces of the mesh (shape: :math:`F \times 3`,
            where :math:`F` is the number of faces).
        num_samples (int): Number of points to sample
        eps (float): A small number to prevent division by zero
                     for small surface areas.

    Returns:
        (torch.Tensor): Uniformly sampled points from the triangle mesh.

    Example:
        >>> points = sample_triangle_mesh(vertices, faces, 10)
        >>> points
        tensor([[ 0.0293,  0.2179,  0.2168],
                [ 0.2003, -0.3367,  0.2187],
                [ 0.2152, -0.0943,  0.1907],
                [-0.1852,  0.1686, -0.0522],
                [-0.2167,  0.3171,  0.0737],
                [ 0.2219, -0.0289,  0.1531],
                [ 0.2217, -0.0115,  0.1247],
                [-0.1400,  0.0364, -0.1618],
                [ 0.0658, -0.0310, -0.2198],
                [ 0.1926, -0.1867, -0.2153]])
    """

    helpers._assert_tensor(vertices)
    helpers._assert_tensor(faces)
    helpers._assert_dim_ge(vertices, 2)
    helpers._assert_dim_ge(faces, 2)
    # We want the last dimension of vertices to be of shape 3.
    helpers._assert_shape_eq(vertices, (-1, 3), dim=-1)

    dist_uni = torch.distributions.Uniform(torch.tensor([0.]).to(
        vertices.device), torch.tensor([1.]).to(vertices.device))

    # calculate area of each face
    x1, x2, x3 = torch.split(torch.index_select(
        vertices, 0, faces[:, 0]) - torch.index_select(
        vertices, 0, faces[:, 1]), 1, dim=1)
    y1, y2, y3 = torch.split(torch.index_select(
        vertices, 0, faces[:, 1]) - torch.index_select(
        vertices, 0, faces[:, 2]), 1, dim=1)
    a = (x2 * y3 - x3 * y2)**2
    b = (x3 * y1 - x1 * y3)**2
    c = (x1 * y2 - x2 * y1)**2
    Areas = torch.sqrt(a + b + c) / 2
    # percentage of each face w.r.t. full surface area
    Areas = Areas / (torch.sum(Areas) + eps)

    # define descrete distribution w.r.t. face area ratios caluclated
    cat_dist = torch.distributions.Categorical(Areas.view(-1))
    face_choices = cat_dist.sample([num_samples])

    # from each face sample a point
    select_faces = faces[face_choices]
    xs = torch.index_select(vertices, 0, select_faces[:, 0])
    ys = torch.index_select(vertices, 0, select_faces[:, 1])
    zs = torch.index_select(vertices, 0, select_faces[:, 2])
    u = torch.sqrt(dist_uni.sample([num_samples]))
    v = dist_uni.sample([num_samples])
    points = (1 - u) * xs + (u * (1 - v)) * ys + u * v * zs

    return points


def normalize(mesh: Type[Mesh], inplace: Optional[bool] = True):
    r"""Normalize a mesh such that it is centered at the origin and has
    unit standard deviation.

    Args:
        mesh (Mesh): Mesh to be normalized.
        inplace (bool, optional): Bool to make this operation in-place.

    Returns:
        (Mesh): Normalized mesh.

    """
    if not isinstance(mesh, Mesh):
        raise TypeError('Input mesh must be of type Mesh. '
                        'Got {0} instead.'.format(type(mesh)))
    if not inplace:
        mesh = mesh.clone()

    mesh.vertices = (mesh.vertices - mesh.vertices.mean(-2).unsqueeze(-2))\
        / (mesh.vertices.std(-2).unsqueeze(-2) + EPS)

    return mesh


def scale(mesh: Type[Mesh], scf: Union[float, Iterable],
          inplace: Optional[bool] = True):
    r"""Scale a mesh given a specified scaling factor. A scalar scaling factor
    can be provided, in which case it is applied isotropically to all dims.
    Optionally, a list/tuple of anisotropic scale factors can be provided per
    dimension.

    Args:
        mesh (Mesh): Mesh to be scaled.
        scf (float or iterable): Scaling factor per dimension. If only a single
            scaling factor is provided (or a list of size 1 is provided), it is
            isotropically applied to all dimensions. Else, a list/tuple of 3
            scaling factors is expected, which are applied to the X, Y, and Z
            directions respectively.
        inplace (bool, optional): Bool to make this operation in-place.

    Returns:
        (Mesh): Scaled mesh.

    """
    if not isinstance(mesh, Mesh):
        raise TypeError('Input mesh must be of type Mesh. '
                        'Got {0} instead.'.format(type(mesh)))
    if not inplace:
        mesh = mesh.clone()

    _scf = []
    if isinstance(scf, float) or isinstance(scf, int):
        _scf = [scf, scf, scf]
    elif isinstance(scf, list) or isinstance(scf, tuple):
        if len(scf) == 1:
            _scf = [scf[0], scf[0], scf[0]]
        elif len(scf) == 3:
            _scf = [scf[0], scf[1], scf[2]]
        else:
            raise ValueError('Exactly 1 or 3 values required for input scf. '
                             'Got {0} instead.'.format(len(scf)))
    else:
        raise TypeError('Input scf must be of type int, float, list, or tuple. '
                        'Got {0} instead.'.format(type(scf)))

    _scf = torch.Tensor(_scf).to(mesh.vertices.device).view(1, 3)
    mesh.vertices = _scf * mesh.vertices

    return mesh


def translate(mesh: Type[Mesh], trans: Union[torch.Tensor, Iterable],
              inplace: Optional[bool] = True):
    r"""Translate a mesh given a (3D) translation vector.

    Args:
        mesh (Mesh): Mesh to be normalized.
        trans (torch.Tensor or iterable): Translation vector (shape:
            torch.Tensor or iterable must have exactly 3 elements).
        inplace (bool, optional): Bool to make this operation in-place.

    Returns:
        (Mesh): Translated mesh.

    """
    if not isinstance(mesh, Mesh):
        raise TypeError('Input mesh must be of type Mesh. '
                        'Got {0} instead.'.format(type(mesh)))
    if not inplace:
        mesh = mesh.clone()
    if torch.is_tensor(trans):
        if trans.numel() != 3:
            raise ValueError('Input trans must contain exactly 3 elements. '
                             'Got {0} instead.'.format(trans.numel()))
        trans = trans.view(1, 3)
    elif isinstance(trans, list) or isinstance(trans, tuple):
        if len(trans) != 3:
            raise ValueError('Exactly 1 or 3 values required for input trans.'
                             'Got {0} instead.'.format(len(trans)))
        trans = torch.Tensor([trans[0], trans[1], trans[2]]).to(
            mesh.vertices.device).view(1, 3)

    mesh.vertices = mesh.vertices + trans
    return mesh


def rotate(mesh: Type[Mesh], rotmat: torch.Tensor,
           inplace: Optional[bool] = True):
    r"""Rotate a mesh given a 3 x 3 rotation matrix.

    Args:
        mesh (Mesh): Mesh to be rotated.
        rotmat (torch.Tensor): Rotation matrix (shape: :math:`3 \times 3`).
        inplace (bool, optional): Bool to make this operation in-place.

    Returns:
        (Mesh): Rotated mesh.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError('Input mesh must be of type Mesh. '
                        'Got {0} instead.'.format(type(mesh)))
    if not inplace:
        mesh = mesh.clone()

    helpers._assert_tensor(rotmat)
    helpers._assert_shape_eq(rotmat, (3, 3))

    mesh.vertices = torch.matmul(rotmat, mesh.vertices.t()).t()

    return mesh


if __name__ == '__main__':

    device = 'cpu'
    mesh = TriangleMesh.from_obj('tests/model.obj')

    # # Test sample_triangle_mesh
    # pts = sample_triangle_mesh(mesh.vertices.to(device),
    #     mesh.faces.to(device), 10)
    # print(pts)

    # # Test normalize
    # mesh = normalize(mesh)

    # # Test scale
    # print(mesh.vertices[:10])
    # mesh = scale(mesh, [2, 1, 2])
    # print(mesh.vertices[:10])

    # # Test translate
    # print(mesh.vertices[:10])
    # mesh = translate(mesh, torch.Tensor([2, 2, 2]))
    # print(mesh.vertices[:10])

    # # Test rotate
    # print(mesh.vertices[:10])
    # rmat = 2 * torch.eye(3)
    # mesh = rotate(mesh, rmat)
    # print(mesh.vertices[:10])
