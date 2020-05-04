# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
# 
# 
# Soft Rasterizer (SoftRas)
# 
# Copyright (c) 2017 Hiroharu Kato
# Copyright (c) 2018 Nikos Kolotouros
# Copyright (c) 2019 Shichen Liu
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn.functional as F


def compute_ambient_light(
        face_vertices: torch.Tensor,
        textures: torch.Tensor,
        ambient_intensity: float = 1.,
        ambient_color: torch.Tensor = None):
    r"""Computes ambient lighting to a mesh, given faces and face textures.

    Args:
        face_vertices (torch.Tensor): A tensor containing a list of (per-face)
            vertices of the mesh (shape: `B` :math:`\times` `num_faces`
            :math:`\times 9`). Here, :math:`B` is the batchsize, `num_faces`
            is the number of faces in the mesh, and since each face is assumed
            to be a triangle, it has 3 vertices, and hence 9 coordinates in
            total.
        textures (torch.Tensor): TODO: Add docstring
        ambient_intensity (float): Intensity of ambient light (in the range
            :math:`\left[0, 1\right]`). If the values provided are outside
            this range, we clip them so that they fall in range.
        ambient_color (torch.Tensor): Color of the ambient light (R, G, B)
            (shape: :math:`3`)

    Returns:
        light (torch.Tensor): A light tensor, which can be elementwise
            multiplied with the textures, to obtain the mesh with lighting
            applied (shape: `B` :math:`\times` `num_faces` :math:`\times
            1 \times 1 \times 1 \times 3`)

    """

    if not torch.is_tensor(face_vertices):
        raise TypeError('Expected input face_vertices to be of type '
                        'torch.Tensor. Got {0} instead.'.format(
                            type(face_vertices)))
    if not torch.is_tensor(textures):
        raise TypeError('Expected input textures to be of type '
                        'torch.Tensor. Got {0} instead.'.format(
                            type(textures)))
    if not isinstance(ambient_intensity, float) and not isinstance(
            ambient_intensity, int):
        raise TypeError('Expected input ambient_intensity to be of '
                        'type float. Got {0} instead.'.format(
                            type(ambient_intensity)))
    if ambient_color is None:
        ambient_color = torch.ones(3, dtype=face_vertices.dtype,
                                   device=face_vertices.device)
    if not torch.is_tensor(ambient_color):
        raise TypeError('Expected input ambient_color to be of type '
                        'torch.Tensor. Got {0} instead.'.format(
                            type(ambient_color)))

    # if face_vertices.dim() != 3 or face_vertices.shape[-1] != 9:
    #     raise ValueError('Input face_vertices must have 3 dimensions '
    #                      'and be of shape (..., ..., 9). Got {0} dimensions '
    #                      'and shape {1} instead.'.format(face_vertices.dim(),
    #                                                      face_vertices.shape))
    # TODO: check texture dims
    if ambient_color.dim() != 1 or ambient_color.shape != (3,):
        raise ValueError('Input ambient_color must have 1 dimension '
                         'and be of shape 3. Got {0} dimensions and shape {1} '
                         'instead.'.format(ambient_color.dim(), ambient_color.shape))

    # Clip ambient_intensity to be in the range [0, 1]
    if ambient_intensity < 0:
        ambient_intensity = 0.
    if ambient_intensity > 1:
        ambient_intensity = 1.

    batchsize = face_vertices.shape[0]
    num_faces = face_vertices.shape[1]
    device = face_vertices.device

    if ambient_color.dim() == 1:
        ambient_color = ambient_color[None, :].to(device)

    light = torch.zeros(batchsize, num_faces, 3).to(device)

    # If the intensity of the ambient light is 0, do not
    # bother computing lighting.
    if ambient_intensity == 0:
        return light

    # Ambient lighting is constant everywhere, and is given as
    # I = I_a * K_a
    # where,
    # I: Intensity at a vertex
    # I_a: Intensity of the ambient light
    # K_a: Ambient reflectance of the vertex (3 channels, R, G, B)
    light += ambient_intensity * ambient_color[:, None, :]

    return light[:, :, None, :]


def apply_ambient_light(
        face_vertices: torch.Tensor,
        textures: torch.Tensor,
        ambient_intensity: float = 1.,
        ambient_color: torch.Tensor = torch.ones(3)):
    r"""Computes and applies ambient lighting to a mesh, given faces and
    face textures.

    Args:
        face_vertices (torch.Tensor): A tensor containing a list of (per-face)
            vertices of the mesh (shape: `B` :math:`\times` `num_faces`
            :math:`\times 9`). Here, :math:`B` is the batchsize, `num_faces`
            is the number of faces in the mesh, and since each face is assumed
            to be a triangle, it has 3 vertices, and hence 9 coordinates in
            total.
        textures (torch.Tensor): TODO: Add docstring
        ambient_intensity (float): Intensity of ambient light (in the range
            :math:`\left[0, 1\right]`). If the values provided are outside
            this range, we clip them so that they fall in range.
        ambient_color (torch.Tensor): Color of the ambient light (R, G, B)
            (shape: :math:`3`)

    Returns:
        textures (torch.Tensor): Updated textures, with ambient lighting
            applied (shape: same as input `textures`) #TODO: Update docstring

    """

    light = compute_ambient_light(face_vertices, textures, ambient_intensity,
                                  ambient_color)
    return textures * light


def compute_directional_light(
        face_vertices: torch.Tensor,
        textures: torch.Tensor,
        directional_intensity: float = 1.,
        directional_color: torch.Tensor = None,
        direction: torch.Tensor = None):
    r"""Computes directional lighting to a mesh, given faces and face textures.

    Args:
        face_vertices (torch.Tensor): A tensor containing a list of (per-face)
            vertices of the mesh (shape: `B` :math:`\times` `num_faces`
            :math:`\times 9`). Here, :math:`B` is the batchsize, `num_faces`
            is the number of faces in the mesh, and since each face is assumed
            to be a triangle, it has 3 vertices, and hence 9 coordinates in
            total.
        textures (torch.Tensor): TODO: Add docstring
        directional_intensity (float): Intensity of directional light (in the
            range :math:`\left[0, 1\right]`). If the values provided are
            outside this range, we clip them so that they fall in range.
        directional_color (torch.Tensor): Color of the directional light
            (R, G, B) (shape: :math:`3`).
        direction (torch.Tensor): Direction of light from the light source.
            (default: :math:`\left( 0, 1, 0 \right)^T`)

    Returns:
        light (torch.Tensor): A light tensor, which can be elementwise
            multiplied with the textures, to obtain the mesh with lighting
            applied (shape: `B` :math:`\times` `num_faces` :math:`\times
            1 \times 1 \times 1 \times 3`)

    """

    if not torch.is_tensor(face_vertices):
        raise TypeError('Expected input face_vertices to be of type '
                        'torch.Tensor. Got {0} instead.'.format(
                            type(face_vertices)))
    if not torch.is_tensor(textures):
        raise TypeError('Expected input textures to be of type '
                        'torch.Tensor. Got {0} instead.'.format(
                            type(textures)))
    if not isinstance(directional_intensity, float) and not isinstance(
            directional_intensity, int):
        raise TypeError('Expected input directional_intensity to be of '
                        'type float. Got {0} instead.'.format(
                            type(directional_intensity)))
    if directional_color is None:
        directional_color = torch.ones(3, dtype=face_vertices.dtype,
                                       device=face_vertices.device)
    if not torch.is_tensor(directional_color):
        raise TypeError('Expected input directional_color to be of type '
                        'torch.Tensor. Got {0} instead.'.format(
                            type(directional_color)))
    if direction is None:
        direction = torch.tensor([0., 1., 0.], dtype=face_vertices.dtype,
                                 device=face_vertices.device)
    if not torch.is_tensor(direction):
        raise TypeError('Expected input direction to be of type '
                        'torch.Tensor. Got {0} instead.'.format(type(direction)))

    # if face_vertices.dim() != 3 or face_vertices.shape[-1] != 9:
    #     raise ValueError('Input face_vertices must have 3 dimensions '
    #                      'and be of shape (..., ..., 9). Got {0} dimensions '
    #                      'and shape {1} instead.'.format(face_vertices.dim(),
    #                                                      face_vertices.shape))
    # TODO: check texture dims
    if directional_color.dim() != 1 or directional_color.shape != (3,):
        raise ValueError('Input directional_color must have 1 dimension '
                         'and be of shape 3. Got {0} dimensions and shape {1} '
                         'instead.'.format(directional_color.dim(),
                                           directional_color.shape))
    if direction.dim() != 1 or direction.shape != (3,):
        raise ValueError('Input direction must have 1 dimension and be '
                         'of shape 3. Got {0} dimensions and shape {1} '
                         'instead.'.format(direction.dim(), direction.shape))

    batchsize = face_vertices.shape[0]
    num_faces = face_vertices.shape[1]
    device = face_vertices.device

    if directional_color.dim() == 1:
        directional_color = directional_color[None, :].to(device)
    if direction.dim() == 1:
        direction = direction[None, :].to(device)

    # Clip directional intensity to be in the range [0, 1]
    if directional_intensity < 0:
        directional_intensity = 0.
    if directional_intensity > 1:
        directional_intensity = 1.

    # Initialize light to zeros
    light = torch.zeros(batchsize, num_faces, 3).to(device)

    # If the intensity of the directional light is 0, do not
    # bother computing lighting.
    if directional_intensity == 0:
        return light

    # Compute face normals.
    v10 = face_vertices[:, :, 0] - face_vertices[:, :, 1]
    v12 = face_vertices[:, :, 2] - face_vertices[:, :, 1]
    normals = F.normalize(torch.cross(v12, v10), p=2, dim=2, eps=1e-6)
    # Reshape, to get back the batchsize dimension.
    normals = normals.reshape(batchsize, num_faces, 3)

    # Get direction to 3 dimensions, if not already there.
    if direction.dim() == 2:
        direction = direction[:, None, :]

    cos = F.relu(torch.sum(normals * direction, dim=2))

    light += directional_intensity * (directional_color[:, None, :]
                                      * cos[:, :, None])

    return light[:, :, None, :]


def apply_directional_light(
        face_vertices: torch.Tensor,
        textures: torch.Tensor,
        directional_intensity: float = 1.,
        directional_color: torch.Tensor = torch.ones(3),
        direction: torch.Tensor = torch.FloatTensor([0, 1, 0])):
    r"""Computes and applies directional lighting to a mesh, given faces
    and face textures.

    Args:
        face_vertices (torch.Tensor): A tensor containing a list of (per-face)
            vertices of the mesh (shape: `B` :math:`\times` `num_faces`
            :math:`\times 9`). Here, :math:`B` is the batchsize, `num_faces`
            is the number of faces in the mesh, and since each face is assumed
            to be a triangle, it has 3 vertices, and hence 9 coordinates in
            total.
        textures (torch.Tensor): TODO: Add docstring
        directional_intensity (float): Intensity of directional light (in the
            range :math:`\left[0, 1\right]`). If the values provided are
            outside this range, we clip them so that they fall in range.
        directional_color (torch.Tensor): Color of the directional light
            (R, G, B) (shape: :math:`3`).
        direction (torch.Tensor): Direction of light from the light source.
            (default: :math:`\left( 0, 1, 0 \right)^T`)

    Returns:
        light (torch.Tensor): A light tensor, which can be elementwise
            multiplied with the textures, to obtain the mesh with lighting
            applied (shape: `B` :math:`\times` `num_faces` :math:`\times
            1 \times 1 \times 1 \times 3`)

    """

    light = compute_directional_light(face_vertices, textures,
                                      directional_intensity, directional_color, direction)
    return textures * light
