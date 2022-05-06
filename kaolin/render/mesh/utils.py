# Copyright (c) 2019,20-21-22 NVIDIA CORPORATION & AFFILIATES.
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

from __future__ import division

import torch

from .. import camera
from ... import ops

def texture_mapping(texture_coordinates, texture_maps, mode='nearest'):
    r"""Interpolates texture_maps by dense or sparse texture_coordinates.
    This function supports sampling texture coordinates for:
    1. An entire 2D image
    2. A sparse point cloud of texture coordinates.

    Args:
        texture_coordinates(torch.FloatTensor):
            dense image texture coordinate, of shape :math:`(\text{batch_size}, h, w, 2)` or
            sparse texture coordinate for points, of shape :math:`(\text{batch_size}, \text{num_points}, 2)`
            Coordinates are expected to be normalized between [0, 1].
            Note that opengl tex coord is different from pytorch's coord.
            opengl coord ranges from 0 to 1, y axis is from bottom to top
            and it supports circular mode(-0.1 is the same as 0.9)
            pytorch coord ranges from -1 to 1, y axis is from top to bottom and does not support circular
            filtering is the same as the mode parameter for torch.nn.functional.grid_sample.
        texture_maps(torch.FloatTensor):
            textures of shape :math:`(\text{batch_size}, \text{num_channels}, h', w')`.
            Here, :math:`h'` & :math:`w'` are the height and width of texture maps.

            If ``texture_coordinates`` are image texture coordinates -
            For each pixel in the rendered image of height we use the coordinates in
            texture_coordinates to query corresponding value in texture maps.
            Note that height :math:`h` and width :math:`w` of the rendered image could be different from
            :math:`h'` & :math:`w'`.

            If ``texture_coordinates`` are sparse texture coordinates -
            For each point in ``texture_coordinates`` we query the corresponding value in ``texture_maps``.

    Returns:
        (torch.FloatTensor):
        interpolated texture of shape :math:`(\text{batch_size}, h, w, \text{num_channels})` or
        interpolated texture of shape :math:`(\text{batch_size}, \text{num_points}, \text{num_channels})`
    """
    batch_size = texture_coordinates.shape[0]
    num_channels = texture_maps.shape[1]
    _texture_coordinates = texture_coordinates.reshape(batch_size, -1, 1, 2)

    # convert coord mode from ogl to pytorch
    # some opengl texture coordinate is larger than 1 or less than 0
    # in opengl it will be normalized by remainder
    # we do the same in pytorch
    _texture_coordinates = torch.clamp(_texture_coordinates, 0., 1.)
    _texture_coordinates = _texture_coordinates * 2 - 1  # [0, 1] to [-1, 1]
    _texture_coordinates[:, :, :, 1] = -_texture_coordinates[:, :, :, 1]  # reverse y

    # sample
    texture_interpolates = torch.nn.functional.grid_sample(texture_maps,
                                                           _texture_coordinates,
                                                           mode=mode,
                                                           align_corners=False,
                                                           padding_mode='border')
    texture_interpolates = texture_interpolates.permute(0, 2, 3, 1)
    return texture_interpolates.reshape(batch_size, *texture_coordinates.shape[1:-1], num_channels)

def spherical_harmonic_lighting(imnormal, lights):
    r"""Creates lighting effects.

    Follows convention set by *Wojciech Jarosz* in 
    `Efficient Monte Carlo Methods for Light Transport in Scattering Media`_.

    .. deprecated:: 0.13.0
       This function is deprecated. Use :func:`kaolin.render.lighting.sh9_irradiance`.

    Args:
        imnormal (torch.FloatTensor):
            per pixel normal, of shape :math:`(\text{batch_size}, \text{height}, \text{width}, 3)`
        lights (torch.FloatTensor):
            spherical harmonic lighting parameters, of shape :math:`(\text{batch_size}, 9)`

    Returns:
        (torch.FloatTensor):
            lighting effect, shape of :math:`(\text{batch_size}, \text{height}, \text{width})`

    .. _Efficient Monte Carlo Methods for Light Transport in Scattering Media:

    https://cs.dartmouth.edu/~wjarosz/publications/dissertation/appendixB.pdf

    """
    # SH lighting
    # light effect
    x = imnormal[:, :, :, 0]
    y = imnormal[:, :, :, 1]
    z = imnormal[:, :, :, 2]

    # spherical harmonic parameters
    band0 = 0.28209479177 * torch.ones_like(x)
    band1_m1 = 0.4886025119 * x
    band1_0 = 0.4886025119 * z
    band1_p1 = 0.4886025119 * y
    band2_m2 = 1.09254843059 * (x * y)
    band2_m1 = 1.09254843059 * (y * z)
    band2_0 = 0.94617469575 * (z * z) - 0.31539156525
    band2_p1 = 0.77254840404 * (x * z)
    band2_p2 = 0.38627420202 * (x * x - y * y)

    bands = torch.stack([band0,
                         band1_m1, band1_0, band1_p1,
                         band2_m2, band2_m1, band2_0, band2_p1, band2_p2],
                        dim=3)
    lighting_effect = torch.sum(bands * lights.view(-1, 1, 1, 9),
                                dim=3)

    return lighting_effect

def prepare_vertices(vertices, faces, camera_proj, camera_rot=None, camera_trans=None,
                     camera_transform=None):
    r"""Wrapper function to move and project vertices to cameras then index them with faces.

    Args:
        vertices (torch.Tensor):
            the meshes vertices, of shape :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        faces (torch.LongTensor):
            the meshes faces, of shape :math:`(\text{num_faces}, \text{face_size})`.
        camera_proj (torch.Tensor):
            the camera projection vector, of shape :math:`(3, 1)`.
        camera_rot (torch.Tensor, optional):
            the camera rotation matrices,
            of shape :math:`(\text{batch_size}, 3, 3)`.
        camera_trans (torch.Tensor, optional):
            the camera translation vectors,
            of  shape :math:`(\text{batch_size}, 3)`.
        camera_transform (torch.Tensor, optional):
            the camera transformation matrices,
            of shape :math:`(\text{batch_size}, 4, 3)`.
            Replace `camera_trans` and `camera_rot`.
    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor):
            The vertices in camera coordinate indexed by faces,
            of shape :math:`(\text{batch_size}, \text{num_faces}, \text{face_size}, 3)`.
            The vertices in camera plan coordinate indexed by faces,
            of shape :math:`(\text{batch_size}, \text{num_faces}, \text{face_size}, 2)`.
            The face normals, of shape :math:`(\text{batch_size}, \text{num_faces}, 3)`.
    """
    # Apply the transformation from camera_rot and camera_trans or camera_transform
    if camera_transform is None:
        assert camera_trans is not None and camera_rot is not None, \
            "camera_transform or camera_trans and camera_rot must be defined"
        vertices_camera = camera.rotate_translate_points(vertices, camera_rot,
                                                         camera_trans)
    else:
        assert camera_trans is None and camera_rot is None, \
            "camera_trans and camera_rot must be None when camera_transform is defined"
        padded_vertices = torch.nn.functional.pad(
            vertices, (0, 1), mode='constant', value=1.
        )
        vertices_camera = (padded_vertices @ camera_transform)
    # Project the vertices on the camera image plan
    vertices_image = camera.perspective_camera(vertices_camera, camera_proj)
    face_vertices_camera = ops.mesh.index_vertices_by_faces(vertices_camera, faces)
    face_vertices_image = ops.mesh.index_vertices_by_faces(vertices_image, faces)
    face_normals = ops.mesh.face_normals(face_vertices_camera, unit=True)
    return face_vertices_camera, face_vertices_image, face_normals
