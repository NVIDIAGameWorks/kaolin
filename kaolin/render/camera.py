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

from __future__ import division

import torch
import torch.nn
from numpy import tan

def rotate_translate_points(points, camera_rot, camera_trans):
    r"""rotate and translate 3D points on based on rotation matrix and transformation matrix.
    Formula is  :math:`\text{P_new} = R * (\text{P_old} - T)`

    Args:
        points (torch.FloatTensor): 3D points, of shape :math:`(\text{batch_size}, \text{num_points}, 3)`.
        camera_rot (torch.FloatTensor): rotation matrix, of shape :math:`(\text{batch_size}, 3, 3)`.
        camera_trans (torch.FloatTensor): translation matrix, of shape :math:`(\text{batch_size}, 3, 1)`.

    Returns:
        (torch.FloatTensor): 3D points in new rotation, of same shape than `points`.
    """
    translated_points = points - camera_trans.view(-1, 1, 3)
    output_points = torch.matmul(translated_points, camera_rot.permute(0, 2, 1))
    return output_points


def generate_rotate_translate_matrices(camera_position, look_at, camera_up_direction):
    r"""generate rotation and translation matrix for given camera parameters.
        Formula is :math:`\text{P_cam} = \text{mtx} * \text{P_world} + \text{shift}`

        Args:
            camera_position (torch.FloatTensor):
                camera positions of shape :math:`(\text{batch_size}, 3)`,
                it means where your cameras are
            look_at (torch.FloatTensor):
                where the camera is watching, of shape :math:`(\text{batch_size}, 3)`,
            camera_up_direction (torch.FloatTensor):
                camera up directions of shape :math:`(\text{batch_size}, 3)`,
                it means what are your camera up directions, generally [0, 1, 0]

        Returns:
            (torch.FloatTensor, torch.FloatTensor):
                the camera rotation matrix of shape :math:`(\text{batch_size}, 3, 3)`
                and the camera transformation matrix of shape :math:`(\text{batch_size}, 3)`
    """

    # 3 variables should be length 1
    camz_bx3 = look_at - camera_position
    camz_length_bx1 = camz_bx3.norm(dim=1, keepdim=True)
    camz_bx3 = camz_bx3 / (camz_length_bx1 + 1e-10)

    camx_bx3 = torch.cross(camz_bx3, camera_up_direction, dim=1)
    camx_len_bx1 = camx_bx3.norm(dim=1, keepdim=True)
    camx_bx3 = camx_bx3 / (camx_len_bx1 + 1e-10)

    camy_bx3 = torch.cross(camx_bx3, camz_bx3, dim=1)
    camy_len_bx3 = camy_bx3.norm(dim=1, keepdim=True)
    camy_bx3 = camy_bx3 / (camy_len_bx3 + 1e-10)

    mtx_bx3x3 = torch.stack([camx_bx3, camy_bx3, -camz_bx3], dim=1)
    shift_bx3 = camera_position

    return mtx_bx3x3, shift_bx3


################################################################
def perspective_camera(points, camera_proj):
    r"""Projects 3D points on 2D images in perspective projection mode.

    Args:
        points (torch.FloatTensor):
            3D points in camera coordinate, of shape :math:`(\text{batch_size}, \text{num_points}, 3)`.
        camera_proj (torch.FloatTensor): projection matrix of shape :math:`(3, 1)`.

    Returns:
        (torch.FloatTensor):
            2D points on image plane of shape :math:`(\text{batch_size}, \text{num_points}, 2)`.
    """

    # perspective, use only one camera intrinsic parameter
    # TODO(cfujitsang): if we have to permute and reshape the camera matrix
    #                   does that mean that they are wrong in the first place ?
    projected_points = points * camera_proj.view(-1, 1, 3)
    projected_2d_points = projected_points[:, :, :2] / projected_points[:, :, 2:3]

    return projected_2d_points


def generate_perspective_projection(fovyangle,
                                    ratio=1.0,
                                    dtype=torch.float):
    r"""Generate perspective projection matrix for a given camera fovy angle.

        Args:
            fovyangle (float):
                field of view angle of y axis, :math:`tan(\frac{fovy}{2}) = \frac{y}{f}`.
            ratio (float):
                aspect ratio :math:`(\frac{width}{height})`. Default: 1.0.

        Returns:
            (torch.FloatTensor):
                camera projection matrix, of shape :math:`(3, 1)`.
        """
    tanfov = tan(fovyangle / 2.0)
    return torch.tensor([[1.0 / (ratio * tanfov)], [1.0 / tanfov], [-1]],
                        dtype=dtype)
