# Copyright (c) 2019,20-21 NVIDIA CORPORATION & AFFILIATES.
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
import torch.nn
import torch.autograd
from torch.autograd import Function

from kaolin import _C


class DIBRasterization(Function):
    """torch.autograd.Function for dibr_rasterization.

    Refer to :func:`dibr_rasterization`.
    """
    @staticmethod
    def forward(ctx,
                height,
                width,
                face_vertices_z,
                face_vertices_image,
                face_features,
                face_normals_z,
                sigmainv,
                boxlen,
                knum,
                multiplier):

        batch_size = face_vertices_z.shape[0]
        num_faces = face_vertices_z.shape[1]
        num_features = face_features.shape[-1]

        dev = face_vertices_z.device
        dtype = face_vertices_z.dtype

        # others will be further processed in DIBR forward
        face_features = face_features.contiguous()
        face_vertices_image = face_vertices_image.contiguous()
        face_normals_z = face_normals_z.contiguous()

        # To avoid numeric error, we enlarge the coordinate by the multipler
        face_vertices_image = multiplier * face_vertices_image

        # for each face [x1, y1, x2, y2, x3, y3], we can get the bbox [xmin, ymin]
        # the pixel outside bbox will never be influenced by this face.
        # we use bounding box to accelerate pixel-face relation calculation
        points_min = torch.min(face_vertices_image, dim=2)[0]
        points_max = torch.max(face_vertices_image, dim=2)[0]
        face_bboxes = torch.cat((points_min, points_max), dim=2)
        are_faces_valid = face_normals_z.squeeze(-1) >= 0
        valid_faces = torch.where(are_faces_valid)
        valid_face_bboxes = face_bboxes[valid_faces[0], valid_faces[1]]
        valid_face_vertices_image = face_vertices_image[valid_faces[0], valid_faces[1]]
        valid_face_vertices_z = face_vertices_z[valid_faces[0], valid_faces[1]]
        valid_face_features = face_features[valid_faces[0], valid_faces[1]]
        first_idx_face_per_mesh = torch.zeros(batch_size + 1, dtype=torch.long, device=dev)
        torch.cumsum(torch.sum(are_faces_valid.reshape(batch_size, -1), dim=1), dim=0,
                     out=first_idx_face_per_mesh[1:])

        # in soft mask calculation, we assum a pixel will be influenced by nearby faces
        # we extend the stict bbox to a bigger bbox
        # and assume a pixel will be influced by faces cover it with bigger bbox.
        # we use bounding box to accelerate pixel-face relation calculation
        points_min = points_min - boxlen * multiplier
        points_max = points_max + boxlen * multiplier
        face_large_bboxes = torch.cat((points_min, points_max), dim=2)

        selected_face_idx = torch.full((batch_size, height, width),
                                       -1,
                                       dtype=torch.long,
                                       device=dev)

        # pixel barycentric weights
        output_weights = torch.zeros((batch_size, height, width, 3),
                                     dtype=dtype,
                                     device=dev)

        # rendered image
        interpolated_features = torch.zeros((batch_size, height, width, num_features),
                                            dtype=dtype,
                                            device=dev)

        _C.render.mesh.packed_rasterize_forward_cuda(
            valid_face_vertices_z.contiguous(),
            valid_face_vertices_image.contiguous(),
            valid_face_bboxes.contiguous(),
            valid_face_features.contiguous(),
            first_idx_face_per_mesh.contiguous(),
            selected_face_idx,
            output_weights,
            interpolated_features,
            multiplier)

        face_idx = valid_faces[1][(selected_face_idx +
                                   first_idx_face_per_mesh[:-1].reshape(-1, 1, 1)).reshape(-1)]
        face_idx = face_idx.reshape(selected_face_idx.shape).contiguous()
        face_idx[selected_face_idx == -1] = -1
        # create variables that would be used to save intermediate values
        # used in backward.
        # rendered probablity of each pixel being influced by all the faces
        # namely, a soft mask
        improb_bxhxwx1 = torch.zeros((batch_size, height, width),
                                     dtype=dtype,
                                     device=dev)

        # intermediate variables for accelerating mask calculation
        probface_bxhxwxk = torch.zeros((batch_size, height, width, knum),
                                       dtype=dtype,
                                       device=dev)
        probcase_bxhxwxk = torch.zeros((batch_size, height, width, knum),
                                       dtype=dtype,
                                       device=dev)
        probdis_bxhxwxk = torch.zeros((batch_size, height, width, knum),
                                      dtype=dtype,
                                      device=dev)

        _C.render.mesh.generate_soft_mask_cuda(
            face_vertices_image,
            face_large_bboxes,
            face_idx,
            probface_bxhxwxk,
            probcase_bxhxwxk,
            probdis_bxhxwxk,
            improb_bxhxwx1,
            multiplier,
            sigmainv)

        ctx.sigmainv = sigmainv
        ctx.multiplier = multiplier
        ctx.save_for_backward(interpolated_features, improb_bxhxwx1, face_idx,
                              output_weights, face_vertices_image, face_features,
                              probface_bxhxwxk, probcase_bxhxwxk, probdis_bxhxwxk)

        ctx.mark_non_differentiable(face_idx)
        return interpolated_features, improb_bxhxwx1, face_idx

    @staticmethod
    def backward(ctx,
                 grad_interpolated_features,
                 grad_improb_bxhxwx1,
                 grad_face_idx):

        interpolated_features, improb_bxhxwx1, face_idx, imwei_bxhxwx3, \
            face_vertices_image, face_features, probface_bxhxwxk, \
            probcase_bxhxwxk, probdis_bxhxwxk = ctx.saved_tensors

        sigmainv = ctx.sigmainv
        multiplier = ctx.multiplier

        # vertices gradients from barycentric interpolation
        grad_face_vertices_image = torch.zeros_like(face_vertices_image)
        # vertices gradients from soft mask
        grad_points2dprob_bxfx6 = torch.zeros_like(face_vertices_image)
        # features gradients
        grad_face_features = torch.zeros_like(face_features)

        colors_bxfx3d = face_features
        gradcolors_bxfx3d = grad_face_features
        _C.render.mesh.rasterize_backward_cuda(
            grad_interpolated_features.contiguous(),
            grad_improb_bxhxwx1.contiguous(),
            interpolated_features,
            improb_bxhxwx1,
            face_idx,
            imwei_bxhxwx3,
            probface_bxhxwxk,
            probcase_bxhxwxk,
            probdis_bxhxwxk,
            face_vertices_image,
            colors_bxfx3d,
            grad_face_vertices_image,
            gradcolors_bxfx3d,
            grad_points2dprob_bxfx6,
            multiplier,
            sigmainv)

        return None, None, None, grad_face_vertices_image + grad_points2dprob_bxfx6, \
            grad_face_features, None, None, None, None, None

def dibr_rasterization(height,
                       width,
                       face_vertices_z,
                       face_vertices_image,
                       face_features,
                       face_normals_z,
                       sigmainv=7000,
                       boxlen=0.02,
                       knum=30,
                       multiplier=1000):
    r"""Fully differentiable DIB-R renderer implementation,
    that renders 3D triangle meshes with per-vertex per-face features to
    generalized feature "images", soft foreground masks, depth and face index maps.

    See for usage with textures and lighting.

    Originally proposed by *Chen, Whenzheng, et al.* in
    `Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer`_ NeurIPS 2019

    Args:
        height (int): the size of rendered images
        width (int): the size of rendered images
        face_vertices_z (torch.FloatTensor):
            3D points depth (z) value of the face vertices in camera coordinate,
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3)`.
        face_vertices_image (torch.FloatTensor):
            2D positions of the face vertices on image plane,
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3, 2)`,
            Note that face_vertices_camera is projected on image plane (z=-1)
            and forms face_vertices_image.
            The coordinates of face_vertices_image are between [-1, 1],
            which corresponds to normalized image pixels.
        face_features (torch.FloatTensor or list of torch.FloatTensor):
            Features (per-vertex per-face) to be drawn,
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3, \text{feature_dim})`,
            feature is the features dimension,
            for instance with vertex colors num_features=3 (R, G, B),
            and texture coordinates num_features=2 (X, Y),
            or a list of num_features,
            of shapes :math:`(\text{batch_size}, \text{num_faces}, 3, \text{feature_dim[i]})`
        face_normals_z (torch.FloatTensor):
            Normal directions in z axis, fo shape :math:`(\text{batch_size}, \text{num_faces})`,
            only faces with normal z >= 0 will be drawn
        sigmainv (int):
            Smoothness term for soft mask, the higher, the sharper,
            the range is [1/3e-4, 1/3e-5]. Default: 7000.
        boxlen (float):
            We assume the pixel will only be influenced by nearby faces and boxlen controls the area size,
            the range is [0.05, 0.2]. Default: 0.1.
        knum (int):
            Maximum faces that influence one pixel. The range is [20, 100]. Default: 30.
            Note that the higher boxlen, the bigger knum.
        multiplier (int):
            To avoid numeric issue, we enlarge the coordinates by a multiplier. Default: 1000.

    Returns:
        (torch.FloatTensor, torch.FloatTensor, torch.LongTensor):

        - The rendered features of shape 
          :math:`(\text{batch_size}, \text{height}, \text{width}, \text{num_features})`,
          if `face_features` is a list of torch.FloatTensor, return of torch.FloatTensor,
          of shapes :math:`(\text{batch_size}, \text{height}, \text{width}, \text{num_features[i]})`.
        - The rendered soft mask. It is generally sued in IoU loss to deform the shape,
          of shape :math:`(\text{batch_size}, \text{height}, \text{width})`.
        - The rendered face index, 0 is void and face index start from 1,
          of shape :math:`(\text{batch_size}, \text{height}, \text{width})`.

    .. _Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer:
        https://arxiv.org/abs/1908.01210
    """
    _face_features = torch.cat(face_features, dim=-1) if isinstance(face_features, (list, tuple)) else \
        face_features

    image_features, soft_mask, face_idx = DIBRasterization.apply(
        height, width, face_vertices_z, face_vertices_image,
        _face_features, face_normals_z, sigmainv, boxlen, knum, multiplier)
    if isinstance(face_features, (list, tuple)):
        _image_features = []
        cur_idx = 0
        for face_feature in face_features:
            _image_features.append(image_features[..., cur_idx:cur_idx + face_feature.shape[-1]])
            cur_idx += face_feature.shape[-1]
        image_features = tuple(_image_features)
    return image_features, soft_mask, face_idx
