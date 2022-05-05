# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
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
from torch.autograd import Function

from kaolin import _C
from .rasterization import rasterize, _legacy_to_opengl, nvdiff, _get_nvdiff_glctx

__all__ = [
    "dibr_soft_mask",
    "dibr_rasterization",
]

class DibrSoftMaskCuda(Function):
    @staticmethod
    def forward(ctx, face_vertices_image, selected_face_idx,
                sigmainv, boxlen, knum, multiplier):
        face_vertices_image = face_vertices_image.contiguous()
        face_vertices_image = face_vertices_image * multiplier
        selected_face_idx = selected_face_idx.contiguous()
        points_min = torch.min(face_vertices_image, dim=-2)[0]
        points_max = torch.max(face_vertices_image, dim=-2)[0]
        face_large_bboxes = torch.cat([
            points_min - boxlen * multiplier,
            points_max + boxlen * multiplier
        ], dim=-1)
        soft_mask, close_face_prob, close_face_idx, close_face_dist_type = \
            _C.render.mesh.dibr_soft_mask_forward_cuda(
                face_vertices_image,
                face_large_bboxes.contiguous(),
                selected_face_idx,
                sigmainv,
                knum,
                multiplier
            )
        ctx.multiplier = multiplier
        ctx.sigmainv = sigmainv
        ctx.save_for_backward(
            soft_mask, face_vertices_image, selected_face_idx,
            close_face_prob, close_face_idx, close_face_dist_type
        )
        return soft_mask

    @staticmethod
    def backward(ctx, grad_soft_mask):
        soft_mask, face_vertices_image, selected_face_idx, close_face_prob, \
            close_face_idx, close_face_dist_type = ctx.saved_tensors
        multiplier = ctx.multiplier
        sigmainv = ctx.sigmainv
        grad_face_vertices_image = _C.render.mesh.dibr_soft_mask_backward_cuda(
            grad_soft_mask.contiguous(),
            soft_mask,
            selected_face_idx,
            close_face_prob,
            close_face_idx,
            close_face_dist_type,
            face_vertices_image,
            sigmainv,
            multiplier)
        return grad_face_vertices_image, None, None, None, None, None

def dibr_soft_mask(face_vertices_image, selected_face_idx,
                   sigmainv=7000, boxlen=0.02, knum=30, multiplier=1000.):
    r"""Compute a soft mask generally used with :func:`kaolin.metrics.render.mask_iou`
    to compute a silhouette loss, as defined by *Chen, Wenzheng, et al.* in
    `Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer`_ Neurip 2019.

    Args:
        face_vertices_image (torch.Tensor):
            2D positions of the face vertices on image plane,
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3, 2)`,
            Note that ``face_vertices_camera`` is projected on image plane (z=-1)
            and forms ``face_vertices_image``.
            The coordinates of face_vertices_image are between :math:`[-1, 1]`,
            which corresponds to normalized image pixels.
        selected_face_idx (torch.LongTensor):
            Rendered face index,
            of shape :math:`(\text{batch_size}, \text{height}, \text{width})`.
            See 2nd returned value from :func:`kaolin.render.mesh.rasterize`.
        sigmainv (float):
            Smoothness term for computing the softmask, the higher the sharper.
            The recommended range is :math:`[1/3e-4, 1/3e-5]`. Defaut: 7000.
        boxlen (float):
            Margin over bounding box of faces which will threshold which pixels
            will be influenced by the face. The value should be adapted to sigmainv,
            to threshold values close to 0. The recommended range is [0.05, 0.2].
            Default: 0.02.
        knum (int):
            Maximum number of faces that can influence one pixel.
            The value should be adapted to boxlen, to avoid missing faces.
            The recommended range is [20, 100]. Default: 30.
        multiplier (float):
            To avoid numerical issue,
            we internally enlarge the 2d coordinates by a multiplier.
            Default: 1000.
    Returns:
        (torch.FloatTensor):
        The soft mask, of shape :math:`(\text{batch_size}, \text{height}, \text{width})`.

    .. _Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer:
        https://arxiv.org/abs/1908.01210
    """
    return DibrSoftMaskCuda.apply(face_vertices_image, selected_face_idx,
                                  sigmainv, boxlen, knum, multiplier)

def dibr_rasterization(height, width, face_vertices_z, face_vertices_image,
                       face_features, face_normals_z, sigmainv=7000,
                       boxlen=0.02, knum=30, multiplier=None, eps=None,
                       rast_backend='cuda'):
    r"""Fully differentiable DIB-R renderer implementation,
    that renders 3D triangle meshes with per-vertex per-face features to
    generalized feature "images", soft foreground masks, and face index maps.

    Args:
        height (int): the size of rendered images.
        width (int): the size of rendered images.
        face_vertices_z (torch.FloatTensor):
            3D points depth (z) value of the face vertices in camera coordinate,
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3)`.
        face_vertices_image (torch.FloatTensor):
            2D positions of the face vertices on image plane,
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3, 2)`,
            Note that ``face_vertices_camera`` is projected on image plane (z=-1)
            and forms ``face_vertices_image``.
            The coordinates of face_vertices_image are between :math:`[-1, 1]`,
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
            Normal directions in z axis, of shape :math:`(\text{batch_size}, \text{num_faces})`,
            only faces with normal z >= 0 will be drawn.
        sigmainv (float):
            Smoothness term for computing the softmask, the higher the sharper.
            The recommended range is :math:`[1/3e-4, 1/3e-5]`. Defaut: 7000.
        boxlen (float):
            Margin over bounding box of faces which will threshold which pixels
            will be influenced by the face. The value should be adapted to sigmainv,
            to threshold values close to 0. The recommended range is [0.05, 0.2].
            Default: 0.02.
        knum (int):
            Maximum number of faces that can influence one pixel.
            The value should be adapted to boxlen, to avoid missing faces.
            The recommended range is [20, 100]. Default: 30.
        multiplier (float):
            To avoid numerical issue,
            we internally enlarge the 2d coordinates by a multiplier.
            Default: 1000.
        eps (float):
            Epsilon value used to normalize barycentric weights in rasterization.
            Especially matter with small triangles,
            to increase or decrease in case of exploding or vanishing gradient.
            Ignored if ``backend`` is 'nvdiffrast'.
            Default: 1e-8.
        backend (string):
            Backend used for the rasterization, can be ['cuda', 'nvdiffrast', nvdiffrast_fwd'].
            'nvdiffrast_fwd' is using `nvdiffrast library` for the forward pass only
            and kaolin's custom Op for backward pass.

    Returns:
        (torch.Tensor, torch.Tensor, torch.LongTensor):

        - The rendered features of shape
          :math:`(\text{batch_size}, \text{height}, \text{width}, \text{num_features})`,
          if `face_features` is a list of torch.FloatTensor, return of torch.FloatTensor,
          of shapes :math:`(\text{batch_size}, \text{height}, \text{width}, \text{num_features[i]})`.
        - The rendered soft mask, of shape :math:`(\text{batch_size}, \text{height}, \text{width})`.
          It is generally used with :func:`kaolin.metrics.render.mask_iou` to compute the silhouette loss.
        - The rendered face index, -1 is None,
          of shape :math:`(\text{batch_size}, \text{height}, \text{width})`.
    """
    interpolated_features, face_idx = rasterize(
        height, width,
        face_vertices_z,
        face_vertices_image,
        face_features,
        face_normals_z >= 0.,
        multiplier,
        eps,
        rast_backend
    )
    _multiplier = 1000. if multiplier is None else multiplier
    soft_mask = dibr_soft_mask(
        face_vertices_image,
        face_idx,
        sigmainv,
        boxlen,
        knum,
        _multiplier
    )
    return interpolated_features, soft_mask, face_idx
