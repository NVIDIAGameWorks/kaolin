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

import logging
import warnings
import torch
import torch.nn
import torch.autograd
from torch.autograd import Function

from kaolin import _C

try:
    import nvdiffrast.torch as nvdiff
    _has_nvdiffrast = True
    _device2glctx = {}
except ImportError:
    _has_nvdiffrast = False
    nvdiff = None
    logger = logging.getLogger(__name__)
    logger.info("Cannot import nvdiffrast")

def _get_nvdiff_glctx(device):
    if device not in _device2glctx:
        _device2glctx[device] = nvdiff.RasterizeGLContext(
            output_db=False, device=device)
    return _device2glctx[device]


__all__ = [
    'rasterize',
]

def _legacy_to_opengl(face_vertices_image, face_vertices_z, valid_faces=None):
    """Transform Kaolin legacy rasterization coordinate system to OpenGL format
    (for nvdiffrast)

    For more details see: https://nvlabs.github.io/nvdiffrast/#coordinate-systems

    Args:
        face_vertices_image (torch.Tensor):
            of shape (batch_size, num_faces, 3, 3) or (num_faces, 3, 3)
        face_vertices_z (torch.Tensor):
            of shape (batch_size, num_faces, 3) or (num_faces, 3)
        valid_faces (torch.BoolTensor):
            of shape (batch_size, num_faces) or (num_faces)

    Returns:
        (torch.Tensor, torch.IntTensor):

        - position of vertices, of shape (batch_size, num_faces * 3, 4) or (num_faces * 3, 4)
        - face indices, of shape (num_faces, 3)
    """
    z = -face_vertices_z / (abs(face_vertices_z).max() + 1e-6)
    _face_vertices_image = face_vertices_image.reshape(*face_vertices_image.shape[:-3], -1, 2)
    pos = torch.stack([
        _face_vertices_image[..., 0],
        -_face_vertices_image[..., 1],
        z.reshape(*z.shape[:-2], -1)
    ], dim=-1)
    if valid_faces is None:
        pos = torch.nn.functional.pad(pos, (0, 1), value=1.)
    else:
        pad = (valid_faces.unsqueeze(-1) * 2. - 1.).expand(*valid_faces.shape, 3)
        pad = pad.reshape(*valid_faces.shape[:-1], -1, 1)
        pos = torch.cat([pos, pad], dim=-1)

    tri = torch.arange(pos.shape[-2], device=pos.device, dtype=torch.int).reshape(-1, 3)
   
    return pos, tri

def _nvdiff_rasterize(height,
                      width,
                      face_vertices_z,
                      face_vertices_image,
                      face_features,
                      valid_faces):
    r"""Base function for ``rasterize`` with backend nvdiffrast
    
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
        valid_faces (torch.BoolTensor):
            Mask of faces being rasterized,
            of shape :math:`(\text{batch_size}, \text{num_faces})`.
            Default: All faces are valid.

    Returns:
        (torch.FloatTensor, torch.LongTensor):

        - The rendered features of shape
          :math:`(\text{batch_size}, \text{height}, \text{width}, \text{num_features})`,
          if `face_features` is a list of torch.FloatTensor, return of torch.FloatTensor,
          of shapes :math:`(\text{batch_size}, \text{height}, \text{width}, \text{num_features[i]})`.
        - The rendered face index, -1 is None,
          of shape :math:`(\text{batch_size}, \text{height}, \text{width})`.
    """
    device = face_vertices_z.device

    if not _has_nvdiffrast:
        raise ValueError("nvdiffrast must be installed to be used as backend, but failed to import. "
                         "See https://nvlabs.github.io/nvdiffrast/#installation for installation instructions.")
    glctx = _get_nvdiff_glctx(device)
    pos, tri = _legacy_to_opengl(face_vertices_image, face_vertices_z, valid_faces)
    
    rast = nvdiff.rasterize(glctx, pos, tri, (height, width), grad_db=False)
    _face_features = face_features.reshape(
        *face_features.shape[:-3], face_features.shape[-3] * 3, -1)
    output = nvdiff.interpolate(_face_features, rast[0], tri)
    interpolated_features = output[0]
    face_idx = (rast[0][..., -1].long() - 1).contiguous()

    return interpolated_features, face_idx

class NvdiffRasterizeFwdCudaBwd(Function):
    r"""torch.autograd.Function for ``rasterize`` with backend nvdiffrast_fwd

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
        valid_faces (torch.BoolTensor):
            Mask of faces being rasterized,
            of shape :math:`(\text{batch_size}, \text{num_faces})`.
            Default: All faces are valid.
        eps (float):
            Epsilon value used to normalize barycentric weights.
            Especially matter with small triangles,
            to increase or decrease in case of exploding or vanishing gradient.
            Default: 1e-8.

    Returns:
        (torch.FloatTensor, torch.LongTensor):

        - The rendered features of shape
          :math:`(\text{batch_size}, \text{height}, \text{width}, \text{num_features})`,
          if `face_features` is a list of torch.FloatTensor, return of torch.FloatTensor,
          of shapes :math:`(\text{batch_size}, \text{height}, \text{width}, \text{num_features[i]})`.
        - The rendered face index, -1 is None,
          of shape :math:`(\text{batch_size}, \text{height}, \text{width})`.
    """
    @staticmethod
    def forward(ctx,
                height,
                width,
                face_vertices_z,
                face_vertices_image,
                face_features,
                valid_faces,
                eps):
        device = face_vertices_z.device

        if not _has_nvdiffrast:
            raise ValueError("nvdiffrast must be installed to be used as backend, but failed to import. "
                             "See https://nvlabs.github.io/nvdiffrast/#installation for installation instructions.")
        glctx = _get_nvdiff_glctx(device)
        pos, tri = _legacy_to_opengl(face_vertices_image, face_vertices_z, valid_faces)

        rast = nvdiff.rasterize(glctx, pos, tri, (height, width), grad_db=False)
        _face_features = face_features.reshape(
            *face_features.shape[:-3], face_features.shape[-3] * 3, -1)
        output = nvdiff.interpolate(_face_features, rast[0], tri)
        interpolated_features = output[0]
        face_idx = (rast[0][..., -1].long() - 1).contiguous()

        output_weights = torch.cat([
            rast[0][..., :2],
            1. - torch.sum(rast[0][..., :2], dim=-1, keepdims=True)
        ], dim=-1)
        ctx.save_for_backward(interpolated_features, face_idx,
                              output_weights, face_vertices_image, face_features)
        ctx.mark_non_differentiable(face_idx)
        ctx.eps = eps

        return interpolated_features, face_idx

    @staticmethod
    def backward(ctx, grad_interpolated_features, grad_face_idx):
        interpolated_features, face_idx, output_weights, \
            face_vertices_image, face_features = ctx.saved_tensors
        eps = ctx.eps

        grad_face_vertices_image, grad_face_features = \
            _C.render.mesh.rasterize_backward_cuda(
                grad_interpolated_features.contiguous(),
                interpolated_features,
                face_idx,
                output_weights,
                face_vertices_image,
                face_features,
                eps)

        return None, None, None, grad_face_vertices_image, \
            grad_face_features, None, None

class RasterizeCuda(Function):
    r"""torch.autograd.Function for ``rasterize`` with backend cuda.
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
        valid_faces (torch.BoolTensor):
            Mask of faces being rasterized,
            of shape :math:`(\text{batch_size}, \text{num_faces})`.
            Default: All faces are valid.
        multiplier (int):
            To avoid numeric issue, we enlarge the coordinates by a multiplier.
            Default: 1000.
        eps (float):
            Epsilon value used to normalize barycentric weights.
            Especially matter with small triangles,
            to increase or decrease in case of exploding or vanishing gradient.
            Default: 1e-8.

    Returns:
        (torch.FloatTensor, torch.LongTensor):

        - The rendered features of shape
          :math:`(\text{batch_size}, \text{height}, \text{width}, \text{num_features})`,
          if `face_features` is a list of torch.FloatTensor, return of torch.FloatTensor,
          of shapes :math:`(\text{batch_size}, \text{height}, \text{width}, \text{num_features[i]})`.
        - The rendered face index, -1 is None,
          of shape :math:`(\text{batch_size}, \text{height}, \text{width})`.
    """
    @staticmethod
    def forward(ctx,
                height,
                width,
                face_vertices_z,
                face_vertices_image,
                face_features,
                valid_faces,
                multiplier,
                eps):
        batch_size = face_vertices_z.shape[0]
        num_faces = face_vertices_z.shape[1]
        feat_dim = face_features.shape[-1]

        device = face_vertices_z.device
        dtype = face_vertices_z.dtype

        # others will be further processed in DIBR forward
        face_features = face_features.contiguous()
        face_vertices_image = face_vertices_image.contiguous()
        if valid_faces is None:
            valid_faces_idx = (
                torch.arange(batch_size, dtype=torch.long,
                             device=device).reshape(-1, 1).repeat(1, num_faces).reshape(-1),
                torch.arange(num_faces, dtype=torch.long,
                             device=device).reshape(1, -1).repeat(batch_size, 1).reshape(-1)
            )
            valid_face_vertices_image = face_vertices_image.reshape(
                batch_size * num_faces, 3, 2)
            valid_face_vertices_z = face_vertices_z.reshape(
                batch_size * num_faces, 3)
            valid_face_features = face_features.reshape(
                batch_size * num_faces, 3, feat_dim)
            num_faces_per_mesh = torch.full((batch_size,), num_faces,
                                            dtype=torch.long, device=device)
        else:
            valid_faces_idx = torch.where(valid_faces)
            valid_face_vertices_image = face_vertices_image[valid_faces_idx[0],
                                                            valid_faces_idx[1]]
            valid_face_vertices_z = face_vertices_z[valid_faces_idx[0],
                                                    valid_faces_idx[1]]
            valid_face_features = face_features[valid_faces_idx[0],
                                                valid_faces_idx[1]]
            num_faces_per_mesh = torch.sum(valid_faces.reshape(batch_size, -1), dim=1)
        first_idx_face_per_mesh = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        torch.cumsum(num_faces_per_mesh, dim=0, out=first_idx_face_per_mesh[1:])

        # To avoid numeric error, we enlarge the coordinate by the multipler
        valid_face_vertices_image = valid_face_vertices_image * multiplier

        # for each face [x1, y1, x2, y2, x3, y3], we can get the bbox [xmin, ymin]
        # the pixel outside bbox will never be influenced by this face.
        # we use bounding box to accelerate pixel-face relation calculation
        points_min = torch.min(valid_face_vertices_image, dim=1)[0]
        points_max = torch.max(valid_face_vertices_image, dim=1)[0]
        valid_face_bboxes = torch.cat((points_min, points_max), dim=1)

        interpolated_features, selected_face_idx, output_weights = \
            _C.render.mesh.packed_rasterize_forward_cuda(
                height,
                width,
                valid_face_vertices_z.contiguous(),
                valid_face_vertices_image.contiguous(),
                valid_face_bboxes.contiguous(),
                valid_face_features.contiguous(),
                first_idx_face_per_mesh.contiguous(),
                multiplier,
                eps)
        face_idx = valid_faces_idx[1][(
            selected_face_idx +
            first_idx_face_per_mesh[:-1].reshape(-1, 1, 1)
        ).reshape(-1)]

        face_idx = face_idx.reshape(selected_face_idx.shape).contiguous()
        face_idx[selected_face_idx == -1] = -1
        ctx.save_for_backward(interpolated_features, face_idx,
                              output_weights, face_vertices_image, face_features)
        ctx.mark_non_differentiable(face_idx)
        ctx.eps = eps

        return interpolated_features, face_idx

    @staticmethod
    def backward(ctx, grad_interpolated_features, grad_face_idx):
        interpolated_features, face_idx, output_weights, \
            face_vertices_image, face_features = ctx.saved_tensors
        eps = ctx.eps

        grad_face_vertices_image, grad_face_features = \
            _C.render.mesh.rasterize_backward_cuda(
                grad_interpolated_features.contiguous(),
                interpolated_features,
                face_idx,
                output_weights,
                face_vertices_image,
                face_features,
                eps)

        return None, None, None, grad_face_vertices_image, \
            grad_face_features, None, None, None, None

def rasterize(height,
              width,
              face_vertices_z,
              face_vertices_image,
              face_features,
              valid_faces=None,
              multiplier=None,
              eps=None,
              backend='cuda'):
    r"""Fully differentiable rasterization implementation,
    that renders 3D triangle meshes with per-vertex per-face features to
    generalized feature "images".

    Backend can be selected among, `nvdiffrast library`_ if available (see `installation instructions`_),
    or custom cuda ops improved from originally proposed by *Chen, Whenzheng, et al.* in 
    `Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer`_ NeurIPS 2019.

    .. note::
       `nvdiffrast library`_ is relying on OpenGL and so can be faster especially
       on larger mesh and resolution.

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
        valid_faces (torch.BoolTensor):
            Mask of faces being rasterized,
            of shape :math:`(\text{batch_size}, \text{num_faces})`.
            Default: All faces are valid.
        multiplier (int):
            To avoid numeric issue, we enlarge the coordinates by a multiplier.
            Used only with ``backend`` 'cuda' at forward pass. Default: 1000.
        eps (float):
            Epsilon value used to normalize barycentric weights.
            Especially matter with small triangles,
            to increase or decrease in case of exploding or vanishing gradient.
            Ignored if ``backend`` is 'nvdiffrast'.
            Default: 1e-8.
        backend (string):
            Backend used for the rasterization, can be ['cuda', 'nvdiffrast', nvdiffrast_fwd'].
            'nvdiffrast_fwd' is using `nvdiffrast library`_ for the forward pass only
            and kaolin's custom Op for backward pass.

    Returns:
        (torch.FloatTensor, torch.LongTensor):

        - The rendered features of shape
          :math:`(\text{batch_size}, \text{height}, \text{width}, \text{num_features})`,
          if `face_features` is a list of torch.FloatTensor, return of torch.FloatTensor,
          of shapes :math:`(\text{batch_size}, \text{height}, \text{width}, \text{num_features[i]})`.
        - The rendered face index, -1 is None,
          of shape :math:`(\text{batch_size}, \text{height}, \text{width})`.

    .. _Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer:
        https://arxiv.org/abs/1908.01210

    .. _nvdiffrast library:
        https://github.com/NVlabs/nvdiffrast

    .. _installation instructions:
        https://nvlabs.github.io/nvdiffrast/#installation
    """
    if multiplier is None:
        multiplier = 1000
    elif backend in ['nvdiffrast', 'nvdiffrast_fwd']:
        warnings.warn(f'in "rasterize": multiplier is ignored with backend "{backend}"',
                      UserWarning)
    if eps is None:
        eps = 1e-8
    elif backend == 'nvdiffrast':
        warnings.warn(f'in "rasterize": eps is ignored with backend "{backend}"',
                      UserWarning)
    batch_size, num_faces, _ = face_vertices_z.shape
    _face_features = torch.cat(face_features, dim=-1) \
        if isinstance(face_features, (list, tuple)) else face_features

    if backend == 'nvdiffrast':
        image_features, face_idx = _nvdiff_rasterize(
            height, width, face_vertices_z, face_vertices_image,
            _face_features, valid_faces)
    elif backend == 'nvdiffrast_fwd':
        image_features, face_idx = NvdiffRasterizeFwdCudaBwd.apply(
            height, width, face_vertices_z, face_vertices_image,
            _face_features, valid_faces, eps)
    elif backend == 'cuda':
        image_features, face_idx = RasterizeCuda.apply(
            height, width, face_vertices_z, face_vertices_image,
            _face_features, valid_faces, multiplier, eps)
    else:
        raise ValueError(f'"{backend}" is not a valid backend, ',
                         'valid choices are ["cuda", "nvdiffrast", "nvdiffrast_fwd"]')

    if isinstance(face_features, (list, tuple)):
        _image_features = []
        cur_idx = 0
        for face_feature in face_features:
            _image_features.append(image_features[..., cur_idx:cur_idx + face_feature.shape[-1]])
            cur_idx += face_feature.shape[-1]
        image_features = tuple(_image_features)
    return image_features, face_idx
