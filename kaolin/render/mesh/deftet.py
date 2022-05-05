# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
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

__all__ = [
    "deftet_sparse_render"
]

def _base_naive_deftet_render(
        pixel_coords,         # (2,)
        render_range,         # (2,)
        face_vertices_image,  # (num_faces, 3, 2)
        face_vertices_z,      # (num_faces, 3)
        face_vertices_min,    # (num_faces, 2)
        face_vertices_max,    # (num_faces, 2)
        valid_faces,          # (num_faces)
        eps):                 # int
    """Base function for :func:`_naive_deftet_sparse_render`
       non-batched and for a single pixel

       This is because most operations are vectorized on faces
       but then only few outputs of those vectorized operations
       are used (so the memory is only used temporarily).
    """
    in_bbox_mask = torch.logical_and(
        pixel_coords.unsqueeze(0) >= face_vertices_min,
        pixel_coords.unsqueeze(0) < face_vertices_max)
    in_bbox_mask = torch.logical_and(in_bbox_mask[:, 0],
                                     in_bbox_mask[:, 1])
    in_bbox_mask = torch.logical_and(in_bbox_mask,
                                     valid_faces)
    in_bbox_idx = torch.where(in_bbox_mask)[0]
    #ax = ax[in_bbox_idx]
    #ay = ay[in_bbox_idx]
    #m = m[in_bbox_idx]
    #p = p[in_bbox_idx]
    #n = n[in_bbox_idx]
    #q = q[in_bbox_idx]
    #k3 = k3[in_bbox_idx]
    #s = pixel_coords[0] - ax
    #t = pixel_coords[1] - ay
    #k1 = s * q - n * t
    #k2 = m * t - s * p

    #w1 = k1 / (k3 + NORM_EPS)
    #w2 = k2 / (k3 + NORM_EPS)
    #w0 = 1. - w1 - w2
    face_vertices_image = face_vertices_image[in_bbox_idx]
    ax = face_vertices_image[:, 0, 0]
    ay = face_vertices_image[:, 0, 1]
    bx = face_vertices_image[:, 1, 0]
    by = face_vertices_image[:, 1, 1]
    cx = face_vertices_image[:, 2, 0]
    cy = face_vertices_image[:, 2, 1]

    a_edge_x = ax - pixel_coords[0];
    a_edge_y = ay - pixel_coords[1];
    b_edge_x = bx - pixel_coords[0];
    b_edge_y = by - pixel_coords[1];
    c_edge_x = cx - pixel_coords[0];
    c_edge_y = cy - pixel_coords[1];
    _w0 = b_edge_x * c_edge_y - b_edge_y * c_edge_x;
    _w1 = c_edge_x * a_edge_y - c_edge_y * a_edge_x;
    _w2 = a_edge_x * b_edge_y - a_edge_y * b_edge_x;
    norm = _w0 + _w1 + _w2;
    norm_eps = eps * torch.sign(norm);
    w0 = _w0 / (norm + norm_eps);
    w1 = _w1 / (norm + norm_eps);
    w2 = _w2 / (norm + norm_eps);

    selected_mask = (w0 >= 0.) & (w1 >= 0.) & (w2 >= 0.)
    selected_face_vertices_z = face_vertices_z[in_bbox_idx][selected_mask]
    selected_weights = torch.stack([
        w0[selected_mask], w1[selected_mask], w2[selected_mask]], dim=-1)
    pixel_depth = torch.sum(selected_weights * face_vertices_z[in_bbox_idx][selected_mask],
                            dim=-1)
    in_render_range_mask = torch.logical_and(
        pixel_depth > render_range[0],
        pixel_depth < render_range[1]
    )
    order = torch.argsort(pixel_depth[in_render_range_mask],
                          descending=True, dim=0)
    return in_bbox_idx[selected_mask][in_render_range_mask][order]

def _naive_deftet_sparse_render(pixel_coords,
                                render_ranges,
                                face_vertices_z,
                                face_vertices_image,
                                face_features,
                                knum,
                                valid_faces=None,
                                eps=1e-8):
    r"""Naive implementation of :func:`deftet_sparse_render`.

    Note:
         The behavior is different than :func:`deftet_sparse_render`
         when knum < max(number of faces per pixel),
         as this returns the first faces by pixel_depth
         while deftet_render returns the first faces
         by the mesh order.

    Note:
         if `face_camera_vertices` and `face_camera_z` are produced by
         camera functions in :mod:`kaolin.render.camera`,
         then the expected range of values for `pixel_coords` is [-1., 1.]
         and the expected of range values for `render_range` is [-inf, 0.].

    Args:
        pixel_coords (torch.Tensor):
            Image coordinates to render,
            of shape :math:`(\text{batch_size}, \text{num_pixels}, 2)`.
        render_ranges (torch.Tensor):
            Range of rendering,
            of shape :math:`(\text{batch_size}, \text{num_pixels}, 2)`.
        face_vertices_z (torch.Tensor):
            3D points values of the face vertices in camera coordinate,
            values in front of camera are expected to be negative,
            higher values being closer to the camera.
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3)`.
        face_vertices_image (torch.Tensor):
            2D positions of the face vertices on image plane,
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3, 2)`,
            Note that face vertices are projected on image plane (z=-1)
            to forms face_vertices_image.
        face_features (torch.Tensor or list of torch.Tensor):
            Features (per-vertex per-face) to be drawn,
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3, \text{feature_dim})`,
            feature is the features dimension,
            for instance with vertex colors num_features=3 (R, G, B),
            and texture coordinates num_features=2 (X, Y),
            or a list of num_features,
            of shapes :math:`(\text{batch_size}, \text{num_faces}, 3, \text{feature_dim[i]})`
        knum (int):
            Maximum number of faces that influence one pixel. Default: 300.
        valid_faces (torch.BoolTensor):
            Mask of faces being rendered,
            of shape :math:`(\text{batch_size}, \text{num_faces})`.
            Default: All faces are valid.
        eps (float):
            Epsilon value used to normalize barycentric weights.
            Default: 1e-8.

    Returns:
        (torch.Tensor or list of torch.Tensor, torch.LongTensor):

            - The rendered features, of shape
              :math:`(\text{batch_size}, \text{num_pixels}, \text{knum}, \text{num_features})`,
              if `face_features` is a list of torch.Tensor,
              then it returns a list of torch.Tensor, of shapes
              :math:`(\text{batch_size}, \text{num_pixels}, \text{knum}, \text{num_features[i]})`.
            - The rendered face index, -1 is void, of shape
              :math:`(\text{batch_size}, \text{num_pixels}, \text{knum})`.
    """
    _face_features = torch.cat(
        face_features, dim=-1
    ) if isinstance(face_features, (list, tuple)) else face_features

    batch_size = pixel_coords.shape[0]
    num_pixels = pixel_coords.shape[1]
    num_faces = face_vertices_z.shape[1]
    feat_dim = _face_features.shape[-1]

    if valid_faces is None:
        valid_faces = torch.ones((batch_size, num_faces),
                                 device=pixel_coords.device,
                                 dtype=torch.bool)

    assert pixel_coords.shape == (batch_size, num_pixels, 2)
    assert render_ranges.shape == (batch_size, num_pixels, 2)
    assert face_vertices_z.shape == (batch_size, num_faces, 3)
    assert face_vertices_image.shape == (batch_size, num_faces, 3, 2)
    assert _face_features.shape == (batch_size, num_faces, 3, feat_dim)
    face_min = torch.min(face_vertices_image, dim=2)[0]
    face_max = torch.max(face_vertices_image, dim=2)[0]
    selected_face_idx = torch.full((batch_size, num_pixels, knum), -1,
                                   device=pixel_coords.device, dtype=torch.long)
    for i in range(batch_size):
        for j in range(num_pixels):
            _face_idx = _base_naive_deftet_render(
                pixel_coords[i, j], render_ranges[i, j], face_vertices_image[i],
                face_vertices_z[i], face_min[i], face_max[i], valid_faces[i], eps)
            selected_face_idx[i, j, :_face_idx.shape[0]] = _face_idx[:knum]
    _idx = selected_face_idx + 1
    ax = face_vertices_image[:, :, 0, 0]
    ay = face_vertices_image[:, :, 0, 1]
    m = face_vertices_image[:, :, 1, 0] - face_vertices_image[:, :, 0, 0]
    p = face_vertices_image[:, :, 1, 1] - face_vertices_image[:, :, 0, 1]
    n = face_vertices_image[:, :, 2, 0] - face_vertices_image[:, :, 0, 0]
    q = face_vertices_image[:, :, 2, 1] - face_vertices_image[:, :, 0, 1]
    k3 = m * q - n * p
    ax = torch.nn.functional.pad(ax, (1, 0), value=0.)
    ay = torch.nn.functional.pad(ay, (1, 0), value=0.)
    m = torch.nn.functional.pad(m, (1, 0), value=0.)
    p = torch.nn.functional.pad(p, (1, 0), value=0.)
    n = torch.nn.functional.pad(n, (1, 0), value=0.)
    q = torch.nn.functional.pad(q, (1, 0), value=0.)
    k3 = torch.nn.functional.pad(k3, (1, 0), value=1.)
    _face_vertices_z = torch.nn.functional.pad(
        face_vertices_z, (0, 0, 1, 0), value=0.)
    _face_features = torch.nn.functional.pad(
        _face_features, (0, 0, 0, 0, 1, 0), value=0.)
    _ax = torch.gather(
        ax, 1, _idx.reshape(batch_size, -1)
    ).reshape(batch_size, num_pixels, knum)
    _ay = torch.gather(
        ay, 1, _idx.reshape(batch_size, -1)
    ).reshape(batch_size, num_pixels, knum)
    _m = torch.gather(
        m, 1, _idx.reshape(batch_size, -1)
    ).reshape(batch_size, num_pixels, knum)
    _p = torch.gather(
        p, 1, _idx.reshape(batch_size, -1)
    ).reshape(batch_size, num_pixels, knum)
    _n = torch.gather(
        n, 1, _idx.reshape(batch_size, -1)
    ).reshape(batch_size, num_pixels, knum)
    _q = torch.gather(
        q, 1, _idx.reshape(batch_size, -1)
    ).reshape(batch_size, num_pixels, knum)
    _k3 = torch.gather(
        k3, 1, _idx.reshape(batch_size, -1)
    ).reshape(batch_size, num_pixels, knum)
    _face_vertices_z = torch.gather(
        _face_vertices_z, 1, _idx.reshape(batch_size, -1, 1).repeat(1, 1, 3)
    ).reshape(batch_size, num_pixels, knum, 3)
    _face_features = torch.gather(
        _face_features, 1, _idx.reshape(batch_size, -1, 1, 1).repeat(1, 1, 3, feat_dim)
    ).reshape(batch_size, num_pixels, knum, 3, feat_dim)

    _s = pixel_coords[:, :, :1] - _ax
    _t = pixel_coords[:, :, 1:] - _ay
    _k1 = _s * _q - _n * _t
    _k2 = _m * _t - _s * _p

    norm_eps = eps * torch.sign(_k3);
    w1 = _k1 / (_k3 + norm_eps)
    w2 = _k2 / (_k3 + norm_eps)
    w0 = 1. - w1 - w2
    weights = torch.stack([w0, w1, w2], dim=-1)
    interpolated_features = torch.sum(_face_features * weights.unsqueeze(-1), dim=-2)

    if isinstance(face_features, (list, tuple)):
        _interpolated_features = []
        cur_idx = 0
        for face_feature in face_features:
            _interpolated_features.append(
                interpolated_features[..., cur_idx:cur_idx + face_feature.shape[-1]])
            cur_idx += face_feature.shape[-1]
        interpolated_features = tuple(_interpolated_features)

    return interpolated_features, selected_face_idx

class DeftetSparseRenderer(Function):
    """torch.autograd.Function for :func:`deftet_sparse_render`."""

    @staticmethod
    def forward(ctx, pixel_coords, render_ranges, face_vertices_z,
                face_vertices_image, face_features, knum, eps):
        # dims
        batch_size = face_vertices_z.shape[0]
        num_faces = face_vertices_z.shape[1]
        feat_dim = face_features.shape[-1]
        pixel_num = pixel_coords.shape[1]

        pixel_coords = pixel_coords.contiguous()
        render_ranges = render_ranges.contiguous()
        face_vertices_z = face_vertices_z.contiguous()
        face_vertices_image = face_vertices_image.contiguous()
        face_features = face_features.contiguous()

        # bbox
        face_min = torch.min(face_vertices_image, dim=2)[0]
        face_max = torch.max(face_vertices_image, dim=2)[0]
        face_bboxes = torch.cat((face_min, face_max), dim=2)

        face_idx, pixel_depth, w0, w1 = _C.render.mesh.deftet_sparse_render_forward_cuda(
            face_vertices_z,
            face_vertices_image,
            face_bboxes,
            pixel_coords,
            render_ranges,
            knum,
            eps)

        sorted_idx = torch.argsort(pixel_depth, descending=True, dim=-1)
        sorted_face_idx = torch.gather(face_idx, -1, sorted_idx).contiguous()
        sorted_w0 = torch.gather(w0, -1, sorted_idx)
        sorted_w1 = torch.gather(w1, -1, sorted_idx)
        sorted_w2 = (sorted_face_idx != -1).float() - (sorted_w0 + sorted_w1)
        _idx = sorted_face_idx + 1
        _idx = _idx.reshape(batch_size, -1, 1, 1).expand(
            batch_size, pixel_num * knum, 3, feat_dim)
        selected_features = torch.gather(
            torch.nn.functional.pad(face_features, (0, 0, 0, 0, 1, 0), value=0.), 1, _idx).reshape(
            batch_size, pixel_num, knum, 3, feat_dim)

        weights = torch.stack([sorted_w0, sorted_w1, sorted_w2], dim=-1).contiguous()
        interpolated_features = torch.sum(weights.unsqueeze(-1) * selected_features,
                                          dim=-2).contiguous()

        ctx.save_for_backward(sorted_face_idx, weights, face_vertices_image, face_features)
        ctx.mark_non_differentiable(sorted_face_idx)
        ctx.eps = eps
        return interpolated_features, sorted_face_idx

    @staticmethod
    def backward(ctx, grad_interpolated_features, grad_face_idx):
        face_idx, weights, face_vertices_image, face_features = ctx.saved_tensors
        eps = ctx.eps

        grad_face_vertices_image = torch.zeros_like(face_vertices_image)
        grad_face_features = torch.zeros_like(face_features)

        grad_face_vertices_image, grad_face_features = \
            _C.render.mesh.deftet_sparse_render_backward_cuda(
                grad_interpolated_features.contiguous(), face_idx, weights,
                face_vertices_image, face_features, eps)

        return None, None, None, grad_face_vertices_image, grad_face_features, None, None


def deftet_sparse_render(pixel_coords, render_ranges, face_vertices_z,
                         face_vertices_image, face_features, knum=300, eps=1e-8):
    r"""Fully differentiable volumetric renderer devised by *Gao et al.* in
    `Learning Deformable Tetrahedral Meshes for 3D Reconstruction`_ NeurIPS 2020.

    This is rasterizing a mesh w.r.t to a list of pixel coordinates,
    but instead of just rendering the closest intersection.
    it will render all the intersections sorted by depth order,
    returning the interpolated features and the indexes of faces intersected
    for each intersection in padded arrays.

    Note:
        The function is not differentiable w.r.t pixel_coords.

    Note:
         if `face_camera_vertices` and `face_camera_z` are produced by
         camera functions in :mod:`kaolin.render.camera`,
         then the expected range of values for `pixel_coords` is [-1., 1.]
         and the expected of range values for `render_range` is [-inf, 0.].

    Args:
        pixel_coords (torch.Tensor):
            Image coordinates to render,
            of shape :math:`(\text{batch_size}, \text{num_pixels}, 2)`.
        render_ranges (torch.Tensor):
            Depth ranges on which intersection get rendered,
            of shape :math:`(\text{batch_size}, \text{num_pixels}, 2)`.
        face_vertices_z (torch.Tensor):
            3D points values of the face vertices in camera coordinate,
            values in front of camera are expected to be negative,
            higher values being closer to the camera.
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3)`.
        face_vertices_image (torch.Tensor):
            2D positions of the face vertices on image plane,
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3, 2)`,
            Note that face vertices are projected on image plane (z=-1)
            to forms face_vertices_image.
        face_features (torch.Tensor or list of torch.Tensor):
            Features (per-vertex per-face) to be drawn,
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3, \text{feature_dim})`,
            feature is the features dimension,
            for instance with vertex colors num_features=3 (R, G, B),
            and texture coordinates num_features=2 (X, Y),
            or a list of num_features,
            of shapes :math:`(\text{batch_size}, \text{num_faces}, 3, \text{feature_dim[i]})`.
        knum (int):
            Maximum number of faces that influence one pixel. Default: 300.
        eps (float):
            Epsilon value used to normalize barycentric weights.
            Default: 1e-8.

    Returns:
        (torch.Tensor or list of torch.Tensor, torch.LongTensor):

            - The rendered features, of shape
              :math:`(\text{batch_size}, \text{num_pixels}, \text{knum}, \text{feature_dim})`,
              if `face_features` is a list of torch.Tensor,
              then it returns a list of torch.Tensor, of shapes
              :math:`(\text{batch_size}, \text{num_pixels}, \text{knum}, \text{feature_dim[i]})`.
            - The rendered face index, -1 is void, of shape
              :math:`(\text{batch_size}, \text{num_pixels}, \text{knum})`.

    .. _Learning Deformable Tetrahedral Meshes for 3D Reconstruction:
            https://arxiv.org/abs/2011.01437
    """
    _face_features = torch.cat(
        face_features, dim=-1
    ) if isinstance(face_features, (list, tuple)) else face_features

    image_features, face_idx = DeftetSparseRenderer.apply(
        pixel_coords, render_ranges, face_vertices_z,
        face_vertices_image, _face_features, knum, eps)
    if isinstance(face_features, (list, tuple)):
        _image_features = []
        cur_idx = 0
        for face_feature in face_features:
            _image_features.append(image_features[..., cur_idx:cur_idx + face_feature.shape[-1]])
            cur_idx += face_feature.shape[-1]
        image_features = tuple(_image_features)
    return image_features, face_idx
