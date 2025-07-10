# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
import warp as wp
import warp.sparse as wps

__all__ = [
    'weight_function_lbs',
    'standard_lbs'
]


@wp.kernel
def _standard_lbs_wp_kernel(
    rest_pts: wp.array2d(dtype=wp.float32),
    tfms: wp.array3d(dtype=wp.float32),
    weights: wp.array2d(dtype=wp.float32),
    output_pts: wp.array2d(dtype=wp.float32),
):  # pragma: no cover
    tid = wp.tid()

    x = wp.vec3(rest_pts[tid, 0], rest_pts[tid, 1], rest_pts[tid, 2])

    # For each transform
    for i in range(tfms.shape[0]):
        w = weights[tid, i]
        if w > 0.0:
            # Extract rotation and translation from transform
            R = wp.mat33(
                tfms[i, 0, 0], tfms[i, 0, 1], tfms[i, 0, 2],
                tfms[i, 1, 0], tfms[i, 1, 1], tfms[i, 1, 2],
                tfms[i, 2, 0], tfms[i, 2, 1], tfms[i, 2, 2]
            )
            t = wp.vec3(tfms[i, 0, 3], tfms[i, 1, 3], tfms[i, 2, 3])

            # Apply weighted transform
            x = x + w * (R * x + t)

    output_pts[tid, 0] = x[0]
    output_pts[tid, 1] = x[1]
    output_pts[tid, 2] = x[2]

# TODO: Warpify this if needed. Kernel is already written
# def weight_function_lbs(torch_pts, wp_tfms, torch_fcn):
#     """
#     Compute the LBS skinning map.
#     """
#     weights = wp.from_torch(torch_fcn(torch_pts))
#     return standard_lbs(wp.from_torch(torch_pts), wp_tfms, weights)


# def standard_lbs(wp_rest_pts, wp_tfms, wp_weights):
#     """
#     Compute the standard LBS skinning map.
#     """
#     wp_output_pts = wp.zeros_like(wp_rest_pts)

#     # Launch kernel to build triplets
#     wp.launch(
#         kernel=_standard_lbs_wp_kernel,
#         dim=wp_rest_pts.shape[0],  # Number of points
#         inputs=[
#             wp_rest_pts,
#             wp_tfms,
#             wp_weights,
#             wp_output_pts
#         ]
#     )
#     return wp.to_torch(wp_output_pts)


def weight_function_lbs(x0, tfms, fcn):
    r"""Applies the linear blend skinning transform batched over all pts: x0 for a batch of transforms, given skinning weight function fcn. Differentiable over fcn.

    Args:
        x0 (torch.Tensor): Rest state points, of shape :math:`(\text{num_pts}, \text{dim})`
        tfms (torch.Tensor): Tensor of affine handles, of shape :math:`(\text{batch_size}, \text{num_handles}, \text{dim}, \text{dim}+1)`
        fcn (callable): Skinning weights function

    Returns:
        (torch.Tensor): Transformed points, of shape :math:`(\text{num_pts}, \text{dim})`
    """
    w_x0 = fcn(x0)
    return standard_lbs(x0, tfms, w_x0)


def standard_lbs(x0, tfms, w_x0):
    r""" 
    Applies the linear blend skinning transform batched over all pts (:math:`x_0`) for a batch of transforms (:math:`T`), given skinning weights (:math:`w_{x0}`), as
    :math:`x_i = \sum_{j=1}^{n} w_j(x_{0,i}) \cdot T_j \begin{bmatrix} x_{0,i} \\ 1 \end{bmatrix} + x_{0,i}`

    Args:
        x0 (torch.Tensor): Rest state points, of shape :math:`(\text{num_pts}, \text{dim})`
        tfms (torch.Tensor): Tensor of affine handles, of shape :math:`(\text{batch_size}, \text{num_handles}, \text{dim}, \text{dim}+1)`
        w_x0 (torch.Tensor): Matrix of skinning weights, of shape :math:`(\text{num_pts}, \text{num_handles})` 

    Returns:
        (torch.Tensor): Transformed points, of shape :math:`(\text{num_pts}, \text{dim})`
    """
    N = x0.shape[0]  # Number of sampled points
    B = tfms.shape[0]  # Sample transform batch size
    H = tfms.shape[1]  # Number of handles
    BH = B * H
    # (N, 1, 3)
    x0_i = x0.unsqueeze(1)
    # (N, 4, 1)
    x03 = torch.cat((x0_i, x0_i.new_ones(N, 1, 1)), dim=2).transpose(1, 2)

    # (N, BH, 3, 4)
    tfms_expanded = tfms.reshape(B * H, 3, 4)
    tfms_expanded = tfms_expanded[None].expand(N, BH, 3, 4)
    # (N, BH, 4, 1)
    x03_expanded = x03[:, None].expand(N, BH, 4, 1)

    # (N, H, 1)
    w_map_x0 = w_x0.unsqueeze(2)

    # (N, BH, 3, 1)
    w_map_x0_expanded = w_map_x0[:, None, :, None, :].expand(
        N, B, H, 3, 1).reshape(N, BH, 3, 1)
    # (N, BH, 3, 1)
    x_map_x0 = w_map_x0_expanded * tfms_expanded @ x03_expanded
    # (N, B, 1, 3)
    x_map_x0 = x_map_x0.reshape(N, B, H, 3, 1)
    x_map_x0 = x_map_x0.sum(2)
    x_map_x0 = x_map_x0.transpose(2, 3)
    x_map_x0 += x0_i[:, None].expand(N, B, 1, 3)
    return x_map_x0
