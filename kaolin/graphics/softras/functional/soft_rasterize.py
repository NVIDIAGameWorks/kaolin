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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import soft_renderer.cuda.soft_rasterize as soft_rasterize_cuda


class SoftRasterizeFunction(Function):

    @staticmethod
    def forward(ctx, face_vertices, textures, image_size=256, 
                background_color=[0, 0, 0], near=1, far=100, 
                fill_back=True, eps=1e-3,
                sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                texture_type='surface'):

        # face_vertices: [nb, nf, 9]
        # textures: [nb, nf, 9]

        func_dist_map = {'hard': 0, 'barycentric': 1, 'euclidean': 2}
        func_rgb_map = {'hard': 0, 'softmax': 1}
        func_alpha_map = {'hard': 0, 'sum': 1, 'prod': 2}
        func_map_sample = {'surface': 0, 'vertex': 1}

        ctx.image_size = image_size
        ctx.background_color = background_color
        ctx.near = near
        ctx.far = far
        ctx.eps = eps
        ctx.sigma_val = sigma_val
        ctx.gamma_val = gamma_val
        ctx.func_dist_type = func_dist_map[dist_func]
        ctx.dist_eps = np.log(1. / dist_eps - 1.)
        ctx.func_rgb_type = func_rgb_map[aggr_func_rgb]
        ctx.func_alpha_type = func_alpha_map[aggr_func_alpha]
        ctx.texture_type = func_map_sample[texture_type]
        ctx.fill_back = fill_back

        face_vertices = face_vertices.clone()
        textures = textures.clone()

        ctx.device = face_vertices.device
        ctx.batch_size, ctx.num_faces = face_vertices.shape[:2]

        faces_info = torch.FloatTensor(ctx.batch_size, ctx.num_faces, 9*3).fill_(0.0).to(device=ctx.device) # [inv*9, sym*9, obt*3, 0*6]
        aggrs_info = torch.FloatTensor(ctx.batch_size, 2, ctx.image_size, ctx.image_size).fill_(0.0).to(device=ctx.device) 

        soft_colors = torch.FloatTensor(ctx.batch_size, 4, ctx.image_size, ctx.image_size).fill_(1.0).to(device=ctx.device) 
        soft_colors[:, 0, :, :] *= background_color[0]
        soft_colors[:, 1, :, :] *= background_color[1]
        soft_colors[:, 2, :, :] *= background_color[2]

        faces_info, aggrs_info, soft_colors = \
            soft_rasterize_cuda.forward_soft_rasterize(face_vertices, textures,
                                                       faces_info, aggrs_info,
                                                       soft_colors,
                                                       image_size, near, far, eps,
                                                       sigma_val, ctx.func_dist_type, ctx.dist_eps,
                                                       gamma_val, ctx.func_rgb_type, ctx.func_alpha_type,
                                                       ctx.texture_type, fill_back)

        ctx.save_for_backward(face_vertices, textures, soft_colors, faces_info, aggrs_info)
        return soft_colors # return d^2

    @staticmethod
    def backward(ctx, grad_soft_colors):

        face_vertices, textures, soft_colors, faces_info, aggrs_info = ctx.saved_tensors
        image_size = ctx.image_size
        background_color = ctx.background_color
        near = ctx.near
        far = ctx.far
        eps = ctx.eps
        sigma_val = ctx.sigma_val
        dist_eps = ctx.dist_eps
        gamma_val = ctx.gamma_val
        func_dist_type = ctx.func_dist_type
        func_rgb_type = ctx.func_rgb_type
        func_alpha_type = ctx.func_alpha_type
        texture_type = ctx.texture_type
        fill_back = ctx.fill_back

        grad_faces = torch.zeros_like(face_vertices, dtype=torch.float32).to(ctx.device).contiguous()
        grad_textures = torch.zeros_like(textures, dtype=torch.float32).to(ctx.device).contiguous()
        grad_soft_colors = grad_soft_colors.contiguous()

        grad_faces, grad_textures = \
            soft_rasterize_cuda.backward_soft_rasterize(face_vertices, textures, soft_colors, 
                                                        faces_info, aggrs_info,
                                                        grad_faces, grad_textures, grad_soft_colors, 
                                                        image_size, near, far, eps,
                                                        sigma_val, func_dist_type, dist_eps,
                                                        gamma_val, func_rgb_type, func_alpha_type,
                                                        texture_type, fill_back)

        return grad_faces, grad_textures, None, None, None, None, None, None, None, None, None, None, None, None, None


def soft_rasterize(face_vertices, textures, image_size=256, 
                   background_color=[0, 0, 0], near=1, far=100, 
                   fill_back=True, eps=1e-3,
                   sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                   gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                   texture_type='surface'):
    if face_vertices.device == "cpu":
        raise TypeError('Rasterize module supports only cuda Tensors')

    return SoftRasterizeFunction.apply(face_vertices, textures, image_size, 
                                       background_color, near, far,
                                       fill_back, eps,
                                       sigma_val, dist_func, dist_eps,
                                       gamma_val, aggr_func_rgb, aggr_func_alpha, 
                                       texture_type)

