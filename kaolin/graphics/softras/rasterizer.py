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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import soft_renderer.functional as srf


class SoftRasterizer(nn.Module):
    def __init__(self, image_size=256, background_color=[0, 0, 0], near=1, far=100, 
                 anti_aliasing=False, fill_back=False, eps=1e-3, 
                 sigma_val=1e-5, dist_func='euclidean', dist_eps=1e-4,
                 gamma_val=1e-4, aggr_func_rgb='softmax', aggr_func_alpha='prod',
                 texture_type='surface'):
        super(SoftRasterizer, self).__init__()

        if dist_func not in ['hard', 'euclidean', 'barycentric']:
            raise ValueError('Distance function only support hard, euclidean and barycentric')
        if aggr_func_rgb not in ['hard', 'softmax']:
            raise ValueError('Aggregate function(rgb) only support hard and softmax')
        if aggr_func_alpha not in ['hard', 'prod', 'sum']:
            raise ValueError('Aggregate function(a) only support hard, prod and sum')
        if texture_type not in ['surface', 'vertex']:
            raise ValueError('Texture type only support surface and vertex')

        self.image_size = image_size
        self.background_color = background_color
        self.near = near
        self.far = far
        self.anti_aliasing = anti_aliasing
        self.eps = eps
        self.fill_back = fill_back
        self.sigma_val = sigma_val
        self.dist_func = dist_func
        self.dist_eps = dist_eps
        self.gamma_val = gamma_val
        self.aggr_func_rgb = aggr_func_rgb
        self.aggr_func_alpha = aggr_func_alpha
        self.texture_type = texture_type

    def forward(self, mesh, mode=None):
        image_size = self.image_size * (2 if self.anti_aliasing else 1)

        images = srf.soft_rasterize(mesh.face_vertices, mesh.face_textures, image_size, 
                                    self.background_color, self.near, self.far, 
                                    self.fill_back, self.eps,
                                    self.sigma_val, self.dist_func, self.dist_eps,
                                    self.gamma_val, self.aggr_func_rgb, self.aggr_func_alpha,
                                    self.texture_type)

        if self.anti_aliasing:
            images = F.avg_pool2d(images, kernel_size=2, stride=2)

        return images