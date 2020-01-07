# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import print_function
from __future__ import division

from ....mathutils.geometry.transformations import compute_camera_params
from ..utils import perspectiveprojectionnp
from .phongrender import PhongRender
from .shrender import SHRender
from .texrender import TexRender as Lambertian
from .vcrender import VCRender
import numpy as np
import torch
import torch.nn as nn


renderers = {'VertexColor': VCRender, 'Lambertian': Lambertian, 'SphericalHarmonics': SHRender, 'Phong': PhongRender}


class Renderer(nn.Module):

    def __init__(self, height, width, mode='VertexColor', camera_center=None,
                 camera_up=None, camera_fov_y=None):
        super(Renderer, self).__init__()
        assert mode in renderers, "Passed mode {0} must in in list of accepted modes: {1}".format(mode, renderers)
        self.mode = mode
        self.renderer = renderers[mode](height, width)
        if camera_center is None:
            self.camera_center = np.array([0, 0, 0], dtype=np.float32)
        if camera_up is None:
            self.camera_up = np.array([0, 1, 0], dtype=np.float32)
        if camera_fov_y is None:
            self.camera_fov_y = 49.13434207744484 * np.pi / 180.0
        self.camera_params = None

    def forward(self, points, *args, **kwargs):

        if self.camera_params is None:
            print('Camera parameters have not been set, default perspective parameters of distance = 1, elevation = 30, azimuth = 0 are being used')
            self.set_look_at_parameters([0], [30], [1])

        assert self.camera_params[0].shape[0] == points[0].shape[0], "Set camera parameters batch size must equal batch size of passed points"

        return self.renderer(points, self.camera_params, *args, **kwargs)

    def set_look_at_parameters(self, azimuth, elevation, distance):

        camera_projection_mtx = perspectiveprojectionnp(self.camera_fov_y, 1.0)
        camera_projection_mtx = torch.FloatTensor(camera_projection_mtx).cuda()

        camera_view_mtx = []
        camera_view_shift = []
        for a, e, d in zip(azimuth, elevation, distance):
            mat, pos = compute_camera_params(a, e, d)
            camera_view_mtx.append(mat)
            camera_view_shift.append(pos)
        camera_view_mtx = torch.stack(camera_view_mtx).cuda()
        camera_view_shift = torch.stack(camera_view_shift).cuda()

        self.camera_params = [camera_view_mtx, camera_view_shift, camera_projection_mtx]

    def set_camera_parameters(self, parameters):
        self.camera_params = parameters
