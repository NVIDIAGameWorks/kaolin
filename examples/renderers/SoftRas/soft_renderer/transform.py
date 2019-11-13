# Copyright (c) 2017 Hiroharu Kato
# Copyright (c) 2018 Nikos Kolotouros
# Copyright (c) 2019 Shichen Liu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import numpy as np
import torch
import torch.nn as nn

import soft_renderer.functional as srf

class Projection(nn.Module):
    def __init__(self, P, dist_coeffs=None, orig_size=512):
        super(Projection, self).__init__()

        self.P = P
        self.dist_coeffs = dist_coeffs
        self.orig_size = orig_size

        if isinstance(self.P, np.ndarray):
            self.P = torch.from_numpy(self.P).cuda()
        if self.P is None or P.ndimension() != 3 or self.P.shape[1] != 3 or self.P.shape[2] != 4:
            raise ValueError('You need to provide a valid (batch_size)x3x4 projection matrix')
        if dist_coeffs is None:
            self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]).repeat(P.shape[0], 1)

    def forward(self, vertices):
        vertices = srf.projection(vertices, self.P, self.dist_coeffs, self.orig_size)
        return vertices


class LookAt(nn.Module):
    def __init__(self, perspective=True, viewing_angle=30, viewing_scale=1.0, eye=None):
        super(LookAt, self).__init__()

        self.perspective = perspective
        self.viewing_angle = viewing_angle
        self.viewing_scale = viewing_scale
        self._eye = eye

        if self._eye is None:
            self._eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]

    def forward(self, vertices):
        vertices = srf.look_at(vertices, self._eye)
        # perspective transformation
        if self.perspective:
            vertices = srf.perspective(vertices, angle=self.viewing_angle)
        else:
            vertices = srf.orthogonal(vertices, scale=self.viewing_scale)
        return vertices


class Look(nn.Module):
    def __init__(self, camera_direction=[0,0,1], perspective=True, viewing_angle=30, viewing_scale=1.0, eye=None):
        super(Look, self).__init__()

        self.perspective = perspective
        self.viewing_angle = viewing_angle
        self.viewing_scale = viewing_scale
        self._eye = eye
        self.camera_direction = [0, 0, 1]

        if self._eye is None:
            self._eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]

    def forward(self, vertices):
        vertices = srf.look(vertices, self._eye, self.camera_direction)
        # perspective transformation
        if self.perspective:
            vertices = srf.perspective(vertices, angle=self.viewing_angle)
        else:
            vertices = srf.orthogonal(vertices, scale=self.viewing_scale)
        return vertices


class Transform(nn.Module):
    def __init__(self, camera_mode='projection', P=None, dist_coeffs=None, orig_size=512,
                 perspective=True, viewing_angle=30, viewing_scale=1.0,
                 eye=None, camera_direction=[0,0,1]):
        super(Transform, self).__init__()

        self.camera_mode = camera_mode
        if self.camera_mode == 'projection':
            self.transformer = Projection(P, dist_coeffs, orig_size)
        elif self.camera_mode == 'look':
            self.transformer = Look(perspective, viewing_angle, viewing_scale, eye, camera_direction)
        elif self.camera_mode == 'look_at':
            self.transformer = LookAt(perspective, viewing_angle, viewing_scale, eye)
        else:
            raise ValueError('Camera mode has to be one of projection, look or look_at')

    def forward(self, mesh):
        mesh.vertices = self.transformer(mesh.vertices)
        return mesh

    def set_eyes_from_angles(self, distances, elevations, azimuths):
        if self.camera_mode not in ['look', 'look_at']:
            raise ValueError('Projection does not need to set eyes')
        self.transformer._eye = srf.get_points_from_angles(distances, elevations, azimuths)

    def set_eyes(self, eyes):
        if self.camera_mode not in ['look', 'look_at']:
            raise ValueError('Projection does not need to set eyes')
        self.transformer._eye = eyes

    @property
    def eyes(self):
        return self.transformer._eyes
