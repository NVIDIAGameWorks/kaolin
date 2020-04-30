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
# A PyTorch implementation of Neural 3D Mesh Renderer
#
# Copyright (c) 2017 Hiroharu Kato
# Copyright (c) 2018 Nikos Kolotouros
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

from __future__ import division

import math
import torch
import torch.nn as nn
import numpy

from .nmr import util
from .nmr import rasterizer


class NeuralMeshRenderer(nn.Module):
    def __init__(self,
                 image_size=256,
                 anti_aliasing=True,
                 background_color=None,
                 fill_back=True,
                 camera_mode='projection',
                 K=None,
                 R=None,
                 t=None,
                 dist_coeffs=None,
                 orig_size=1024,
                 perspective=True,
                 viewing_angle=30,
                 camera_direction=None,
                 near=0.1,
                 far=100,
                 light_intensity_ambient=0.5,
                 light_intensity_directional=0.5,
                 light_color_ambient=None,
                 light_color_directional=None,
                 light_direction=None):
        """Initialize the NeuralMeshRenderer.

        NOTE: NeuralMeshRenderer works only in GPU mode!

        Args:
            image_size (int): Size of the (square) image to be rendered.
            anti_aliasing (bool): Whether or not to perform anti-aliasing
                (default: True)
            background_color (torch.Tensor): Background color of rendered image
                (size: math:`3`, default: :math:`\left[0, 0, 0\right]`)
            fill_back (bool): Whether or not to fill color to the back
                side of each triangle as well (sometimes helps, when
                the triangles in the mesh are not properly oriented.)
                (default: True)
            camera_mode (str): Choose from among `projection`, `look`, and
                `look_at`. In the `projection` mode, the camera is at the
                origin, and its optical axis is aligned with the positive
                Z-axis. In the `look_at` mode, the object (not the camera)
                is placed at the origin. The camera "looks at" the object
                from a predefined "eye" location, which is computed from
                the `viewing_angle` (another input to this function). In
                the `look` mode, only the direction in which the camera
                needs to look is specified. It does not necessarily look
                towards the origin, as it allows the specification of a
                custom "upwards" direction (default: 'projection').
            K (torch.Tensor): Camera intrinsics matrix. Note that, unlike
                standard notation, K here is a 4 x 4 matrix (with the last
                row and last column drawn from the 4 x 4 identity matrix)
                (default: None)
            R (torch.Tensor): Rotation matrix (again, 4 x 4, as opposed
                to the usual 3 x 3 convention).
            t (torch.Tensor): Translation vector (3 x 1). Note that the
                (negative of the) tranlation is applied before rotation,
                to be consistent with the projective geometry convention
                of transforming a 3D point X by doing
                torch.matmul(R.transpose(), X - t) (default: None)
            viewing_angle (float): Angle at which the object is to be viewed
                (assumed to be in degrees!) (default: 30.)
            camera_direction (float): Direction in which the camera is facing
                (used only in the `look` and `look_at` modes) (default:
                :math:`[0, 0, 1]`)
            near (float): Near clipping plane (for depth values) (default: 0.1)
            far (float): Far clipping plane (for depth values) (default: 100)
            light_intensity_ambient (float): Intensity of ambient light (in the
                range :math:`\left[ 0, 1 \right]`) (default: 0.5).
            light_intensity_directional (float): Intensity of directional light
                (in the range :math:`\left[ 0, 1 \right]`) (default: 0.5).
            light_color_ambient (torch.Tensor): Color of ambient light
                (default: :math:`\left[ 1, 1, 1 \right]`)
            light_color_directional (torch.Tensor): Color of directional light
                (default: :math:`\left[ 1, 1, 1 \right]`)
            light_direction (torch.Tensor): Light direction, for directional
                light (default: :math:`\left[ 0, 1, 0 \right]`)
        """
        super(NeuralMeshRenderer, self).__init__()

        # default arguments
        if background_color is None:
            background_color = [0, 0, 0]

        if camera_direction is None:
            camera_direction = [0, 0, 1]

        if light_color_ambient is None:
            light_color_ambient = [1, 1, 1]

        if light_color_directional is None:
            light_color_directional = [1, 1, 1]

        if light_direction is None:
            light_direction = [0, 1, 0]

        # rendering
        self.image_size = image_size
        self.anti_aliasing = anti_aliasing
        self.background_color = background_color
        self.fill_back = fill_back

        # camera
        self.camera_mode = camera_mode
        if self.camera_mode == 'projection':
            self.K = K
            self.R = R
            self.t = t
            if isinstance(self.K, numpy.ndarray):
                self.K = torch.cuda.FloatTensor(self.K)
            if isinstance(self.R, numpy.ndarray):
                self.R = torch.cuda.FloatTensor(self.R)
            if isinstance(self.t, numpy.ndarray):
                self.t = torch.cuda.FloatTensor(self.t)
            self.dist_coeffs = dist_coeffs
            if dist_coeffs is None:
                self.dist_coeffs = torch.zeros(1, 5, device='cuda')
            self.orig_size = orig_size
        elif self.camera_mode in ['look', 'look_at']:
            self.perspective = perspective
            self.viewing_angle = viewing_angle
            self.eye = [
                0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]
            self.camera_direction = camera_direction
        else:
            raise ValueError(
                'Camera mode has to be one of projection, look or look_at')

        self.near = near
        self.far = far

        # light
        self.light_intensity_ambient = light_intensity_ambient
        self.light_intensity_directional = light_intensity_directional
        self.light_color_ambient = light_color_ambient
        self.light_color_directional = light_color_directional
        self.light_direction = light_direction

        # rasterization
        self.rasterizer_eps = 1e-3

    def forward(self, vertices, faces, textures=None, mode=None, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        '''
        Implementation of forward rendering method
        The old API is preserved for back-compatibility with the Chainer implementation
        '''

        if mode is None:
            return self.render(vertices, faces, textures, K, R, t, dist_coeffs, orig_size)
        elif mode is 'rgb':
            return self.render_rgb(vertices, faces, textures, K, R, t, dist_coeffs, orig_size)
        elif mode == 'silhouettes':
            return self.render_silhouettes(vertices, faces, K, R, t, dist_coeffs, orig_size)
        elif mode == 'depth':
            return self.render_depth(vertices, faces, K, R, t, dist_coeffs, orig_size)
        else:
            raise ValueError(
                "mode should be one of None, 'silhouettes' or 'depth'")

    def render_silhouettes(self, vertices, faces, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):

        # fill back
        if self.fill_back:
            faces = torch.cat(
                (faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = util.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = util.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = util.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = util.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = util.projection(
                vertices, K, R, t, dist_coeffs, orig_size)

        # rasterization
        faces = util.vertices_to_faces(vertices, faces)
        images = rasterizer.rasterize_silhouettes(
            faces, self.image_size, self.anti_aliasing)
        return images

    def render_depth(self, vertices, faces, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):

        # fill back
        if self.fill_back:
            faces = torch.cat(
                (faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = util.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = util.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = util.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = util.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = util.projection(
                vertices, K, R, t, dist_coeffs, orig_size)

        # rasterization
        faces = util.vertices_to_faces(vertices, faces)
        images = rasterizer.rasterize_depth(
            faces, self.image_size, self.anti_aliasing)
        return images

    def render_rgb(self, vertices, faces, textures, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        # fill back
        if self.fill_back:
            faces = torch.cat(
                (faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            textures = torch.cat(
                (textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # lighting
        faces_lighting = util.vertices_to_faces(vertices, faces)
        textures = util.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = util.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = util.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = util.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = util.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = util.projection(
                vertices, K, R, t, dist_coeffs, orig_size)

        # rasterization
        faces = util.vertices_to_faces(vertices, faces)
        images = rasterizer.rasterize(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return images

    def render(self, vertices, faces, textures, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        """Renders the RGB, depth, and alpha channels.

        Args:
            vertices (torch.Tensor): Vertices of the mesh (shape: :math:`B
                \times V \times 3`), where :math:`B` is the batchsize,
                and :math:`V` is the number of vertices in the mesh.
            faces (torch.Tensor): Faces of the mesh (shape: :math:`B \times
                F \times 3`), where :math:`B` is the batchsize, and :math:`F`
                is the number of faces in the mesh.
            textures (torch.Tensor): Mesh texture (shape: :math:`B \times F
                \times 4 \times 4 \times 4 \times 3`)
            K (torch.Tensor): Camera intrinsics (default: None) (shape:
                :math:`B \times 4 \times 4` or :math:`4 \times 4`)
            R (torch.Tensor): Rotation matrix (default: None) (shape:
                :math:`B \times 4 \times 4` or :math:`4 \times 4`)
            t (torch.Tensor): Translation vector (default: None)
                (shape: :math:`B \times 3` or :math:`3`)
        """
        # fill back
        if self.fill_back:
            faces = torch.cat(
                (faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            textures = torch.cat(
                (textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # lighting
        faces_lighting = util.vertices_to_faces(vertices, faces)
        textures = util.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = util.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = util.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = util.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = util.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = util.projection(
                vertices, K, R, t, dist_coeffs, orig_size)

        # rasterization
        faces = util.vertices_to_faces(vertices, faces)
        out = rasterizer.rasterize_rgbad(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return out['rgb'], out['depth'], out['alpha']
