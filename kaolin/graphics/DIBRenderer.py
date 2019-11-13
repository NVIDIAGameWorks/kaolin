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
# DIB-R
# 
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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


class DIBRenderer(torch.nn.Module):
    r"""Placeholder class for DIB-Renderer implementation.

    A differentiable renderer takes in vertex geometry, faces (usually
    triangles), and optionally texture (and any other information
    deemed relevant), and implements the following main steps.
        * Lighting (ambient and optionally, directional lighting, i.e.,
          diffuse/Lambertian reflection, and optionally specular reflection)
        * (Vertex) Shading (Gouraud/Phong/etc.)
        * Geometric transformation (usually, this is needed to view the
          object to be rendered from a specific viewing angle, etc.)
          (the geometric transformations are usually applied AFTER lighting,
          as it is easier to perform lighting in the object frame, rather
          than at the scene level)
        * Projection (perspective/orthographic/weak-orthographic(affine))
          from 3D (camera) coordinates to 2D (image) coordinates.
        * Rasterization to convert the primitives (usually triangles) to
          pixels, and determine texture (when available).

    """

    def __init__(self):
        super(DifferentiableRenderer, self).__init__()

    def forward(self, vertices, faces, textures=None):
        # Generally, this method will call the other
        # (already implemented) methods.
        # self.lighting()
        # self.shading()
        # self.transform_to_camera_frame()
        # self.project_to_image()
        # self.rasterize()
        raise NotImplementedError

    def lighting(self):
        raise NotImplementedError

    def shading(self):
        raise NotImplementedError

    def transform_to_camera_frame(self):
        raise NotImplementedError

    def project_to_image(self):
        raise NotImplementedError

    def rasterize(self):
        raise NotImplementedError
