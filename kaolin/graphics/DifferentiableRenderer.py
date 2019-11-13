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

from abc import abstractmethod

import torch


class DifferentiableRenderer(torch.nn.Module):
    r""" Base class for differentiable renderers. All major components
    of the graphics processing pipeline have been instantiated as
    abstract methods.

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

    @abstractmethod
    def __init__(self):
        super(DifferentiableRenderer, self).__init__()

    @abstractmethod
    def forward(self, vertices, faces, textures=None):
        # Generally, this method will call the other
        # (already implemented) methods.
        # self.lighting()
        # self.shading()
        # self.transform_to_camera_frame()
        # self.project_to_image()
        # self.rasterize()
        raise NotImplementedError

    @abstractmethod
    def lighting(self):
        raise NotImplementedError

    @abstractmethod
    def shading(self):
        raise NotImplementedError

    @abstractmethod
    def transform_to_camera_frame(self):
        raise NotImplementedError

    @abstractmethod
    def project_to_image(self):
        raise NotImplementedError

    @abstractmethod
    def rasterize(self):
        raise NotImplementedError
