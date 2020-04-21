# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import kaolin
import matplotlib.pyplot as plt

# Example script that uses SoftRas to render an image, given a mesh input

if __name__ == "__main__":

    # Initialize the soft rasterizer.
    renderer = kaolin.graphics.SoftRenderer(camera_mode="look_at", device="cuda:0")

    # Read in the input mesh.
    filename_input = "tests/model.obj"
    mesh = kaolin.rep.TriangleMesh.from_obj(filename_input)

    # Extract the vertices, faces, and texture the mesh (currently color with white).
    vertices = mesh.vertices.float()
    faces = mesh.faces.long()
    face_textures = faces.clone()
    vertices = vertices[None, :, :].cuda()
    faces = faces[None, :, :].cuda()
    face_textures = face_textures[None, :, :].cuda()
    textures = torch.ones(1, faces.shape[1], 2, 3, dtype=torch.float32).cuda()

    # Render an image.
    rgb, d, _, = renderer.forward(vertices, faces, textures)

    # Display.
    plt.imshow(rgb[0].permute(1, 2, 0).cpu().numpy())
    plt.show()
