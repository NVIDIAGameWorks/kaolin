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

import os

import imageio
import numpy as np
import torch
import tqdm

import kaolin

# Example script that uses SoftRas to render an image, given a mesh input

if __name__ == "__main__":

    # Initialize the soft rasterizer.
    renderer = kaolin.graphics.SoftRenderer(camera_mode="look_at", device="cuda:0")

    # Camera settings.
    camera_distance = 2.  # Distance of the camera from the origin (i.e., center of the object).
    elevation = 30.       # Angle of elevation

    # Infer the base path of the kaolin repo
    KAOLIN_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

    # Read in the input mesh.
    infile = os.path.join(KAOLIN_ROOT, "tests", "graphics", "banana.obj")
    mesh = kaolin.rep.TriangleMesh.from_obj(infile)

    # Output filename (to write out a rendered .gif to).
    outfile = os.path.join(KAOLIN_ROOT, "examples", "renderers", "softras_render.gif")

    # Extract the vertices, faces, and texture the mesh (currently color with white).
    vertices = mesh.vertices.float()
    faces = mesh.faces.long()
    face_textures = faces.clone()
    vertices = vertices[None, :, :].cuda()
    faces = faces[None, :, :].cuda()
    face_textures = face_textures[None, :, :].cuda()
    # Initialize all faces to yellow (to color the banana)!
    textures = torch.cat(
        (
            torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device="cuda:0"),
            torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device="cuda:0"),
            torch.zeros(1, faces.shape[1], 2, 1, dtype=torch.float32, device="cuda:0"),
        ),
        dim=-1,
    )

    # Normalize vertices
    vertices_max = vertices.max()
    vertices_min = vertices.min()
    vertices_middle = (vertices_max + vertices_min) / 2.
    vertices = vertices - vertices_middle
    # Scale the vertices slightly (so that they occupy a sizeable image area).
    # Skip if using models other than the banana.obj file.
    coef = 5
    vertices = vertices * coef

    # Loop over a set of azimuth angles, and render the image.
    loop = loop = tqdm.tqdm(list(range(0, 360, 6)))
    loop.set_description("Rendering using softras...")
    writer = imageio.get_writer(outfile, mode="I")
    for azimuth in loop:
        renderer.set_eye_from_angles(camera_distance, elevation, azimuth)
        # Render an image.
        rgba = renderer.forward(vertices, faces, textures)
        img = rgba[0].permute(1, 2, 0).detach().cpu().numpy()
        writer.append_data((255 * img).astype(np.uint8))
    writer.close()