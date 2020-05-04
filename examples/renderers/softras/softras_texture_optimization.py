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

import argparse
import os

import imageio
import numpy as np
import torch
from tqdm import tqdm, trange

import kaolin

# Example script that uses SoftRas to optimize the texture for a given mesh.


class Model(torch.nn.Module):
    """Wrap textures into an nn.Module, for optimization. """

    def __init__(self, textures):
        super(Model, self).__init__()
        self.textures = torch.nn.Parameter(textures)

    def forward(self):
        return torch.sigmoid(self.textures)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20,
                        help="Number of iterations to run optimization for.")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization steps.")
    args = parser.parse_args()

    # Initialize the soft rasterizer.
    renderer = kaolin.graphics.SoftRenderer(camera_mode="look_at", device="cuda:0")

    # Camera settings.
    camera_distance = 2.  # Distance of the camera from the origin (i.e., center of the object)
    elevation = 30.       # Angle of elevation
    azimuth = 0.          # Azimuth angle

    # Directory in which sample data is located.
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "sampledata")

    # Read in the input mesh. TODO: Add filepath as argument.
    mesh = kaolin.rep.TriangleMesh.from_obj(os.path.join(DATA_DIR, "banana.obj"))

    # Output filename to write out a rendered .gif to, showing the progress of optimization.
    progressfile = "texture_optimization_progress.gif"
    # Output filename to write out a rendered .gif file to, rendering the optimized mesh.
    outfile = "texture_optimization_output.gif"

    # Extract the vertices, faces, and texture the mesh (currently color with white).
    vertices = mesh.vertices
    faces = mesh.faces
    vertices = vertices[None, :, :].cuda()
    faces = faces[None, :, :].cuda()
    textures = torch.ones(1, faces.shape[1], 2, 3, dtype=torch.float32, device="cuda:0")

    # Translate the mesh such that its centered at the origin.
    vertices_max = vertices.max()
    vertices_min = vertices.min()
    vertices_middle = (vertices_max + vertices_min) / 2.
    vertices = vertices - vertices_middle
    # Scale the vertices slightly (so that they occupy a sizeable image area).
    # Skip if using models other than the banana.obj file.
    coef = 5
    vertices = vertices * coef

    img_target = torch.from_numpy(
        imageio.imread(os.path.join(DATA_DIR, "banana.png")).astype(np.float32) / 255
    ).cuda()
    img_target = img_target[None, ...].permute(0, 3, 1, 2)

    # Create a 'model' (an nn.Module) that wraps around the vertices, making it 'optimizable'.
    # TODO: Replace with a torch optimizer that takes vertices as a 'params' argument.
    # Deform the vertices slightly.
    model = Model(textures).cuda()
    # renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
    optimizer = torch.optim.Adam(model.parameters(), 1., betas=(0.5, 0.99))
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)
    mseloss = torch.nn.MSELoss()

    # Perform texture optimization.
    if not args.no_viz:
        writer = imageio.get_writer(progressfile, mode="I")
    for i in trange(args.iters):
        optimizer.zero_grad()
        textures = model()
        rgba = renderer(vertices, faces, textures)
        loss = mseloss(rgba, img_target)
        loss.backward()
        optimizer.step()
        if i % 5 == 0:
            # TODO: Add functionality to write to gif output file.
            tqdm.write(f"Loss: {loss.item():.5}")
            if not args.no_viz:
                img = rgba[0].permute(1, 2, 0).detach().cpu().numpy()
                writer.append_data((255 * img).astype(np.uint8))
    if not args.no_viz:
        writer.close()

        # Write optimized mesh to output file.
        writer = imageio.get_writer(outfile, mode="I")
        for azimuth in trange(0, 360, 6):
            renderer.set_eye_from_angles(camera_distance, elevation, azimuth)
            rgba = renderer.forward(vertices, faces, model())
            img = rgba[0].permute(1, 2, 0).detach().cpu().numpy()
            writer.append_data((255 * img).astype(np.uint8))
        writer.close()
