# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Neural Mesh Renderer

# MIT License

# Copyright (c) 2017 Hiroharu Kato
# Copyright (c) 2018 Nikos Kolotouros
# A PyTorch implementation of Neural 3D Mesh Renderer (https://github.com/hiroharu-kato/neural_renderer)

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

from __future__ import division

from kaolin.graphics import NeuralMeshRenderer as Renderer
from kaolin.graphics.nmr.util import get_points_from_angles
from kaolin.rep import TriangleMesh
from skimage.io import imread
from util import normalize_vertices
import argparse
import imageio
import numpy as np
import os
import torch
import torch.nn as nn
import tqdm

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='NMR Example 2: Optimize vertices')

    parser.add_argument('--mesh', type=str, default=os.path.join(ROOT_DIR, 'rocket.obj'),
                        help='Path to the mesh OBJ file')
    parser.add_argument('--image', type=str, default=os.path.join(ROOT_DIR, 'example2_ref.png'),
                        help='Path to the target image file to optimize to')
    parser.add_argument('--output_path', type=str, default=os.path.join(ROOT_DIR, 'results'),
                        help='Path to the output directory')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to optimize')
    parser.add_argument('--camera_distance', type=float, default=2.732,
                        help='Distance from camera to object center')
    parser.add_argument('--elevation', type=float, default=0,
                        help='Camera elevation')
    parser.add_argument('--azimuth', type=float, default=90,
                        help='Camera azimuth')
    parser.add_argument('--texture_size', type=int, default=2,
                        help='Dimension of texture')

    return parser.parse_args()


class Model(nn.Module):

    def __init__(self, mesh_path, image_path, args):
        super(Model, self).__init__()

        self.args = args

        ###########################
        # Load mesh
        ###########################

        mesh = TriangleMesh.from_obj(mesh_path)
        mesh.cuda()
        # Normalize into unit cube, and expand such that batch size = 1
        vertices = normalize_vertices(mesh.vertices).unsqueeze(0)
        faces = mesh.faces.unsqueeze(0)

        self.vertices = nn.Parameter(vertices)
        self.register_buffer('faces', faces)

        ###########################
        # Generate texture (NMR format)
        ###########################

        textures = torch.ones(
            1, self.faces.shape[1], self.args.texture_size, self.args.texture_size, self.args.texture_size,
            3, dtype=torch.float32,
            device='cuda'
        )
        self.register_buffer('textures', textures)

        ###########################
        # Load target image
        ###########################

        image_ref = torch.from_numpy(imread(image_path).astype(
            np.float32).mean(-1) / 255.)[None, ::]
        self.register_buffer('image_ref', image_ref)

        ###########################
        # Setup renderer
        ###########################

        renderer = Renderer(camera_mode='look_at')
        self.renderer = renderer

    def forward(self):
        ###########################
        # Render
        ###########################

        self.renderer.eye = get_points_from_angles(
            self.args.camera_distance, self.args.elevation, self.args.azimuth)
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        loss = torch.sum((image - self.image_ref[None, :, :]) ** 2)

        return loss


def main():
    args = parse_arguments()

    ###########################
    # Setup model
    ###########################

    model = Model(args.mesh, args.image, args)
    model.cuda()

    ###########################
    # Optimize
    ###########################

    loop = tqdm.tqdm(range(args.epochs))
    loop.set_description('Optimizing')

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad])

    os.makedirs(args.output_path, exist_ok=True)
    writer = imageio.get_writer(os.path.join(
        args.output_path, 'example2_optimization.gif'), mode='I')
    for i in loop:
        optimizer.zero_grad()

        loss = model()

        loss.backward()
        optimizer.step()

        images, _, _ = model.renderer(
            model.vertices, model.faces, model.textures)

        image = images.detach()[0].permute(1, 2, 0).cpu().numpy()
        writer.append_data((255 * image).astype(np.uint8))

    writer.close()

    ###########################
    # Render optimized mesh
    ###########################

    loop = tqdm.tqdm(range(0, 360, 4))
    loop.set_description('Drawing')

    os.makedirs(args.output_path, exist_ok=True)
    writer = imageio.get_writer(os.path.join(
        args.output_path, 'example2_mesh.gif'), mode='I')
    for azimuth in loop:
        model.renderer.eye = get_points_from_angles(
            args.camera_distance, args.elevation, args.azimuth)

        images, _, _ = model.renderer(
            model.vertices, model.faces, model.textures)

        image = images.detach()[0].permute(1, 2, 0).cpu().numpy()
        writer.append_data((255 * image).astype(np.uint8))

    writer.close()


if __name__ == '__main__':
    main()
