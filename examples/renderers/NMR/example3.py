# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

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

###########################
# Settings
###########################

CAMERA_DISTANCE = 2.732
ELEVATION = 0
TEXTURE_SIZE = 4
NUM_EPOCHS = 300


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='NMR Example 3: Optimize texture')

    parser.add_argument('--mesh', type=str, default=os.path.join(ROOT_DIR, 'rocket.obj'),
                        help='Path to the mesh OBJ file')
    parser.add_argument('--image', type=str, default=os.path.join(ROOT_DIR, 'example3_ref.png'),
                        help='Path to the target image file to optimize to')
    parser.add_argument('--output_path', type=str, default=os.path.join(ROOT_DIR, 'results'),
                        help='Path to the output directory')

    return parser.parse_args()


class Model(nn.Module):

    def __init__(self, mesh_path, image_path):
        super(Model, self).__init__()

        ###########################
        # Load mesh
        ###########################

        mesh = TriangleMesh.from_obj(mesh_path)
        # Normalize into unit cube, and expand such that batch size = 1
        vertices = normalize_vertices(mesh.vertices).unsqueeze(0)
        faces = mesh.faces.unsqueeze(0)

        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)

        ###########################
        # Initialize texture (NMR format)
        ###########################

        textures = torch.zeros(
            1, self.faces.shape[1], TEXTURE_SIZE, TEXTURE_SIZE, TEXTURE_SIZE,
            3, dtype=torch.float32
        )
        self.textures = nn.Parameter(textures)

        ###########################
        # Load target image
        ###########################

        image_ref = torch.from_numpy(imread(image_path).astype(
            'float32') / 255.).permute(2, 0, 1)[:3, ...][None, ::]
        self.register_buffer('image_ref', image_ref)

        ###########################
        # Setup renderer
        ###########################

        renderer = Renderer(camera_mode='look_at')
        # renderer.perspective = False
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        self.renderer = renderer

    def forward(self):
        ###########################
        # Render
        ###########################

        self.renderer.eye = get_points_from_angles(
            CAMERA_DISTANCE, ELEVATION,
            np.random.uniform(0, 360)
        )
        image, _, _ = self.renderer(
            self.vertices,
            self.faces,
            torch.tanh(self.textures)
        )
        loss = torch.sum((image - self.image_ref) ** 2)

        return loss


def main():
    args = parse_arguments()

    ###########################
    # Setup model
    ###########################

    model = Model(args.mesh, args.image)
    model.cuda()

    ###########################
    # Optimize
    ###########################

    loop = tqdm.tqdm(range(NUM_EPOCHS))
    loop.set_description('Optimizing')

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.1, betas=(0.5, 0.999)
    )

    azimuth = 0.0

    os.makedirs(args.output_path, exist_ok=True)
    writer = imageio.get_writer(os.path.join(
        args.output_path, 'example3_optimization.gif'), mode='I')
    for i in loop:
        optimizer.zero_grad()

        loss = model()

        loss.backward()
        optimizer.step()

        model.renderer.eye = get_points_from_angles(
            CAMERA_DISTANCE, ELEVATION, azimuth)

        images, _, _ = model.renderer(
            model.vertices,
            model.faces,
            torch.tanh(model.textures)
        )

        image = images.detach().cpu().numpy()[0].transpose(
            (1, 2, 0))
        writer.append_data((255 * image).astype(np.uint8))

        azimuth = (azimuth + 4) % 360

    writer.close()

    ###########################
    # Render optimized mesh
    ###########################

    loop = tqdm.tqdm(range(0, 360, 4))
    loop.set_description('Drawing')

    os.makedirs(args.output_path, exist_ok=True)
    writer = imageio.get_writer(os.path.join(
        args.output_path, 'example3_mesh.gif'), mode='I')
    for azimuth in loop:
        model.renderer.eye = get_points_from_angles(
            CAMERA_DISTANCE, ELEVATION, azimuth)

        images, _, _ = model.renderer(
            model.vertices,
            model.faces,
            torch.tanh(model.textures)
        )

        image = images.detach().cpu().numpy()[0].transpose(
            (1, 2, 0))
        writer.append_data((255 * image).astype(np.uint8))

    writer.close()


if __name__ == '__main__':
    main()
