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

import argparse
import os
import numpy as np
import torch
import tqdm
import imageio

from kaolin.graphics import NeuralMeshRenderer as Renderer
from kaolin.graphics.nmr.util import get_points_from_angles
from kaolin.rep import TriangleMesh
from util import normalize_vertices

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def parse_arguments():
    parser = argparse.ArgumentParser(description='NMR Example 1: Render mesh')

    parser.add_argument('--mesh', type=str, default=os.path.join(ROOT_DIR, 'rocket.obj'),
                        help='Path to the mesh OBJ file')
    parser.add_argument('--output_path', type=str, default=os.path.join(ROOT_DIR, 'results'),
                        help='Path to the output directory')
    parser.add_argument('--camera_distance', type=float, default=2.732,
                        help='Distance from camera to object center')
    parser.add_argument('--elevation', type=float, default=30,
                        help='Camera elevation')
    parser.add_argument('--texture_size', type=int, default=2,
                        help='Dimension of texture')

    return parser.parse_args()


def main():
    args = parse_arguments()

    ###########################
    # Load mesh
    ###########################

    mesh = TriangleMesh.from_obj(args.mesh)
    mesh.cuda()
    # Normalize into unit cube, and expand such that batch size = 1
    vertices = normalize_vertices(mesh.vertices).unsqueeze(0)
    faces = mesh.faces.unsqueeze(0)

    ###########################
    # Generate texture (NMR format)
    ###########################

    textures = torch.ones(
        1, faces.shape[1], args.texture_size, args.texture_size, args.texture_size,
        3, dtype=torch.float32,
        device='cuda'
    )

    ###########################
    # Render
    ###########################

    renderer = Renderer(camera_mode='look_at')

    loop = tqdm.tqdm(range(0, 360, 4))
    loop.set_description('Drawing')

    os.makedirs(args.output_path, exist_ok=True)
    writer = imageio.get_writer(os.path.join(
        args.output_path, 'example1.gif'), mode='I')
    for azimuth in loop:
        renderer.eye = get_points_from_angles(
            args.camera_distance, args.elevation, args.azimuth)

        images, _, _ = renderer(vertices, faces, textures)

        image = images.detach()[0].permute(1, 2, 0).cpu().numpy()  # [image_size, image_size, RGB]
        writer.append_data((255 * image).astype(np.uint8))

    writer.close()


if __name__ == '__main__':
    main()
