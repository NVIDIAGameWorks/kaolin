# Copyright (c) 2019, NEVADA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kaolin.graphics.dib_renderer.renderer import Renderer
from kaolin.rep import TriangleMesh
import argparse
import imageio
import numpy as np
import os
import torch
import tqdm

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

###########################
# Settings
###########################

CAMERA_DISTANCE = 2
CAMERA_ELEVATION = 30
MESH_SIZE = 5
HEIGHT = 256
WIDTH = 256


def parse_arguments():
    parser = argparse.ArgumentParser(description='Kaolin DIB-R Example')

    parser.add_argument('--mesh', type=str, default=os.path.join(ROOT_DIR, 'banana.obj'),
                        help='Path to the mesh OBJ file')
    parser.add_argument('--output_path', type=str, default=os.path.join(ROOT_DIR, 'results'),
                        help='Path to the output directory')

    return parser.parse_args()


def main():
    args = parse_arguments()

    ###########################
    # Load mesh
    ###########################

    mesh = TriangleMesh.from_obj(args.mesh)
    vertices = mesh.vertices
    faces = mesh.faces.int()

    # Expand such that batch size = 1

    vertices = vertices[None, :, :].cuda()
    faces = faces[None, :, :].cuda()

    ###########################
    # Normalize mesh position
    ###########################

    vertices_max = vertices.max()
    vertices_min = vertices.min()
    vertices_middle = (vertices_max + vertices_min) / 2.
    vertices = (vertices - vertices_middle) * MESH_SIZE

    ###########################
    # Generate vertex color
    ###########################

    vert_min = torch.min(vertices, dim=1, keepdims=True)[0]
    vert_max = torch.max(vertices, dim=1, keepdims=True)[0]
    colors = (vertices - vert_min) / (vert_max - vert_min)

    ###########################
    # Render
    ###########################

    renderer = Renderer(HEIGHT, WIDTH, mode='VertexColor')

    loop = tqdm.tqdm(list(range(0, 360, 4)))
    loop.set_description('Drawing')

    os.makedirs(args.output_path, exist_ok=True)
    writer = imageio.get_writer(os.path.join(args.output_path, 'example.gif'), mode='I')
    for azimuth in loop:
        renderer.set_look_at_parameters([90 - azimuth],
                                        [CAMERA_ELEVATION],
                                        [CAMERA_DISTANCE])

        predictions, _, _ = renderer(points=[vertices, faces[0].long()], colors=[colors])
        image = predictions.detach().cpu().numpy()[0]
        writer.append_data((image * 255).astype(np.uint8))

    writer.close()


if __name__ == '__main__':
    main()
