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

from PIL import Image
from kaolin.graphics import DIBRenderer as Renderer
from kaolin.graphics.dib_renderer.utils.sphericalcoord import get_spherical_coords_x
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
    parser.add_argument('--use_texture', action='store_true',
                        help='Whether to render a textured mesh')
    parser.add_argument('--texture', type=str, default=os.path.join(ROOT_DIR, 'texture.png'),
                        help='Specifies path to the texture to be used')
    parser.add_argument('--output_path', type=str, default=os.path.join(ROOT_DIR, 'results'),
                        help='Path to the output directory')

    return parser.parse_args()


def main():
    args = parse_arguments()

    ###########################
    # Load mesh
    ###########################

    mesh = TriangleMesh.from_obj(args.mesh)
    vertices = mesh.vertices.cuda()
    faces = mesh.faces.int().cuda()

    # Expand such that batch size = 1

    vertices = vertices.unsqueeze(0)

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

    if not args.use_texture:
        vert_min = torch.min(vertices, dim=1, keepdims=True)[0]
        vert_max = torch.max(vertices, dim=1, keepdims=True)[0]
        colors = (vertices - vert_min) / (vert_max - vert_min)

    ###########################
    # Generate texture mapping
    ###########################

    if args.use_texture:
        uv = get_spherical_coords_x(vertices[0].cpu().numpy())
        uv = torch.from_numpy(uv).cuda()

        # Expand such that batch size = 1
        uv = uv.unsqueeze(0)

    ###########################
    # Load texture
    ###########################

    if args.use_texture:
        # Load image as numpy array
        texture = np.array(Image.open(args.texture))

        # Convert numpy array to PyTorch tensor
        texture = torch.from_numpy(texture).cuda()

        # Convert from [0, 255] to [0, 1]
        texture = texture.float() / 255.0

        # Convert to NxCxHxW layout
        texture = texture.permute(2, 0, 1).unsqueeze(0)

    ###########################
    # Render
    ###########################

    if args.use_texture:
        renderer_mode = 'Lambertian'

    else:
        renderer_mode = 'VertexColor'

    renderer = Renderer(HEIGHT, WIDTH, mode=renderer_mode)

    loop = tqdm.tqdm(list(range(0, 360, 4)))
    loop.set_description('Drawing')

    os.makedirs(args.output_path, exist_ok=True)
    writer = imageio.get_writer(os.path.join(args.output_path, 'example.gif'), mode='I')
    for azimuth in loop:
        renderer.set_look_at_parameters([90 - azimuth],
                                        [CAMERA_ELEVATION],
                                        [CAMERA_DISTANCE])

        if args.use_texture:
            predictions, _, _ = renderer(points=[vertices, faces.long()],
                                         uv_bxpx2=uv,
                                         texture_bx3xthxtw=texture)

        else:
            predictions, _, _ = renderer(points=[vertices, faces.long()],
                                         colors_bxpx3=colors)

        image = predictions.detach().cpu().numpy()[0]
        writer.append_data((image * 255).astype(np.uint8))

    writer.close()


if __name__ == '__main__':
    main()
