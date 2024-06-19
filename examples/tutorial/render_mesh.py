#!/usr/bin/env python

# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

import argparse
import logging
import os
import re
import sys
import torch

import kaolin
import kaolin.render.easy_render as easy_render

logger = logging.getLogger(__name__)


def _parse_color(in_str):
    try:
        color = [float(int(x)) / 255.0 for x in in_str.strip().split(',')]
        return (color[0], color[1], color[2])
    except Exception as e:
        return None


# TODO: add camera options
# TODO: expose as binary
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample script for rendering a mesh from a fixed viewpoint.')
    parser.add_argument('--mesh_filename', type=str, required=True,
                        help='Mesh filename in obj, usd or gltf format.')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Image resolution to render at.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory to write checkpoints to; must exist.')
    parser.add_argument('--base_name', type=str, default=None,
                        help='Base name to use; will determine automatically if not set.')
    parser.add_argument('--backend', type=str, default=None,
                        help='Backend to use for differentiable rendering, choose "nvdiffrast" or "cuda".')
    parser.add_argument('--use_default_material', type=str, default=None,
                        help='Set to 3 comma-deliminted integers of the diffuse color to use or blank to use default')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    if not os.path.isdir(args.output_dir):
        raise RuntimeError(f'Output directory does not exist: --output_dir={args.output_dir}')

    # Read the mesh
    mesh = kaolin.io.import_mesh(args.mesh_filename, triangulate=True)
    mesh = mesh.cuda()
    mesh.vertices = kaolin.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)

    if args.use_default_material is not None:
        mesh.materials = [easy_render.default_material(_parse_color(args.use_default_material)).cuda()]
        mesh.material_assignments[...] = 0

    print(mesh)

    # Create a pinhole camera
    camera = easy_render.default_camera(args.resolution).cuda()

    # Create lighting
    lighting = easy_render.default_lighting().cuda()

    # Render the mesh
    res = easy_render.render_mesh(camera, mesh, lighting=lighting, backend=args.backend)
    logger.info(kaolin.utils.testing.tensor_info(res["render"], name='rendering', print_stats=True))

    # Write out the rendering
    bname = args.base_name
    if bname is None:
        bname = re.sub(r'[^a-zA-Z0-9_]+', '_', os.path.split(args.mesh_filename)[-1]) + \
                ('' if args.backend is None else f'_{args.backend}') + \
                ('' if args.use_default_material is None else '_defmat')
    for k, v in res.items():
        logger.info(kaolin.utils.testing.tensor_info(v, name=f'output of pass {k}', print_stats=True))
        if torch.is_floating_point(v):
            fname = os.path.join(args.output_dir, bname + f'_{k}.png')
            kaolin.io.utils.write_image(v.squeeze(0), fname)

    print(f'Wrote to {args.output_dir}')

