#!/usr/bin/env python

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
import random
import torch
import sys

import kaolin

logger = logging.getLogger(__name__)

def __normalize_vertices(vertices):
    """
    Normalizes vertices to fit an [-1...1] bounding box,
    common during training, but not necessary for visualization.
    """
    return kaolin.ops.pointcloud.center_points(res.vertices.unsqueeze(0), normalize=True).squeeze(0) * 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Example exporting 3D data during training as timed USDs; '
        'also demonstrates OBJ import and mesh to pointcloud conversions.')
    parser.add_argument('--test_objs', type=str, required=True,
                        help='Comma separated list of several example obj files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory to write checkpoints to; must exist.')
    parser.add_argument('--iterations', type=int, default=101,
                        help='How many training iterations to emulate.')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                        help='Frequency with which to write out checkpoints.')
    parser.add_argument('--skip_normalization', action='store_true',
                        help='If not set, will normalize bounding box of each input '
                        'to be within -1..1 cube.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    if not os.path.isdir(args.output_dir):
        raise RuntimeError(
            'Output directory does not exist: --output_dir={}'.format(
                args.output_dir))

    # Read test 3D models & setup fake training ----------------------------------
    obj_files = args.test_objs.split(',')
    logger.info('Parsing {} OBJ files: '.format(len(obj_files)))
    face_list = []
    gt_vert_list = []
    input_pt_clouds = []
    delta_list = []
    delta_pt_list = []
    # TODO: add textured example
    for f in obj_files:
        res = kaolin.io.obj.import_mesh(f)
        vertices = res.vertices if args.skip_normalization else __normalize_vertices(vertices)
        num_samples = random.randint(1000, 1500)  # Vary to ensure robustness
        pts = kaolin.ops.mesh.sample_points(
            vertices.unsqueeze(0), res.faces, num_samples)[0].squeeze(0)

        # Randomly displace vertices to emulate training
        delta = (2.0 * torch.rand(vertices.shape, dtype=vertices.dtype) - 1.0) * 0.25
        delta_pts = (2.0 * torch.rand(pts.shape, dtype=pts.dtype) - 1.0) * 0.25

        face_list.append(res.faces)
        gt_vert_list.append(vertices)
        delta_list.append(delta)
        input_pt_clouds.append(pts)
        delta_pt_list.append(delta_pts)

    # Emulate visualizing during training -------------------------------------
    logger.info('Emulating training run for {} iterations'.format(args.iterations))

    # Create a Timelapse instance
    timelapse = kaolin.visualize.Timelapse(args.output_dir)

    # Save static objects such as ground truth or inputs that do not change with iterations
    # just once.
    timelapse.add_mesh_batch(
        category='ground_truth',
        faces_list=face_list,
        vertices_list=gt_vert_list)
    timelapse.add_pointcloud_batch(
        category='input',
        pointcloud_list=input_pt_clouds)

    for iteration in range(args.iterations):
        if iteration % args.checkpoint_interval == 0:
            # Emulate a training update
            out_pt_clouds = []
            out_vert_list = []
            out_voxels = []
            for i in range(len(gt_vert_list)):
                delta_weight = 1.0 - iteration / (args.iterations - 1)
                out_vert_list.append(gt_vert_list[i] * (1.0 + delta_list[i] * delta_weight))
                out_pt_clouds.append(input_pt_clouds[i] * (1.0 + delta_pt_list[i] * delta_weight))
                vg = kaolin.ops.conversions.trianglemeshes_to_voxelgrids(
                    out_vert_list[-1].unsqueeze(0), face_list[i], 30)
                out_voxels.append(vg.squeeze(0).bool())

            # Save model predictions to track training progress over time
            timelapse.add_mesh_batch(
                iteration=iteration,
                category='output',
                faces_list=face_list,
                vertices_list=out_vert_list)
            timelapse.add_pointcloud_batch(
                iteration=iteration,
                category='output',
                pointcloud_list=out_pt_clouds)
            timelapse.add_voxelgrid_batch(
                iteration=iteration,
                category='output',
                voxelgrid_list=out_voxels)

    logger.info('Emulated training complete!\n'
                'You can now view created USD files by running:\n\n'
                f'kaolin-dash3d --logdir={args.output_dir}\n\n'
                'And then navigating to localhost:8080\n')

    # TODO(mshugrina): once dash3d is also integrated, write an integration test
    # to ensure timelapse output is properly parsed by the visualizer
