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

import os
import argparse

from kaolin.io import usd
from kaolin.io.utils import mesh_handler_naive_triangulate


def import_kitchen_set(kitchen_set_usd):
    # The Kitchen Set example organizes assets in a particular way. Since we want to import complete objects and not
    # not each separate part of an object, we'll find all the paths that are named :code:`Geom`:
    scene_paths = usd.get_scene_paths(kitchen_set_usd, r'.*/Geom$')

    # The meshes in this dataset have a heterogeneous topology, meaning the number of vertices 
    # for each polygon varies. To deal with those, we'll pass in a handler function that will 
    # homogenize those meshes to homogenous triangle meshes.
    usd_meshes = usd.import_meshes(
        kitchen_set_usd,
        scene_paths=scene_paths,
        heterogeneous_mesh_handler=mesh_handler_naive_triangulate
    )
    return usd_meshes


def save_kitchen_set_dataset(meshes, out_dir):
    for i, m in enumerate(meshes):
        out_path = os.path.join(out_dir, f'mesh_{i}.usd')
        usd.export_mesh(
            file_path=out_path,
            vertices=m.vertices[..., [0, 2, 1]],    # flipping Y and Z to make models Y-up
            faces=m.faces
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=r'Convert Pixar\'s Kitchen Set scene (http://graphics.pixar.com/usd/downloads.html) '
        'into a dataset of Pytorch Tensors, ready to be use to train our next awesome model.')
    parser.add_argument('--kitchen_set_dir', type=str, required=True,
                        help='Location of the kitchen_set data.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory to export the dataset to; must exist.')

    args = parser.parse_args()

    # We will be importing Pixar's Kitchen Set scene (http://graphics.pixar.com/usd/downloads.html) as a 
    # dataset of Pytorch Tensors, ready to be used to train our next awesome model. 
    kitchen_set_usd = os.path.join(args.kitchen_set_dir, 'Kitchen_set.usd')
    meshes = import_kitchen_set(kitchen_set_usd)

    print(len(meshes))  # 426
    # And just like that, we have a dataset of 426 diverse objects for our use!

    # Now let's save our dataset so we can use it again later.
    save_kitchen_set_dataset(meshes, args.output_dir)

    # We can now fire up Omniverse Kaolin and use the Dataset Visualizer extension to 
    # see what this dataset looks like and start using it in our next project!
