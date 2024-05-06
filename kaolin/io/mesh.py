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

import os

import kaolin.io.gltf
import kaolin.io.obj
import kaolin.io.off
import kaolin.io.usd
import kaolin.io.utils


def import_mesh(filename, triangulate=False):
    """ Automatically selects appropriate reader and reads mesh from file. Supported formats: USD, obj, gltf.
    Will read all present mesh attributes. For more fine-grained control over format-specific imports,
    refer to `kaolin.io.usd.import_mesh`, `kaolin.io.obj.import_mesh` and `kaolin.io.gltf.import_mesh`.
    # TODO: proper docs refs

    Args:
        filename (str): path to the filename
        triangulate (bool): if input should be triangulated on import

    Returns:
        (kaolin.rep.SurfaceMesh): the read mesh object
    """
    if not os.path.isfile(filename):
        raise RuntimeError(f'Input file does not exist: {filename}')

    # TODO: if off and gltf support consistent settings with the rest, then we can expose these
    default_settings = {'with_materials': True, 'with_normals': True, "triangulate": triangulate,
                        'heterogeneous_mesh_handler': kaolin.io.utils.mesh_handler_naive_triangulate}

    # TODO: add support for off
    extension = filename.split('.')[-1].lower()
    if extension == 'obj':
        mesh = kaolin.io.obj.import_mesh(filename, **default_settings, raw_materials=False)
    elif extension in ["usd", "usda", "usdc"]:
        mesh = kaolin.io.usd.import_mesh(filename, **default_settings)
    elif extension in ['gltf', 'glb']:
        mesh = kaolin.io.gltf.import_mesh(filename)
    else:
        raise ValueError(f'Unsupported filename extension {extension}')

    return mesh