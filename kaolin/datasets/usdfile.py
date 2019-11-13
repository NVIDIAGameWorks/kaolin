# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import logging as log
import torch
from torch.utils import data
from pathlib import Path
from pxr import Usd, UsdGeom, UsdLux, Sdf, Gf, Vt

from kaolin import helpers


def get_mesh_attributes(usd_mesh):
    face_count = torch.tensor(usd_mesh.GetFaceVertexCountsAttr().Get())[0]
    vertices = torch.tensor(usd_mesh.GetPointsAttr().Get())
    faces = torch.tensor(usd_mesh.GetFaceVertexIndicesAttr().Get())
    faces = faces.view(-1, face_count)
    return {'vertices': vertices, 'faces': faces}


class USDMeshes(data.Dataset):
    """Import mesh objects from a USD file.

        Args:
            usd_filepath (str): Path to usd file (*.usd, *.usda)

        Returns:
            dict: {
                'attributes': {'name': str}
                'data': {'vertices': torch.Tensor, 'faces': torch.Tensor}
            }

    Example:
            >>> usd_dataset = USDMeshes(usd_filepath='Kitchen_set.usd',
            >>>                         cache_dir='./datasets/USDMeshes/')
            >>> obj = next(iter(usd_dataset))
            >>> obj['data']['vertices'].shape
            torch.Size([114, 3])
            >>> obj['data']['faces'].shape
            torch.Size([448, 3])
    """
    def __init__(self, usd_filepath: str, cache_dir: str = '../data/USDMeshes'):
        usd_filepath = Path(usd_filepath)
        assert usd_filepath.suffix in ['.usd', '.usda']
        assert usd_filepath.exists(), f'USD file at {usd_filepath} was not found.'

        self.cache = helpers.Cache(get_mesh_attributes, cache_dir,
                                   cache_key=helpers._get_hash(usd_filepath))
        self.names = self.cache.cached_ids

        stage = Usd.Stage.Open(str(usd_filepath))
        mesh_prims = [x for x in stage.Traverse() if UsdGeom.Mesh(x)]
        uncached_mesh_prims = filter(lambda x: x.GetName() not in self.names, mesh_prims)
        for mesh_prim in uncached_mesh_prims:
            name = mesh_prim.GetName()
            mesh = UsdGeom.Mesh(mesh_prim)
            face_counts = torch.tensor(mesh.GetFaceVertexCountsAttr().Get())
            if not torch.allclose(face_counts, face_counts[0]):
                log.warn(f'Skipping mesh {name}, not all faces have the same '
                             'number of vertices.')
            else:
                self.cache(name, usd_mesh=mesh)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        attributes = {'name': self.names[index]}
        data = self.cache(self.names[index])
        return {'attributes': attributes, 'data': data}
