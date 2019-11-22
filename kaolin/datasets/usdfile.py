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
from kaolin.rep.TriangleMesh import Mesh


def get_mesh_attributes(usd_mesh):
    vertices = torch.tensor(usd_mesh.GetPointsAttr().Get())

    face_counts = usd_mesh.GetFaceVertexCountsAttr().Get()
    faces_raw = usd_mesh.GetFaceVertexIndicesAttr().Get()
    if all(fc == face_counts[0] for fc in face_counts):
        faces = torch.tensor(faces_raw)
        faces = faces.view(-1, face_counts[0])
    else:
        idx = 0
        faces = []
        for face_count in face_counts:
            faces.append(faces_raw[idx:(idx + face_count)])
            idx += face_count
        faces = [list(f) for f in faces]
        faces = Mesh.homogenize_faces(faces)
        faces = torch.tensor(faces)
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
            self.cache(name, usd_mesh=mesh)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        attributes = {'name': self.names[index]}
        data = self.cache(self.names[index])
        return {'attributes': attributes, 'data': data}
