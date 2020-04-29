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

import torch
import time
import numpy as np
import kaolin as kal
from kaolin.rep import TriangleMesh
from kaolin.rep import QuadMesh
from kaolin.rep import Mesh

mesh = TriangleMesh.from_obj(('tests/model.obj') )
vertices = mesh.vertices
faces = mesh.faces
device = vertices.device

faces_gpu = faces.cuda()
vertices_gpu = vertices.cuda()
edge2key, edges, vv, vv_count, ve, ve_count, vf, vf_count, ff, ff_count, ee, ee_count, \
    ef, ef_count = Mesh.compute_adjacency_info(vertices_gpu, faces_gpu)
_edge2key, _edges, _vv, _vv_count, _ve, _ve_count, _vf, _vf_count,  _ff, _ff_count, _ee, \
    _ee_count, _ef, _ef_count = Mesh.old_compute_adjacency_info(vertices, faces)

edges = edges.cpu()
vv = vv.cpu()
vv_count = vv_count.cpu()
ve = ve.cpu()
ve_count = ve_count.cpu()
vf = vf.cpu()
vf_count = vf_count.cpu()
ff = ff.cpu()
ff_count = ff_count.cpu()
ee = ee.cpu()
ee_count = ee_count.cpu()
ef = ef.cpu()
ef_count = ef_count.cpu()

assert set(edge2key.keys()) == set(_edge2key.keys())
assert set(edge2key.values()) == set(_edge2key.values())

_key2key = {_key: edge2key[_edge] for _edge, _key in _edge2key.items()}
_key2key_arr = torch.zeros(edges.shape[0], device=device, dtype=torch.long)
for _key, key in _key2key.items():
    _key2key_arr[_key] = key
_key2key[-1] = -1
assert (edges[_key2key_arr] == _edges).all()
assert (vv_count == _vv_count).all()
assert (vv_count == torch.sum(vv != -1, dim=-1)).all()
assert (torch.sort(vv, dim=1)[0] == torch.sort(_vv, dim=1)[0]).all()
assert (ve_count == _ve_count).all()
assert (ve_count == torch.sum(ve != -1, dim=-1)).all()

for i in range(vertices.shape[0]):
    assert (torch.unique(ve[i, :ve_count[i]], sorted=True, dim=0) ==
            torch.unique(_key2key_arr[_ve[i, :ve_count[i]]], sorted=True, dim=0)).all()

assert (vf_count == _vf_count).all()
assert (vf_count == torch.sum(vf != -1, dim=-1)).all()
assert (torch.sort(vf, dim=1)[0] == torch.sort(_vf, dim=1)[0]).all()

assert (ef_count[_key2key_arr] == _ef_count).all()
assert (torch.sort(ef[_key2key_arr, :], dim=1)[0] == torch.sort(_ef, dim=1)[0]).all()

tmp_ee = torch.full(_ee.shape, fill_value=-1, dtype=torch.long)
for i in range(_ee.shape[0]):
    for j in range(_ee.shape[1]):
        tmp_ee[i, j] = _key2key_arr[_ee[i, j]]
assert (torch.sort(ee[_key2key_arr], dim=1)[0] == torch.sort(tmp_ee, dim=1)[0]).all()

assert (ff_count == _ff_count).all()
assert (torch.sort(ff, dim=1)[0] == torch.sort(_ff, dim=1)[0]).all()
