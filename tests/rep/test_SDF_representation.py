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

import pytest

import torch
import sys

import kaolin as kal
from kaolin.rep import TriangleMesh

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_check_sign(device): 
    mesh = TriangleMesh.from_obj(('tests/model.obj') )
    if device == 'cuda': 
        mesh.cuda()
    points = torch.rand((1000,3), device=device) -.5
    signs = kal.rep.SDF.check_sign(mesh, points)
    assert (signs == True).sum() > 0 
    assert (signs == False).sum() > 0 

    points = (torch.rand((1000,3), device=device) -.5) * .001
    signs = kal.rep.SDF.check_sign(mesh, points)
    assert (signs == False).sum() == 0 


    points = torch.rand((1000,3), device=device) +10
    signs = kal.rep.SDF.check_sign(mesh, points)
    assert (signs == True).sum() == 0 

# def test_check_sign_gpu(): 
#     test_check_sign("cuda")


# def test_check_sign_fast(device='cuda'):
#     mesh = TriangleMesh.from_obj('tests/model.obj')
#     mesh.to(device)
#     points = torch.rand(1000, 3).to(device) - .5
#     signs = kal.rep.SDF.check_sign_fast(mesh, points)
#     assert (signs == True).float().sum() > 0
#     assert (signs == False).sum() > 0


# if __name__ == '__main__':

# 	mesh = TriangleMesh.from_obj('tests/model.obj')
# 	mesh.cuda()
# 	points = torch.rand(1000, 3).cuda() - .5
# 	sign_fast = kal.rep.SDF.check_sign_fast(mesh, points)

# 	mesh = TriangleMesh.from_obj('tests/model.obj')
# 	mesh.cuda()
# 	sign = kal.rep.SDF.check_sign(mesh, points)
# 	import numpy as np
# 	sign = torch.from_numpy(np.asarray(sign)).cuda()
    
# 	print((sign == sign_fast).float().sum())
