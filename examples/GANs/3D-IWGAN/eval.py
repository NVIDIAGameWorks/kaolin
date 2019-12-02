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


import os
import argparse
import torch

from architectures import Generator
import kaolin as kal 


torch.manual_seed(0)
torch.cuda.manual_seed(0)


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--expid', type=str, default='3D_IWGAN', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use.')
parser.add_argument('--batchsize', type=int, default=50, help='Batch size.')
args = parser.parse_args()

logdir = os.path.join(args.logdir, args.expid)

gen = Generator().to(args.device)
gen.load_state_dict(torch.load(os.path.join(logdir, 'gen.pth')))
gen.eval()

z = torch.normal(torch.zeros(args.batchsize, 200), torch.ones(args.batchsize, 200)).to(args.device)

fake_voxels = gen(z)

for i, model in enumerate(fake_voxels): 
    print('Rendering model {}'.format(i))
    model = model[:-2, :-2, :-2]
    model = kal.transforms.voxelfunc.max_connected(model, .7)
    verts, faces = kal.conversions.voxelgrid_to_quadmesh(model)
    mesh = kal.rep.QuadMesh.from_tensors(verts, faces)
    mesh.laplacian_smoothing(iterations=3)
    mesh.show()
