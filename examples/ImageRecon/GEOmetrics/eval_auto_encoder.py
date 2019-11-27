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

import argparse
import torch
from tqdm import tqdm

from architectures import MeshEncoder, VoxelDecoder
import kaolin as kal 

parser = argparse.ArgumentParser()
parser.add_argument('-expid', type=str, default='Direct', help='Unique experiment identifier.')
parser.add_argument('-device', type=str, default='cuda', help='Device to use')
parser.add_argument('-categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('-vis', action='store_true', help='Visualize show_each model while evaluating')
parser.add_argument('-batch_size', type=int, default=25, help='batch size')
args = parser.parse_args()


# Data
mesh_set = kal.dataloader.ShapeNet.Surface_Meshes(root='../../datasets/', categories=args.categories,
                                                  resolution=32, train=False, split=.7, mode='Tri')
voxel_set = kal.dataloader.ShapeNet.Voxels(root='../../datasets/', categories=args.categories,
                                           train=False, resolutions=[32], split=.7)
valid_set = kal.dataloader.ShapeNet.Combination([mesh_set, voxel_set], root='../../datasets/')


encoder = MeshEncoder(30).to(args.device)
decoder = VoxelDecoder(30).to(args.device)


encoder.load_state_dict(torch.load('log/{}/auto_best_encoder.pth'.format(args.expid)))
decoder.load_state_dict(torch.load('log/{}/auto_best_decoder.pth'.format(args.expid)))

loss_epoch = 0.
num_batches = 0
num_items = 0

encoder.eval(), decoder.eval()
with torch.no_grad():
    for i in tqdm(range(len(valid_set))): 
        # Data Creation
        tgt_voxels = valid_set[i]['32'].to(args.device)
        inp_verts = valid_set[i]['verts'].to(args.device)
        inp_faces = valid_set[i]['faces'].to(args.device)
        inp_adj = valid_set[i]['adj'].to(args.device)

        # Inference
        latent_encoding = encoder(inp_verts, inp_adj).unsqueeze(0)
        pred_voxels = decoder(latent_encoding)[0]

        # Losses
        iou = kal.metrics.voxel.iou(pred_voxels.contiguous(), tgt_voxels.contiguous())

        if args.vis: 
            tgt_mesh = kal.rep.TriangleMesh.from_tensors(inp_verts, inp_faces)
            print('Rendering Input Mesh')
            tgt_mesh.show()
            print('Rendering Target Voxels')
            kal.visualize.show_voxel(tgt_voxels, mode='exact', thresh=.5)
            print('Rendering Predicted Voxels')
            kal.visualize.show_voxel(pred_voxels, mode='exact', thresh=.5)
            print('----------------------')
            num_items += 1

        loss_epoch += iou.item()
        num_batches += 1.

out_loss = loss_epoch / float(num_batches)
print('IoU over validation set is {0}'.format(out_loss))
