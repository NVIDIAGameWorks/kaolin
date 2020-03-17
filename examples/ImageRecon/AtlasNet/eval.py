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
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from easydict import EasyDict


from PIL import Image

import kaolin as kal
from kaolin.datasets import shapenet

from kaolin.models.AtlasNet import AtlasNet
from kaolin.models.AtlasNet import Resnet18 as Encoder


from utils import merge_mesh, preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--shapenet-root', type=str, help='Root directory of the ShapeNet dataset.')
parser.add_argument('--shapenet-images-root', type=str, help='Root directory of the ShapeNet Rendering dataset.')
parser.add_argument('--cache-dir', type=str, default='cache', help='Path to write intermediate representation to.')
parser.add_argument('--expid', type=str, default='AtlasNet_1', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('--no-vis', action='store_true', help='Turn off visualization of each model while evaluating')
parser.add_argument('--f-score', action='store_true', help='compute F-score')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size.')
parser.add_argument('--logdir', type=str, default='log', help='Directory where log data was saved to.')
args = parser.parse_args()


# Data
points_set_valid = shapenet.ShapeNet_Points(root=args.shapenet_root, cache_dir=args.cache_dir, categories=args.categories,
                                            train=False, split=.7, num_points=5000)
images_set_valid = shapenet.ShapeNet_Images(root=args.shapenet_images_root, categories=args.categories,
                                            train=False, split=.7, views=1, transform=preprocess)
meshes_set_valid = shapenet.ShapeNet_Meshes(root=args.shapenet_root, categories=args.categories,
                                            train=False, split=.7)
valid_set = shapenet.ShapeNet_Combination([points_set_valid, images_set_valid, meshes_set_valid])

dataloader_val = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

# Model
opt = EasyDict({
    "device": args.device,
    "number_points": 2048,
    "number_points_eval": 10000,
    "nb_primitives": 25,
    "remove_all_batchNorms": False,
    "template_type": "SQUARE", #Can also be SPHERE
    "bottleneck_size": 1024,
    "dim_template": 2,
    "hidden_neurons": 512,
    "num_layers": 2,
    "activation": "relu", # can be "relu", "sigmoid", "softplus", "logsigmoid", "softsign", "tanh"
})

encoder = Encoder(c_dim=opt.bottleneck_size).to(args.device)
decoder = AtlasNet(opt).to(args.device)


logdir = os.path.join(args.logdir, args.expid)

# Load saved weights
checkpoint = torch.load(os.path.join(logdir, 'recent.ckpt'))
encoder.load_state_dict(checkpoint['encoder'])
encoder.eval()
decoder.load_state_dict(checkpoint['decoder'])
decoder.eval()


loss_epoch = 0.
f_epoch = 0.
num_batches = 0
num_items = 0

with torch.no_grad():
    for sample in tqdm(valid_set):
        data = sample['data']
        # data creation
        tgt_points = data['points'].to(args.device)
        tgt_points = tgt_points.unsqueeze(0)
        tgt_points = tgt_points - tgt_points.mean(1, keepdim=True)
        tgt_points = tgt_points / torch.sqrt(
            torch.max((tgt_points ** 2).sum(2, keepdim=True), 1, keepdim=True)[0])
        tgt_points = tgt_points[0]


        inp_images = data['images'][:3].to(args.device).unsqueeze(0)
        tgt_verts = data['vertices'].to(args.device)
        tgt_faces = data['faces'].to(args.device)

        # Inference
        img_features = encoder(inp_images)
        generated_mesh, generated_verts, generated_faces = merge_mesh(decoder.generate_mesh(img_features), args.device)
        loss = kal.metrics.point.chamfer_distance(generated_verts, tgt_points.float())

        if not args.no_vis:
            generated_mesh = kal.rep.TriangleMesh.from_tensors(generated_verts, generated_faces)
            tgt_mesh = kal.rep.TriangleMesh.from_tensors(tgt_verts, tgt_faces)

            print('Displaying input image')
            img = inp_images[0].data.cpu().numpy().transpose((1, 2, 0))
            img = (img * 255.).astype(np.uint8)
            Image.fromarray(img).show()
            print('Rendering Target Mesh')
            kal.visualize.show_mesh(tgt_mesh)
            print('Rendering Predicted Mesh')
            kal.visualize.show_mesh(generated_mesh)
            print('----------------------')
            num_items += 1

        if args.f_score:
            # Compute f score
            f_score = kal.metrics.point.f_score(tgt_points, generated_verts, extend=False)
            f_epoch += f_score.item()

        loss_epoch += loss.item()

        num_batches += 1.

out_loss = loss_epoch / num_batches
print(f'Loss over validation set is {out_loss}')
if args.f_score:
    out_f = f_epoch / num_batches
    print(f'F-score over validation set is {out_f}')