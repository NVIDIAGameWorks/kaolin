# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

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
from PIL import Image
import torch

import kaolin as kal
from kaolin.models.OccupancyNetwork import OccupancyNetwork

from utils import occ_function, extract_mesh, preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='Path to image to generate mesh from.')
parser.add_argument('--expid', type=str, default='OccNet_1', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('--no-vis', action='store_true', help='Visualize each model while evaluating')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')

args = parser.parse_args()


# Model
model = OccupancyNetwork(args.device)

THRESHOLD = 0.2
PADDING = 0.1
UPSAMPLING_STEPS = 2
RESOLUTION_0 = 64

# Load saved weights
logdir = os.path.join(args.logdir, args.expid)
checkpoint = torch.load(os.path.join(logdir, 'best.ckpt'))
model.load_state_dict(checkpoint['model'])


with torch.no_grad():
    model.encoder.eval()
    model.decoder.eval()

    if os.path.isdir(args.image):
        images = [os.path.join(args.image, f) for f in os.listdir(args.image)]
    else:
        images = [args.image]

    for image in images:
        image_pil = Image.open(image)
        image_tensor = preprocess(image_pil).unsqueeze(0).to(args.device)
        encoding = model.encode_inputs(image_tensor)
        z = model.get_z_from_prior((1,), sample=True).to(args.device)

        threshold = THRESHOLD
        box_size = 1 + PADDING

        sdf = occ_function(model, encoding)
        voxelgrid = kal.conversions.sdf_to_voxelgrid(
            sdf, resolution=RESOLUTION_0, upsampling_steps=UPSAMPLING_STEPS,
            threshold=threshold, bbox_dim=PADDING)

        # Extract mesh from sdf 
        verts, faces = extract_mesh(voxelgrid, model, z, encoding)
        verts, faces = verts.to(args.device), faces.to(args.device)
        mesh = kal.rep.TriangleMesh.from_tensors(verts, faces)

        if verts.shape[0] == 0:     # if mesh is empty count as 0 f-score
            print('Generated mesh is empty!')

        if not args.no_vis: 
            print('Displaying input image')
            img = image_tensor[0].permute(1, 2, 0).data.cpu().numpy() * 255
            img = (img).astype(np.uint8)
            Image.fromarray(img).show()
            print('Rendering Predicted Mesh')
            mesh.show()
            print('----------------------')
