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
from torchvision import transforms
import kaolin as kal
from collections import defaultdict
import numpy as np
from torch._six import container_abcs

from kaolin.rep import TriangleMesh


preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])


def merge_mesh(mesh_list, device):
    vert = torch.cat([torch.from_numpy(mesh.vertices).to(device).float() for mesh in mesh_list], dim=0)
    faces_list = []
    current_index = 0
    for mesh in mesh_list:
        faces = torch.from_numpy(mesh.faces).to(device).float() + current_index
        # faces2 = faces.clone()
        # faces2 = faces2[:, [0,2,1]]
        current_index = current_index + mesh.vertices.shape[0]
        faces_list.append(faces)
        # faces_list.append(faces2)

    faces = torch.cat(faces_list, dim=0)
    return kal.rep.TriangleMesh.from_tensors(vert, faces), vert, faces