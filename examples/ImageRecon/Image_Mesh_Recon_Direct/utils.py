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
from torchvision.transforms import Normalize as norm 

import kaolin as kal



preprocess = transforms.Compose([
   transforms.CenterCrop(137),
   transforms.ToTensor()
])


def loss_lap(mesh1, deltas ): 
	mesh2 = kal.rep.TriangleMesh.from_tensors(mesh1.vertices - deltas, mesh1.faces)
	loss =   kal.metrics.mesh.laplacian_loss(mesh1, mesh2)
	loss += torch.sum((deltas)**2, 1).mean() * .0666 
	return loss 