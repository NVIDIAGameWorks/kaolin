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


import kaolin as kal 
import torch
from torch._six import container_abcs


def collate_fn(batch):
    elem = batch[0]
    elem_module = type(batch[0]).__module__

    if isinstance(elem, torch.Tensor):
        if elem.is_sparse:
            return {
                'values': tuple(b.coalesce().values() for b in batch),
                'indices': tuple(b.coalesce().indices() for b in batch),
            }
        elif all([list(d.size()) == list(batch[0].size()) for d in batch]):
            return torch.stack(batch, 0)
        else:
            return batch
    elif elem_module == 'numpy':
        return torch.tensor(np.stack(batch))
    elif isinstance(elem, kal.rep.VoxelGrid):
        return torch.stack([b.voxels for b in batch], dim=0)
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    return batch


def down_sample(tgt): 
    inp = []
    for t in tgt : 
        low_res_inp = kal.rep.voxel.scale_down(t, scale=[2, 2, 2])
        low_res_inp = kal.rep.voxel.threshold(low_res_inp, .1)
        inp.append(low_res_inp.unsqueeze(0))
    inp = torch.cat(inp, dim=0)
    return inp


def up_sample(inp):
    scaling = torch.nn.Upsample(scale_factor=4, mode='nearest')
    NN_pred = []
    for voxel in inp: 
        NN_pred.append(scaling(voxel.unsqueeze(0).unsqueeze(0)).squeeze(1))
    NN_pred = torch.stack(NN_pred).squeeze(1)
    return NN_pred


def to_occupancy_map(inp, threshold=None):
    if threshold is None: 
        threshold = inp.shape[-1]
    zeros = inp < threshold
    ones = inp >= threshold
    inp = inp.clone()
    inp[ones] = 1 
    inp[zeros] = 0 
    return inp


def upsample_odm(inp): 
    scaling = torch.nn.Upsample(scale_factor=4, mode='nearest')
    inp = scaling(inp)
    return inp
