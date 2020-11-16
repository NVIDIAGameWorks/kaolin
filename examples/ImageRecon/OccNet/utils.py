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


import torch 
from torchvision import transforms
import numpy as np
from kaolin import mcubes
from torch._six import container_abcs


def rgba_to_rgb_white(image_tensor):
    """ Converts transparent pixels to white.
        Returns RGB image.
    """
    if image_tensor.size(0) == 4:
        image_tensor[:, image_tensor[3] == 0.] = 1.
    return image_tensor[:3]


preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    rgba_to_rgb_white,
])


def get_prior_z(cfg, device, **kwargs):
    """ Returns prior distribution for latent code z.
    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    """
    z_dim = 0
    p0_z = dist.Normal(
        torch.zeros(z_dim, device=device),
        torch.ones(z_dim, device=device)
    )

    return p0_z


def occ_function(model, code):
    z = torch.zeros(1, 0)

    def eval_query(query):
        pred_occ = model.decode(query.unsqueeze(0), z, code).probs.squeeze(0)
        return pred_occ
    return eval_query


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
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    return batch


def extract_mesh(occ_hat, model, z, c=None, padding=0.1, threshold=0.2):
    n_x, n_y, n_z = occ_hat.shape
    box_size = 1. + padding
    # Make sure that mesh is watertight
    occ_hat_padded = np.pad(
        occ_hat, 1, 'constant', constant_values=-1e6)
    vertices, triangles = mcubes.marching_cubes(
        occ_hat_padded, threshold)
    vertices -= 0.5
    vertices -= 1
    vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
    vertices = box_size * (vertices - 0.5)

    return torch.FloatTensor(vertices.astype(float)).cuda(), torch.LongTensor(triangles.astype(int)).cuda()
