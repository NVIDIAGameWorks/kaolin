# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Iterable, Optional, Union, List

import torch
import os
from glob import glob
from tqdm import tqdm

from kaolin.rep.TriangleMesh import TriangleMesh
from kaolin.transforms import transforms as tfs


class ModelNet(object):
    r""" Dataset class for the ModelNet dataset.

    Args:
        basedir (str): Path to the base directory of the ModelNet dataset.
        split (str, optional): Split to load ('train' vs 'test',
            default: 'train').
        categories (iterable, optional): List of categories to load
            (default: ['chair']).
        transform (callable, optional): A function/transform to apply on each
            loaded example.
        device (str or torch.device, optional): Device to use (cpu,
            cuda, cuda:1, etc.).  Default: 'cpu'

    Examples:
        >>> dataset = ModelNet(basedir='data/ModelNet')
        >>> train_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=8)
        >>> obj, label = next(iter(train_loader))
    """

    def __init__(self, basedir: str,
                 split: Optional[str] = 'train',
                 categories: Optional[Iterable] = ['bed'],
                 transform: Optional[Callable] = None,
                 device: Optional[Union[torch.device, str]] = 'cpu'):

        assert split.lower() in ['train', 'test']

        self.basedir = basedir
        self.transform = transform
        self.device = device
        self.categories = categories
        self.names = []
        self.filepaths = []
        self.cat_idxs = []

        if not os.path.exists(basedir):
            raise ValueError('ModelNet was not found at "{0}".'.format(basedir))

        available_categories = [p for p in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, p))]

        for cat_idx, category in enumerate(categories):
            assert category in available_categories, 'object class {0} not in list of available classes: {1}'.format(
                category, available_categories)

            cat_paths = glob(os.path.join(basedir, category, split.lower(), '*.off'))

            self.cat_idxs += [cat_idx] * len(cat_paths)
            self.names += [os.path.splitext(os.path.basename(cp))[0] for cp in cat_paths]
            self.filepaths += cat_paths

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        category = torch.tensor(self.cat_idxs[index], dtype=torch.long, device=self.device)
        data = TriangleMesh.from_off(self.filepaths[index])
        data.to(self.device)
        if self.transform:
            data = self.transform(data)

        return data, category


class ModelNetVoxels(object):
    r""" Dataloader for downloading and reading from ModelNet.


    Args:
        basedir (str): location the dataset should be downloaded to /loaded from
        cache_dir (str, optional)
        split (str, optional): Split to load ('train' vs 'test',
            default: 'train').
        categories (str, optional): list of object classes to be loaded
        resolutions (list of int, optional): list of voxel grid resolutions to create,
            default: [32]
        device (str or torch.device, optional): Device to use (cpu,
            cuda, cuda:1, etc.).  Default: 'cpu'

    Returns:
        .. code-block::

        dict: {
            'attributes': {'name': str, 'class': str},
            'data': {'voxels': torch.Tensor}
        }


    Examples:
        >>> dataset = ModelNet(basedir='data/ModelNet', resolutions=[32])
        >>> train_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=8)
        >>> obj = next(iter(train_loader))
        >>> obj['data']['32'].size()
        torch.Size(32, 32, 32)
    """

    def __init__(self, basedir: str, cache_dir: Optional[str] = None, 
                 split: Optional[str] = 'train', categories: list = ['bed'],
                 resolutions: List[int] = [32],
                 device: Optional[Union[torch.device, str]] = 'cpu'):

        self.basedir = basedir
        self.device = torch.device(device)
        self.cache_dir = cache_dir if cache_dir is not None else os.path.join(basedir, 'cache')
        self.params = {'resolutions': resolutions}
        self.cache_transforms = {}

        mesh_dataset = ModelNet(basedir=basedir, split=split, categories=categories, device=device)

        self.names = mesh_dataset.names
        self.categories = mesh_dataset.categories
        self.cat_idxs = mesh_dataset.cat_idxs

        for res in self.params['resolutions']:
            self.cache_transforms[res] = tfs.CacheCompose([
                tfs.TriangleMeshToVoxelGrid(res, normalize=True, vertex_offset=0.5),
                tfs.FillVoxelGrid(thresh=0.5),
                tfs.ExtractProjectOdmsFromVoxelGrid()
            ], self.cache_dir)

            desc = 'converting to voxels to resolution {0}'.format(res)
            for idx in tqdm(range(len(mesh_dataset)), desc=desc, disable=False):
                name = mesh_dataset.names[idx]
                if name not in self.cache_transforms[res].cached_ids:
                    mesh, _ = mesh_dataset[idx]
                    mesh.to(device=device)
                    self.cache_transforms[res](name, mesh)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        name = self.names[index]

        for res in self.params['resolutions']:
            data[str(res)] = self.cache_transforms[res](name)
        attributes['name'] = name
        attributes['category'] = self.categories[self.cat_idxs[index]]
        return {'data': data, 'attributes': attributes}
