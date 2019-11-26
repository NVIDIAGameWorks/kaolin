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
import sys
from glob import glob
import scipy.io as sio
from tqdm import tqdm

import kaolin as kal
from kaolin.rep.TriangleMesh import TriangleMesh
from kaolin.transforms import transforms as tfs


# _MODELNET10_CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
#                        'night_stand', 'sofa', 'table', 'toilet']


# class ModelNet10(torch.utils.data.Dataset):
#     r"""Dataset class for the ModelNet10 dataset.

#     Args:
#         basedir (str): Path to the base directory of the ModelNet10 dataset.
#         rep (str, optional): Type of representation to convert the dataset into
#             (default: 'mesh').
#         split (str, optional): Split to load ('train' vs 'test',
#             default: 'train').
#         categories (iterable, optional): List of categories to load
#             (default: ['chair']).
#         device (str or torch.device, optional): Device to use (cpu,
#             cuda-capable device, etc.).
#         transform (callable, optional): A function/transform to apply on each
#             loaded example.

#     **kwargs
#         num_points (int, optional): Number of points in the returned pointcloud
#             (if using pointcloud representation, default: 1024).

#     """

#     def __init__(self, basedir: str, rep: Optional[str] = 'mesh',
#                  split: Optional[str] = 'train',
#                  categories: Optional[Iterable] = ['bed'],
#                  device: Optional[Union[torch.device, str]] = 'cpu',
#                  transform: Optional[Callable] = None,
#                  **kwargs):

#         super(ModelNet10, self).__init__()

#         if rep.lower() not in ['mesh', 'pointcloud']:
#             raise ValueError('Argument \'rep\' must be one of \'mesh\' '
#                 ' or \'pointcloud\'. Got {0} instead.'.format(rep))
#         if split.lower() not in ['train', 'test']:
#             raise ValueError('Argument \'split\' must be one of \'train\' '
#                 ' or \'test\'. Got {0} instead.'.format(split))

#         self.categories = categories
#         self.paths = []
#         self.labels = []
#         for idx, cat in enumerate(self.categories):

#             if cat not in _MODELNET10_CLASSES:
#                 raise ValueError('Invalid ModelNet10 class {0}. Valid classes '
#                                  'are {1}'.format(cat, _MODELNET10_CLASSES))

#             catdir = os.path.join(basedir, cat, split)
#             for path in glob(os.path.join(catdir, '*.off')):
#                 self.paths.append(path)
#                 self.labels.append(idx)

#         self.rep = rep
#         self.device = device
#         self.transform = transform

#         # Set defaults for kwargs
#         if 'num_points' in kwargs:
#             self.num_points = kwargs['num_points']
#         else:
#             self.num_points = 1024

#     def __len__(self):
#         r"""Returns the length of the dataset. """
#         return len(self.paths)

#     def __getitem__(self, idx):
#         r"""Returns the item at index `idx`. """

#         mesh = kal.rep.TriangleMesh.from_off(self.paths[idx])
#         mesh.to(self.device)
#         label = torch.LongTensor([self.labels[idx]]).to(self.device)
#         if self.rep == 'mesh':
#             if self.transform is not None:
#                 mesh = self.transform(mesh)
#             return mesh, label
#         elif self.rep == 'pointcloud':
#             pts, _ = mesh.sample(self.num_points)
#             if self.transform is not None:
#                 pts = self.transform(pts)
#             return pts, label
#         else:
#             raise NotImplementedError


class ModelNet(object):
    r""" Dataset class for the ModelNet dataset.

    Args:
        basedir (str): Path to the base directory of the ModelNet10 dataset.
        split (str, optional): Split to load ('train' vs 'test',
            default: 'train').
        categories (iterable, optional): List of categories to load
            (default: ['chair']).
        device (str or torch.device, optional): Device to use (cpu,
            cuda-capable device, etc.).
        transform (callable, optional): A function/transform to apply on each
            loaded example.

    **kwargs
        num_points (int, optional): Number of points in the returned pointcloud
            (if using pointcloud representation, default: 1024).

    Examples:
        >>> dataset = ModelNet(basedir='data/ModelNet')
        >>> train_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=8)
        >>> obj = next(iter(train_loader))
        >>> obj['data']['data']
        torch.Size([10, 30, 30, 30])
    """

    def __init__(self, basedir: str,
                 split: Optional[str] = 'train',
                 categories: Optional[Iterable] = ['bed'],
                 device: Optional[Union[torch.device, str]] = 'cpu',
                 transform: Optional[Callable] = None,
                 **kwargs):
        assert split.lower() in ['train', 'test']

        self.basedir = basedir
        self.transform = transform
        self.device = device
        self.categories = categories
        self.names = []
        self.filepaths = []
        self.cat_idxs = []

        if not os.path.exists(basedir):
            ValueError('ModelNet was not found at {0}.'.format(basedir))

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


class ModelNet_Voxels(object):
    r""" Dataloader for downloading and reading from ModelNet.


    Args:
        basedir (str): location the dataset should be downloaded to /loaded from
        train (bool): if True loads training set, else loads test
        download (bool): downloads the dataset if not found in basedir
        categories (str): list of object classes to be loaded

    Returns:
        .. code-block::

        dict: {
            'attributes': {'name': str, 'class': str},
            'data': {'voxels': torch.Tensor}
        }


    Examples:
        >>> dataset = ModelNet(basedir='../data/')
        >>> train_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=8)
        >>> obj = next(iter(train_loader))
        >>> obj['data']['data']
        torch.Size([10, 30, 30, 30])
    """

    def __init__(self, basedir: str = 'data/ModelNet/', cache_dir: str = 'cache', train: bool = True, categories: list = ['bed'],
                 resolutions: List[int] = [32], device: str = 'cpu'):

        self.basedir = basedir
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self.params = {'resolutions': resolutions}
        self.cache_transforms = {}

        mesh_dataset = ModelNet(basedir, train, categories, device)

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
                    sample = mesh_dataset[idx]
                    mesh = TriangleMesh.from_tensors(sample['data']['vertices'],
                                                     sample['data']['faces'])
                    mesh.to(device='cuda:1')
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
