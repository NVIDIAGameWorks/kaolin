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

import torch
import os
import sys
import urllib.request
import zipfile
from glob import glob
import scipy.io as sio


def download_modelnet_category(modelnet_location):
    r"""Downloads a modelnet category to a specified direcotry

    You may complete this function to automate downloading from a
    central source.
    """
    NotImplemented


class ModelNet(object):
    r"""
    Dataloader for downloading and reading from ModelNet


    Args:
        root (str): location the dataset should be downloaded to /loaded from
        train (bool): if True loads training set, else loads test
        download (bool): downloads the dataset if not found in root
        categories (str): list of object classes to be loaded
        single_view (bool): if true only one roation is used, if not all 12 views are loaded

    Returns:
        .. code-block::
        
        dict: {
            'attributes': {'name': str, 'class': str},
            'data': {'voxels': torch.Tensor}
        }


    Examples:
        >>> dataset = ModelNet(root='../data/')
        >>> train_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=8)
        >>> obj = next(iter(train_loader))
        >>> obj['data']['data']
        torch.Size([10, 30, 30, 30])
    """

    def __init__(self, root: str = '../data/', train: bool = True, test: bool = True,
                 download: bool = True, categories: list = ['chair'], single_view: bool = True):
        if not os.path.exists(root + '/ModelNet/'):
            assert download, "ModelNet is not found, and download is set to False"
            assert (train or test), 'either train or test must be set to True'

            download_modelnet_category(root)

        if not single_view:
            side = ''
        else:
            side = '_1'

        all_classes = [os.path.basename(os.path.dirname(c))
                       for c in glob(root + '/ModelNet/volumetric_data/*/')]
        self.names = []
        for category in categories:
            assert category in all_classes, 'object class {0} not in list of availible classes: {1}'.format(
                category, all_classes)
            if train:
                self.names += glob(
                    root + '/ModelNet/volumetric_data/{0}/*/{1}/*{2}.mat'.format(category, 'train', side))
            if test:
                self.names += glob(
                    root + '/ModelNet/volumetric_data/{0}/*/{1}/*{2}.mat'.format(category, 'test', side))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, item):
        object_path = self.names[item]

        # convert the path to the linux like path in o
        if sys.platform.startswith('win'):
            object_path = object_path.replace('\\', '/')

        
        object_class = object_path.split('/')[-4]
        object_name = object_path.split('/')[-1]
        object_data = sio.loadmat(object_path)['instance']
        # object_shape = object_data.shape

        data = dict()
        attributes = dict()
        attributes['name'] = object_name
        attributes['class'] = object_class
        data['voxels'] = torch.FloatTensor(object_data.astype(float))

        return {'attributes': attributes, 'data': data}
