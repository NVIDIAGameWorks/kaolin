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

from abc import ABC, abstractmethod
import sys
import os
from pathlib import Path
import torch
import torch.utils.data as data
from torch.multiprocessing import Pool
import warnings
import urllib.request
import zipfile
import json
import re
from collections import OrderedDict
from glob import glob
import numpy as np
import random

from tqdm import tqdm
import scipy.sparse
import tarfile
from PIL import Image

import kaolin as kal
from kaolin.rep.TriangleMesh import TriangleMesh
from kaolin.rep.QuadMesh import QuadMesh

from kaolin.transforms import pointcloudfunc as pcfunc
from kaolin.transforms import meshfunc
from kaolin.transforms import voxelfunc
from kaolin.transforms import transforms as tfs
from kaolin import helpers
import kaolin.conversions.meshconversions as mesh_cvt

import functools
import inspect

def _preprocess_task(args):
    torch.set_num_threads(1)
    with torch.no_grad():
        idx, get_data, get_attributes, cache_transform = args
        name = get_attributes(idx)['name']
        if name not in cache_transform.cached_ids:
            data = get_data(idx)
            cache_transform(name, *data)

class KaolinDatasetMeta(type):
    def __new__(cls, cls_name, base_cls, class_dict):
        if cls_name != "KaolinDataset":
            class_dict['__doc__'] += \
                """Additional args:
        preprocessing_params (dict): parameters for the preprocessing:
            - 'cache_dir': path to the cached preprocessed data.
            - 'num_workers': number of process used in parallel for preprocessing (default: number of cores)
        preprocessing_transform (Callable): Called on the outputs of _get_data over the indices
                                            from 0 to len(self) during the construction of the dataset,
                                            the preprocessed outputs are then cached to 'cache_dir'.
        transform (Callable): Called on the preprocessed data at __getitem__.
        no_progress (bool): disable tqdm progress bar for preprocessing."""
        return type.__new__(cls, cls_name, base_cls, class_dict)

class KaolinDataset(data.Dataset, metaclass=KaolinDatasetMeta):
    """
    Abstract class for dataset with handling of multiprocess or cuda preprocessing.

    A KaolinDataset children class will need the above implementation:
       1) _initialize:
           Initialization function called at the beginning of the constructor.
       2) _get_data:
           Data getter that will be preprocessed => cached => transformed, take an index as input.
       3) _get_attributes:
           Attributes getter that will be preprocess / transform independent.
       4) __len__:
           Return the size of the dataset
    """
    def __init__(self, *args, preprocessing_transform=None, preprocessing_params: dict = None,
                 transform=None, no_progress: bool = False, **kwargs):
        """
        Args:
            positional and keyword arguments for initialize(*args, **kwargs) (see class and initialize documentation)
            preprocessing_params (dict): parameters for the preprocessing:
                - 'cache_dir': path to the cached preprocessed data.
                - 'num_workers': number of process used in parallel for preprocessing (default: number of cores)
            preprocessing_transform (Callable): Called on the outputs of _get_data over the indices
                                                from 0 to len(self) during the construction of the dataset,
                                                the preprocessed outputs are then cached to 'cache_dir'.
            transform (Callable): Called on the preprocessed data at __getitem__.
            no_progress (bool): disable tqdm progress bar for preprocessing.
        """
        self.initialize(*args, **kwargs)
        if preprocessing_transform is not None:
            desc = 'applying preprocessing'
            if preprocessing_params is None:
                preprocessing_params = {}
            assert preprocessing_params.get('cache_dir') is not None
            self.cache_convert = helpers.Cache(
                preprocessing_transform, preprocessing_params['cache_dir'],
                cache_key=helpers._get_hash(repr(preprocessing_transform)))
            if preprocessing_params.get('use_cuda') is None:
                preprocessing_params['use_cuda'] = False
            num_workers = preprocessing_params.get('num_workers')
            if num_workers == 0:
                with torch.no_grad():
                    for idx in tqdm(range(len(self)), desc=desc, disable=no_progress):
                        name = self._get_attributes(idx)['name']
                        if name not in self.cache_convert.cached_ids:
                            data = self._get_data(idx)
                            self.cache_convert(name, *data)
            else:
                p = Pool(num_workers)
                iterator = p.imap_unordered(
                    _preprocess_task,
                    [(idx, self._get_data, self._get_attributes, self.cache_convert)
                    for idx in range(len(self))])
                for i in tqdm(range(len(self)), desc=desc, disable=no_progress):
                    iterator.next()
        else:
            self.cache_convert = None
        self.transform = transform

    def __getitem__(self, index):
        """Returns the item at index idx. """
        attributes = self._get_attributes(index)
        data = self.cache_convert(attributes['name']) if self.cache_convert is not None else \
               self._get_data(index)
        if self.transform is not None:
            data = self.transform(data)
        return {'data': data, 'attributes': attributes}

    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def _get_attributes(self, index):
        pass

    @abstractmethod
    def _get_data(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

class CombinationDataset(KaolinDataset):
    """Dataset combining multiple list of datasets outputs as inputs
    the output of _get_attributes will be a tuple of all the _get_attributes of the dataset list
    the output of _get_data wiil be a tuple of all the 'data' of the __getitem__ of the dataset list

    Args:
        datasets: list or tuple of KaolinDataset
    """
    def _initialize(self, datasets):
        self.len = len(datasets[0])
        for i, d in enumerate(datasets):
            assert len(d) == self.len, \
                "dataset {} have different length ({})than the first dataset ({})".format(i, len(d), self.len)
        self.datasets = datasets

    def __len__(self):
        return self.len

    def _get_attributes(self, index):
        return (d._get_attributes(index) for d in self.datasets)

    def _get_data(self, index):
        return (d[index]['data'] for d in self.datasets)
