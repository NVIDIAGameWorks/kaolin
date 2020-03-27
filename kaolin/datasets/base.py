# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from abc import abstractmethod
from tqdm import tqdm

import torch
from torch.multiprocessing import Pool
from torch.utils.data import Dataset

from kaolin import helpers


def _preprocess_task(args):
    torch.set_num_threads(1)
    with torch.no_grad():
        idx, get_data, get_attributes, cache_transform = args
        name = get_attributes(idx)['name']
        if name not in cache_transform.cached_ids:
            data = get_data(idx)
            cache_transform(name, data)


class KaolinDatasetMeta(type):
    def __new__(metacls, cls_name, base_cls, class_dict):
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
        return type.__new__(metacls, cls_name, base_cls, class_dict)


class KaolinDataset(Dataset, metaclass=KaolinDatasetMeta):
    """
    Abstract class for dataset with handling of multiprocess or cuda preprocessing.

    A KaolinDataset children class will need the above implementation:
       1) initialize:
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
            desc = 'Applying preprocessing'
            if preprocessing_params is None:
                preprocessing_params = {}

            cache_dir = preprocessing_params.get('cache_dir')
            assert cache_dir is not None, 'Cache directory is not given'

            self.cache_convert = helpers.Cache(
                preprocessing_transform,
                cache_dir=cache_dir,
                cache_key=helpers._get_hash(repr(preprocessing_transform))
            )

            use_cuda = preprocessing_params.get('use_cuda', False)

            num_workers = preprocessing_params.get('num_workers')

            if num_workers == 0:
                with torch.no_grad():
                    for idx in tqdm(range(len(self)), desc=desc, disable=no_progress):
                        name = self._get_attributes(idx)['name']
                        if name not in self.cache_convert.cached_ids:
                            data = self._get_data(idx)
                            self.cache_convert(name, data)

            else:
                p = Pool(num_workers)
                iterator = p.imap_unordered(
                    _preprocess_task,
                    [(idx, self._get_data, self._get_attributes, self.cache_convert)
                     for idx in range(len(self))])

                for i in tqdm(range(len(self)), desc=desc, disable=no_progress):
                    next(iterator)

        else:
            self.cache_convert = None

        self.transform = transform

    def __getitem__(self, index):
        """Returns the item at index idx. """
        attributes = self._get_attributes(index)
        data = (self._get_data(index) if self.cache_convert is None else
                self.cache_convert(attributes['name']))

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
    """Dataset combining a list of datasets into a unified dataset object.
    Useful when multiple output representations are needed from a common base representation
    (Eg. when a mesh is to be served as both a pointcloud and a voxelgrid, etc.)
    the output of _get_attributes will be a tuple of all the _get_attributes of the dataset list
    the output of _get_data wiil be a tuple of all the 'data' of the __getitem__ of the dataset list

    Args:
        datasets: list or tuple of KaolinDataset
    """

    def initialize(self, datasets):
        self.len = len(datasets[0])
        for i, d in enumerate(datasets):
            assert len(d) == self.len, \
                f"All datasets must have the same length. Invalid length at index {i} (expected: {self.len}, got: {len(d)})"
        self.datasets = datasets

    def __len__(self):
        return self.len

    def _get_attributes(self, index):
        return (d._get_attributes(index) for d in self.datasets)

    def _get_data(self, index):
        return (d[index]['data'] for d in self.datasets)
