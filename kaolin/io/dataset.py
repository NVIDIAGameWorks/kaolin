# Copyright (c) 2020,21 NVIDIA CORPORATION & AFFILIATES.. All rights reserved.
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

import hashlib
from abc import abstractmethod
from collections import namedtuple
from pathlib import Path

import torch
from torch.multiprocessing import Pool
from torch.utils.data import Dataset
from tqdm import tqdm


def _get_hash(x):
    """Generate a hash from a string, or dictionary.
    """
    if isinstance(x, dict):
        x = tuple(sorted(pair for pair in x.items()))

    return hashlib.md5(bytes(repr(x), 'utf-8')).hexdigest()


class Cache(object):
    """Caches the results of a function to disk.
    If already cached, data is returned from disk. Otherwise,
    the function is executed. Output tensors are always on CPU device.

    Args:
        func (Callable): The function to cache.
        cache_dir (str or Path): Directory where objects will be cached.
        cache_key (str): The corresponding cache key for this function.
    """

    def __init__(self, func, cache_dir, cache_key):
        self.func = func
        self.cache_dir = Path(cache_dir) / str(cache_key)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cached_ids = set([p.stem for p in self.cache_dir.glob('*')])

    def __call__(self, unique_id: str, *args, **kwargs):
        """Execute self.func if not cached, otherwise, read data from disk.

        Args:
            unique_id (str): The unique id with which to name the cached file.
            *args: The arguments to be passed to self.func.
            **kwargs: The arguments to be passed to self.func.

        Returns:
            Results from self.func.
        """

        fpath = self.cache_dir / f'{unique_id}.p'

        if not fpath.exists():
            output = self.func(*args, **kwargs)
            self._write(output, fpath)
            self.cached_ids.add(unique_id)
        else:
            output = self._read(fpath)

        # Read file to move tensors to CPU.
        return self._read(fpath)

    def _write(self, x, fpath):
        torch.save(x, fpath)

    def _read(self, fpath):
        return torch.load(fpath, map_location='cpu')

    def try_get(self, unique_id: str):
        """Read cache from disk. If not found, raise error.

        Args:
            unique_id (str): The unique id with which to name the cached file.

        Returns:
            Results from self.func if exists on disk.
        """
        fpath = self.cache_dir / f'{unique_id}.p'

        if not fpath.exists():
            raise ValueError(
                'Cache does not exist for key {}'.format(unique_id))
        else:
            output = self._read(fpath)

        # Read file to move tensors to CPU.
        return self._read(fpath)


def _preprocess_task(args):
    torch.set_num_threads(1)
    with torch.no_grad():
        idx, get_data, get_cache_key, cache_transform = args
        key = get_cache_key(idx)
        data = get_data(idx)
        cache_transform(key, data)


def _get_data(dataset, index):
    return dataset.get_data(index) if hasattr(dataset, 'get_data') else \
        dataset[index]


def _get_attributes(dataset, index):
    return dataset.get_attributes(index) if hasattr(dataset, 'get_attributes') \
        else {}


def _get_cache_key(dataset, index):
    return dataset.get_cache_key(index) if hasattr(dataset, 'get_cache_key') \
        else str(index)


KaolinDatasetItem = namedtuple('KaolinDatasetItem', ['data', 'attributes'])

class KaolinDataset(Dataset):
    """A dataset supporting the separation of data and attributes, and combines
    them in its `__getitem__`.
    The return value of `__getitem__` will be a named tuple containing the
    return value of both `get_data` and `get_attributes`.
    The difference between `get_data` and `get_attributes` is that data are able
    to be transformed or preprocessed (such as using `ProcessedDataset`), while
    attributes are generally not.
    """

    def __getitem__(self, index):
        """Returns the item at the given index.
        Will contain a named tuple of both data and attributes.
        """
        attributes = self.get_attributes(index)
        data = self.get_data(index)
        return KaolinDatasetItem(data=data, attributes=attributes)

    @abstractmethod
    def get_data(self, index):
        """Returns the data at the given index."""
        pass

    @abstractmethod
    def get_attributes(self, index):
        """Returns the attributes at the given index.
        Attributes are usually not transformed by wrappers such as
        `ProcessedDataset`.
        """
        pass

    @abstractmethod
    def __len__(self):
        """Returns the number of entries."""
        pass

class ProcessedDataset(KaolinDataset):
    def __init__(self, dataset, preprocessing_transform=None,
                 cache_dir=None, num_workers=None, transform=None,
                 no_progress: bool = False):
        """
        A wrapper dataset that applies a preprocessing transform to a given base
        dataset. The result of the preprocessing transform will be cached to
        disk.

        The base dataset should support a `get_data(idx)` method that returns
        the data to be preprocessed. If such method is not found, then its
        `__getitem__(idx)` method will be used instead.

        The base dataset can optionally support a `get_attributes(idx)` method
        that returns the data that should not be preprocessed. If such method is
        not found, an empty dict will be used as attributes. The `__getitem__`
        of `ProcessedDataset` will contain both the data and the attributes.

        The base dataset can optionally support a `get_cache_key(idx)` method
        that returns the string key to use for caching. If such method is not
        found, the index (as a string) will be used as cache key.

        Note: if CUDA is used in preprocessing, `num_workers` must be set to 0.

        Args:
            dataset (torch.utils.data.Dataset):
                The base dataset to preprocess.
            cache_dir (str):
                Path to the cached preprocessed data. Must be given if
                `preprocessing_transform` is not None.
            num_workers (int):
                Number of process used in parallel for preprocessing (default:
                number of cores)
            preprocessing_transform (Callable):
                Called on the outputs of get_data over the indices from 0 to
                `len(self)` during the construction of the dataset, the
                preprocessed outputs are then cached to 'cache_dir'.
            transform (Callable):
                Called on the preprocessed data at `__getitem__`.
                The result of this function is not cached, unlike
                `preprocessing_transform`.
            no_progress (bool): Disable tqdm progress bar for preprocessing.
        """
        # TODO: Consider integrating combination into `ProcessedDataset`.

        self.dataset = dataset
        self.transform = transform

        if preprocessing_transform is not None:
            desc = 'Applying preprocessing'

            assert cache_dir is not None, 'Cache directory is not given'

            self.cache_convert = Cache(
                preprocessing_transform,
                cache_dir=cache_dir,
                cache_key=_get_hash(repr(preprocessing_transform))
            )

            uncached = [idx for idx in range(len(self)) if
                        self.get_cache_key(idx) not in
                        self.cache_convert.cached_ids]
            if len(uncached) > 0:
                if num_workers == 0:
                    with torch.no_grad():
                        for idx in tqdm(range(len(self)), desc=desc,
                                        disable=no_progress):
                            key = self.get_cache_key(idx)
                            data = self._get_base_data(idx)
                            self.cache_convert(key, data)
                else:
                    p = Pool(num_workers)
                    iterator = p.imap_unordered(
                        _preprocess_task,
                        [(idx, self._get_base_data, self.get_cache_key,
                          self.cache_convert)
                         for idx in uncached])
                    for i in tqdm(range(len(uncached)), desc=desc,
                                  disable=no_progress):
                        next(iterator)
        else:
            self.cache_convert = None

    def __len__(self):
        return len(self.dataset)

    def get_data(self, index):
        """Returns the data at the given index. """
        print(index)
        data = (self._get_base_data(index) if self.cache_convert is None else
                self.cache_convert.try_get(self.get_cache_key(index)))

        if self.transform is not None:
            data = self.transform(data)

        return data

    def _get_base_data(self, index):
        return _get_data(self.dataset, index)

    def get_attributes(self, index):
        return _get_attributes(self.dataset, index)

    def get_cache_key(self, index):
        return _get_cache_key(self.dataset, index)


class CombinationDataset(KaolinDataset):
    """Dataset combining a list of datasets into a unified dataset object.

    Useful when multiple output representations are needed from a common base
    representation (Eg. when a mesh is to be served as both a pointcloud and a
    voxelgrid, etc.)

    The output of get_attributes will be a tuple of all the get_attributes of
    the dataset list.

    The output of get_data will be a tuple of all the get_data of the dataset
    list.

    If a dataset does not have get_data, `__getitem__` will be used instead.

    The output of get_cache_key will be the cache key of the first dataset. If
    that dataset does not provide `get_cache_key`, the index will be used
    instead.

    Args:
        datasets: list or tuple of datasets
    """

    def __init__(self, datasets):
        assert len(datasets) > 0, "Must provide at least one dataset"

        self.len = len(datasets[0])
        for i, d in enumerate(datasets):
            assert len(d) == self.len, \
                f"All datasets must have the same length. Invalid length at index {i} (expected: {self.len}, got: {len(d)})"
        self.datasets = datasets

    def __len__(self):
        return self.len

    def get_data(self, index):
        return tuple(_get_data(d, index) for d in self.datasets)

    def get_attributes(self, index):
        return tuple(_get_attributes(d, index) for d in self.datasets)

    def get_cache_key(self, index):
        return _get_cache_key(self.datasets[0], index)
