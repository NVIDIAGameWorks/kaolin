# Copyright (c) 2020,21-22 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

import os
import hashlib
import warnings
import copy
from collections.abc import Sequence
from abc import abstractmethod
from collections import namedtuple
from pathlib import Path
import shutil
from tqdm import tqdm

import torch
from torch.multiprocessing import Pool
from torch.utils.data import Dataset

from  ..utils.testing import contained_torch_equal

def _parallel_save_task(args):
    torch.set_num_threads(1)
    return _save_task(*args)

def _save_task(cache_dir, idx, getitem, to_save_on_disk, to_not_save):
    with torch.no_grad():
        if len(to_save_on_disk) > 0:
            data_dir = cache_dir / str(idx)
            data_dir.mkdir(exist_ok=True)
        data = getitem(idx)
        outputs = {}
        for k, v in data.items():
            if k in to_save_on_disk:
                torch.save(v, data_dir / f'{k}.pt')
            elif k not in to_not_save:
                outputs[k] = v
        return outputs

def _get_saving_actions(dataset, cache_dir, save_on_disk=False,
                        force_overwrite=False, ignore_diff_error=False):
    size = len(dataset)
    # Is there anything to save on disk?
    if isinstance(save_on_disk, bool):
        any_save_on_disk = save_on_disk
    elif isinstance(save_on_disk, Sequence):
        any_save_on_disk = len(save_on_disk) > 0
    else:
        raise TypeError("save_on_disk must be a boolean or a sequence of str")

    # We need to query the data from dataset[0] for sanity check
    # of saved files and arguments such as `save_on_disk`
    _data = dataset[0]
    if not isinstance(_data, dict):
        raise TypeError("the dataset.__getitem__ must output a dictionary")

    # Convert save_on_disk to a set of strings
    if isinstance(save_on_disk, bool):
        save_on_disk = set(_data.keys()) if save_on_disk else set()
    else:
        save_on_disk = set(save_on_disk)

    to_save_on_ram = set(_data.keys()).difference(save_on_disk)
    to_not_save = set() # Values that are already saved on disk
    to_save_on_disk = set() # Values that will be force stored on disk

    if any_save_on_disk:
        if cache_dir is None:
            raise ValueError("cache_dir should be provided with save_on_disk")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_ids = list(p.stem for p in cache_dir.glob(r'*'))

        # Check that the keys on save_on_disk are actual outputs from preprocessing
        for k in save_on_disk:
            if k not in _data.keys():
                raise ValueError(f"the dataset doesn't provide an output field '{k}'")

        if force_overwrite:
            if len(cached_ids) != len(dataset):
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            to_save_on_disk = save_on_disk
        else:
            # If the number of folder (len(dataset)) is different, 
            # something is probably wrong with the existing data
            #   note: reporting error directory with POSIX path to ease regex matching without raising encoding errors due to Windows backslashes
            if len(cached_ids) > 0 and len(cached_ids) != len(dataset):
                raise RuntimeError(f"{len(cached_ids)} files already exist on "
                                   f"{cache_dir.resolve().as_posix()} this dataset as {len(dataset)} files "
                                   "so caching is too ambiguous and error-prone "
                                   "please force rewriting by setting 'force_overwrite'")

            # We accept that the cache has partial values defined,
            # for instance if we add/remove a key from save_on_disk
            # TODO(cfujitsang): need to check if avoiding store isn't over-optimization
            #                   at the cost of potential user errors
            #                   since we are not avoiding to run the preprocessing
            for k, v in _data.items():
                if k in save_on_disk:
                    path:Path = cache_dir / '0' / f'{k}.pt'
                    # note: reporting error directory with POSIX path to ease regex matching without raising encoding errors due to Windows backslashes
                    path_str = path.resolve().as_posix()
                    if path.exists(): # There is already a file for a given key 
                        # Is the value stored the same than the one from the data?
                        assert ignore_diff_error or contained_torch_equal(v, torch.load(path)), \
                            f"file '{path_str}' is different than " \
                            "its matching field from the input dataset, set 'force_overwriting' " \
                            "to True to overwrite the files cached."
                        to_not_save.add(k)
                    else:
                        to_save_on_disk.add(k)
    return to_save_on_disk, to_save_on_ram, to_not_save
    
class CachedDataset(Dataset):
    """A wrapper dataset that caches the data to disk or RAM depending on
    ``save_on_disk``.

    For all ``dataset[i]`` with ``i`` from 0 to ``len(dataset)`` the output is store on RAM
    or disk depending on ``save_on_disk``.

    The base dataset or the ``preprocessing_transform`` if defined,
    should have a ``__getitem__(idx)`` method that returns a dictionary.

    .. note::
    
        if CUDA is used in preprocessing, ``num_workers`` must be set to 0.

    Args:
        dataset (torch.utils.data.Dataset or Sequence):
            The base dataset to use.
        cache_dir (optional, str):
            Path where the data must be saved. Must be given if
            ``save_on_disk`` is not False.
        save_on_disk (optional, bool or Sequence[str]):
            If True all the preprocessed outputs are stored on disk,
            if False all the preprocessed outputs are stored on RAM,
            if it's a sequence of strings then all the corresponding fields
            are stored on disk.
        num_workers (optional, int):
            Number of process used in parallel for preprocessing.
            Default: 0 (run in main process).
        force_overwrite (optional, bool):
            If True, force overwriting on disk even if files already exist.
            Default: False.
        cache_at_runtime (optional, bool):
            If True, instead of preprocessing everything at construction
            of the dataset, each new ``__getitem__`` will cache if necessary.
            Default: False.
        progress_message (optional, str):
            Message to be displayed during preprocessing.
            This is unuse with cache_at_runtime=True.
            Default: don't show any message.
        transform (optional, Callable):
            If defined, called on the data at ``__getitem__``.
            The result of this function is not cached.
            Default: don't apply any transform.
    """

    def __init__(self, dataset, cache_dir=None, save_on_disk=False,
                 num_workers=0, force_overwrite=False, cache_at_runtime=False,
                 progress_message=None, ignore_diff_error=False, transform=None):
        self.size = len(dataset)
        self.transform = transform
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.to_save_on_disk, self.to_save_on_ram, self.to_not_save = _get_saving_actions(
            dataset,
            self.cache_dir,
            save_on_disk=save_on_disk,
            force_overwrite=force_overwrite,
            ignore_diff_error=ignore_diff_error
        )
        self.on_disk = self.to_save_on_disk.union(self.to_not_save)

        if len(self.to_save_on_ram) == 0 and len(self.to_save_on_disk) == 0:
            # If nothing is to be stored on RAM and everything is already stored on disk
            # there is no point in running the preprocessing
            self.data = [{} for i in range(len(self))]
        elif cache_at_runtime:
            # __getitem__(idx) will execute the preprocessing task
            # at runtime in case self.data[idx] is None.
            self.data = [None] * len(self)
            self.dataset = dataset
            self.num_not_saved = len(self)
        else:
            self.data = []
            # Run the preprocessing + saving
            try:
               if num_workers > 0:
                   # With multiprocessing
                   p = Pool(num_workers)
                   try:
                       iterator = p.imap(
                           _parallel_save_task, [(
                               self.cache_dir, idx, dataset.__getitem__,
                               self.to_save_on_disk, self.to_not_save,
                           ) for idx in range(len(self))])
                       for _ in tqdm(range(len(self)), desc=progress_message):
                           self.data.append(next(iterator))
                   finally:
                       p.close()
                       p.join()
               else:
                   for idx in tqdm(range(len(self)), desc=progress_message):
                       self.data.append(_save_task(
                           self.cache_dir,
                           idx,
                           dataset.__getitem__,
                           self.to_save_on_disk,
                           self.to_not_save
                       ))
            except Exception as e:
                # Cleaning if the preprocessing is returning an error
                # there is not point in keeping the files that have been
                # generated since they are most likely wrong
                self._clean_cache_dir()
                raise e

    def _clean_cache_dir(self):
        to_remove_paths = set()
        if len(self.to_save_on_disk) > 0:
            for k in self.to_save_on_disk:
                to_remove_paths.update(set(self.cache_dir.glob(f'[0-9]*/{k}.pt')))
            if set(self.cache_dir.glob('[0-9]*/*.pt')) == set(to_remove_paths):
                shutil.rmtree(self.cache_dir)
            else:
                for k in self.to_save_on_disk:
                    for path in self.cache_dir.glob(f'[0-9]*/{k}.pt'):
                        os.remove(path)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        outputs = copy.copy(self.data[idx])
        if outputs is None:
            try:
                self.data[idx] = _save_task(
                    self.cache_dir,
                    idx,
                    self.dataset.__getitem__,
                    self.to_save_on_disk,
                    self.to_not_save
                )
            except Exception as e:
                # Cleaning if the preprocessing is returning an error
                # there is not point in keeping the files that have been
                # generated since they are most likely wrong
                self._clean_cache_dir()
                raise e
            self.num_not_saved -= 1
            if self.num_not_saved == 0:
                # This is to save memory
                self.dataset = None
            outputs = copy.copy(self.data[idx])

        for k in self.on_disk:
             outputs[k] = torch.load(self.cache_dir / str(idx) / f'{k}.pt')

        if self.transform is not None:
            outputs = self.transform(outputs)
        return outputs

### DEPRECATED ###

def _get_hash(x):
    """Generate a hash from a string, or dictionary.
    """
    if isinstance(x, dict):
        x = tuple(sorted(pair for pair in x.items()))

    return hashlib.md5(bytes(repr(x), 'utf-8')).hexdigest()

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

class Cache(object):
    """Caches the results of a function to disk.
    If already cached, data is returned from disk. Otherwise,
    the function is executed. Output tensors are always on CPU device.

    .. deprecated:: 0.13.0
       :class:`Cache` is deprecated.

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
        fpath.parent.mkdir(parents=True, exist_ok=True)
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

### DEPRECATED ###

KaolinDatasetItem = namedtuple('KaolinDatasetItem', ['data', 'attributes'])

class KaolinDataset(Dataset):
    """A dataset supporting the separation of data and attributes, and combines
    them in its `__getitem__`.
    The return value of `__getitem__` will be a named tuple containing the
    return value of both `get_data` and `get_attributes`.
    The difference between `get_data` and `get_attributes` is that data are able
    to be transformed or preprocessed (such as using `ProcessedDataset`), while
    attributes are generally not.

    .. deprecated:: 0.13.0
       :class:`KaolinDataset` is deprecated.
       Datasets should always output a dictionary to be compatible with :class:`ProcessedDataset`.
    """
    def __getitem__(self, index):
        """Returns the item at the given index.
        Will contain a named tuple of both data and attributes.
        """
        return KaolinDatasetItem(
            data=self.get_data(index),
            attributes=self.get_attributes(index)
        )

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
        
        .. deprecated:: 0.13.0
           :class:`ProcessedDataset` is deprecated. See :class:`ProcessedDatasetV2`.

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
        warnings.warn("ProcessedDataset is deprecated, "
                      "please use ProcessedDatasetV2",
                      DeprecationWarning, stacklevel=2)

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
                    try:
                        iterator = p.imap_unordered(
                            _preprocess_task,
                            [(idx, self._get_base_data, self.get_cache_key,
                              self.cache_convert)
                             for idx in uncached])
                        for i in tqdm(range(len(uncached)), desc=desc,
                                      disable=no_progress):
                            next(iterator)
                    finally:
                        p.close()
                        p.join()
        else:
            self.cache_convert = None

    def __len__(self):
        return len(self.dataset)

    def get_data(self, index):
        """Returns the data at the given index. """
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
        
    .. deprecated:: 0.13.0
       :class:`CombinationDataset` is deprecated. See :class:`ProcessedDatasetV2`.

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
