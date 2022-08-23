# Copyright (c) 2019,20-21 NVIDIA CORPORATION & AFFILIATES.
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
import warnings
from pathlib import Path

from torch.utils.data import Dataset

from kaolin.io.dataset import KaolinDataset, KaolinDatasetItem
from kaolin.io.off import import_mesh


class ModelNet(Dataset):
    r"""Dataset class for the ModelNet dataset.

    The `__getitem__` method will return:

        * if output_dict=True: a dictionary with the following key-value pairs:

            * 'mesh': containing a namedtuple returned by :func:`kaolin.io.off.import_mesh`.
            * 'name': the model name (i.e the subfolder name)
            * 'path': the full path to the .off
            * 'label': the category of the model

        * if output_dict=False (deprecated): a :obj:`KaolinDatasetItem` with the fields:

            * ``data``: containing a namedtuple returned by :func:`kaolin.io.off.import_mesh`.
            * ``attributes``: containing a dictionary with the following key-value pairs:

                * 'name': the model name (i.e the subfolder name)
                * 'path': the full path to the .off
                * 'label': the category of the model

    .. deprecated:: 0.13.0
       output_dict=False is deprecated.
       Datasets should always output a dictionary to be compatible with :class:`ProcessedDataset`.

    Args:
        root (str): Path to the base directory of the ModelNet dataset.
        categories (list): 
            List of categories to load. Default: all categories available.
        split (str): Split to load ('train' vs 'test', default: 'train').
        transform (Callable):
            A function/transform that takes in a dictionary or :class:`KaolinDatasetItem`
            and returns a transformed version.
        output_dict (bool):
            If True, __getitem__ output a dictionary, else :class:`KaolinDatasetItem` (deprecated)
            Default: False.
    """
    def __init__(self, root, categories=None, split='train', transform=None, output_dict=False):

        assert split in ['train', 'test'], f'Split must be either train or test, but got {split}.'

        if not os.path.exists(root):
            raise ValueError(f'ModelNet was not found at "{root}.')
        self.root = Path(root)
        self.transform = transform
        self.paths = []
        self.labels = []

        if not output_dict:
            warnings.warn("output_dict=False is deprecated, "
                          "datasets __getitem__ should always output a dictionary "
                          "to be compatible with :func:`ProcessedDatasetV2`",
                          DeprecationWarning, stacklevel=2)
        self.output_dict = output_dict

        all_categories = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]

        # If categories is None, load all categories
        if categories is None:
            categories = all_categories

        for idx, category in enumerate(categories):
            assert category in all_categories, f'Object class {category} not in \
                                                 list of available classes: {all_categories}'

            model_paths = sorted((self.root / category / split.lower()).glob('*'))

            self.paths += model_paths
            self.labels += [category] * len(model_paths)


        self.names = [os.path.join(p.parent.name, p.name) for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.output_dict:
            output = {
                'mesh': self.get_data(index),
                **self.get_attributes(index)
            }
        else:
            output = KaolinDatasetItem(
                data=self.get_data(index),
                attributes=self.get_attributes(index)
            )

        if self.transform is not None:
            output = self.transform(output)

        return output


    def get_data(self, index):
        obj_location = self.paths[index]
        mesh = import_mesh(str(obj_location), with_face_colors=True)
        return mesh

    def get_attributes(self, index):
        attributes = {
            'name': self.names[index],
            'path': self.paths[index],
            'label': self.labels[index]
        }
        return attributes

    def get_cache_key(self, index):
        return self.names[index]
