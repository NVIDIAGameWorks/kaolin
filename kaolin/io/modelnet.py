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
from pathlib import Path

from kaolin.io.dataset import KaolinDataset
from kaolin.io.off import import_mesh


class ModelNet(KaolinDataset):
    r"""Dataset class for the ModelNet dataset.

    The `__getitem__` method will return a `KaolinDatasetItem`, with its data field 
    containing a namedtuple returned by :func:`kaolin.io.off.import_mesh`.

    Args:
        root (str): Path to the base directory of the ModelNet dataset.
        split (str): Split to load ('train' vs 'test', default: 'train').
        categories (list): 
            List of categories to load. If None is provided, 
            all categories will be loaded. (default: None).
    """
    def __init__(self, root, categories=None, split='train'):
        assert split in ['train', 'test'], f'Split must be either train or test ,but got {split}.'

        self.root = Path(root)
        self.paths = []
        self.labels = []

        if not os.path.exists(root):
            raise ValueError(f'ModelNet was not found at "{root}.')

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
