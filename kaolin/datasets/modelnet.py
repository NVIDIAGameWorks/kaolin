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

from typing import Iterable, Optional

import torch
import os
from glob import glob

from kaolin.rep.TriangleMesh import TriangleMesh


class ModelNet(object):
    r"""Dataset class for the ModelNet dataset.

    Args:
        root (str): Path to the base directory of the ModelNet dataset.
        split (str, optional): Split to load ('train' vs 'test',
            default: 'train').
        categories (iterable, optional): List of categories to load
            (default: ['chair']).

    Examples:
        >>> dataset = ModelNet(root='data/ModelNet')
        >>> train_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=8)
        >>> obj, label = next(iter(train_loader))
    """

    def initialize(self, root: str,
                   split: Optional[str] = 'train',
                   categories: Optional[Iterable] = ['bed']):
        """Initialize the dataset.

        Args:
            root (str): Path to the base directory of the ModelNet dataset.
            split (str, optional): Split to load ('train' vs 'test',
                default: 'train').
            categories (iterable, optional): List of categories to load
                (default: ['chair']).
        """

        assert split.lower() in ['train', 'test']

        self.root = root
        self.categories = categories
        self.names = []
        self.filepaths = []
        self.cat_idxs = []

        if not os.path.exists(root):
            raise ValueError('ModelNet was not found at "{0}".'.format(root))

        available_categories = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]

        for cat_idx, category in enumerate(categories):
            assert category in available_categories, 'object class {0} not in list of available classes: {1}'.format(
                category, available_categories)

            cat_paths = glob(os.path.join(root, category, split.lower(), '*.off'))

            self.cat_idxs += [cat_idx] * len(cat_paths)
            self.names += [os.path.splitext(os.path.basename(cp))[0] for cp in cat_paths]
            self.filepaths += cat_paths

    def __len__(self):
        return len(self.names)

    def _get_data(self, index):
        data = TriangleMesh.from_off(self.filepaths[index])
        data.to(self.device)
        if self.transform:
            data = self.transform(data)

        return data

    def _get_attributes(self, index):
        category = torch.tensor(self.cat_idxs[index], dtype=torch.long, device=self.device)
        return {
            'category': category,
        }
