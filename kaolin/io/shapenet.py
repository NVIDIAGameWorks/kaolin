# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

import warnings
from pathlib import Path

from kaolin.io.dataset import KaolinDataset
from kaolin.io.obj import import_mesh

# Synset to Label mapping (for ShapeNet core classes)
synset_to_label = {
    '04379243': 'table',
    '03211117': 'monitor',
    '04401088': 'phone',
    '04530566': 'watercraft',
    '03001627': 'chair',
    '03636649': 'lamp',
    '03691459': 'speaker',
    '02828884': 'bench',
    '02691156': 'plane',
    '02808440': 'bathtub',
    '02871439': 'bookcase',
    '02773838': 'bag',
    '02801938': 'basket',
    '02880940': 'bowl',
    '02924116': 'bus',
    '02933112': 'cabinet',
    '02942699': 'camera',
    '02958343': 'car',
    '03207941': 'dishwasher',
    '03337140': 'file',
    '03624134': 'knife',
    '03642806': 'laptop',
    '03710193': 'mailbox',
    '03761084': 'microwave',
    '03928116': 'piano',
    '03938244': 'pillow',
    '03948459': 'pistol',
    '04004475': 'printer',
    '04099429': 'rocket',
    '04256520': 'sofa',
    '04554684': 'washer',
    '04090263': 'rifle',
    '02946921': 'can'
}

# Label to Synset mapping (for ShapeNet core classes)
label_to_synset = {v: k for k, v in synset_to_label.items()}


def _convert_categories(categories):
    if categories is None:
        synset = [value for key, value in label_to_synset.items()]
    else:
        if not (c in synset_to_label.keys() + label_to_synset.keys()
                for c in categories):
            warnings.warn('Some or all of the categories requested are not part of \
                ShapeNetCore. Data loading may fail if these categories are not avaliable.')
        synsets = [label_to_synset[c] if c in label_to_synset.keys()
                else c for c in categories]
    return synsets


class ShapeNet(KaolinDataset):
    r"""ShapeNetV1 Dataset class for meshes.

    The `__getitem__` method will return a `KaolinDatasetItem`, with its `data`
    field containing a `kaolin.io.obj.ObjMesh`.

    Args:
        root (str): path to ShapeNet root directory
        categories (list): List of categories to load from ShapeNet. This list may
                           contain synset ids, class label names (for ShapeNetCore classes),
                           or a combination of both.
        train (bool): If True, return the training set, otherwise the test set
        split (float): fraction of the dataset to be used for training (>=0 and <=1)
    """

    def __init__(self, root: str, categories: list, train: bool = True,
                 split: float = .7):
        """
        Args:
            root (str): path to ShapeNet root directory
            categories (list):
                List of categories to load from ShapeNet. This list may
                contain synset ids, class label names (for ShapeNetCore classes),
                or a combination of both.
            train (bool): If True, return the training set, otherwise the test set
            split (float): fraction of the dataset to be used for training (>=0 and <=1)
        """

        self.root = Path(root)
        self.paths = []
        self.synset_idxs = []
        self.synsets = _convert_categories(categories)
        self.labels = [synset_to_label[s] for s in self.synsets]

        # loops through desired classes
        for i in range(len(self.synsets)):
            syn = self.synsets[i]
            class_target = self.root / syn
            if not class_target.exists():
                raise ValueError(
                    'Class {0} ({1}) was not found at location {2}.'.format(
                        syn, self.labels[i], str(class_target)))

            # find all objects in the class
            models = sorted(class_target.glob('*'))
            stop = int(len(models) * split)
            if train:
                models = models[:stop]
            else:
                models = models[stop:]
            self.paths += models
            self.synset_idxs += [i] * len(models)

        self.names = [p.name for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def get_data(self, index):
        obj_location = self.paths[index] / 'model.obj'
        mesh = import_mesh(str(obj_location))
        return mesh

    def get_attributes(self, index):
        synset_idx = self.synset_idxs[index]
        attributes = {
            'name': self.names[index],
            'path': self.paths[index] / 'model.obj',
            'synset': self.synsets[synset_idx],
            'label': self.labels[synset_idx]
        }
        return attributes

    def get_cache_key(self, index):
        return self.names[index]
