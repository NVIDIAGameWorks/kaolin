# Copyright (c) 2019,20-21-22 NVIDIA CORPORATION & AFFILIATES.
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
from kaolin.io.obj import import_mesh, ignore_error_handler

# Synset to Label mapping (for ShapeNet core classes)
synset_to_labels = {
    '04379243': ['table'],
    '03211117': ['display', 'video display'],
    '04401088': ['telephone', 'phone', 'telephone set'],
    '04530566': ['vessel', 'watercraft'],
    '03001627': ['chair'],
    '03636649': ['lamp'],
    '03691459': ['loudspeaker', 'speaker', 'speaker unit', 'loudspeaker system', 'speaker system'],
    '02828884': ['bench'],
    '02691156': ['airplane', 'aeroplane', 'plane'],
    '02808440': ['bathtub', 'bathing tub', 'bath', 'tub'],
    '02871439': ['bookshelf'],
    '02773838': ['bag', 'traveling bag', 'travelling bag', 'grip', 'suitcase'],
    '02801938': ['basket', 'handbasket'],
    '02880940': ['bowl'],
    '02924116': ['bus', 'autobus', 'coach', 'charabanc', 'double-decker', 'jitney',
                 'motorbus', 'motorcoach', 'omnibus', 'passenger vehi'],
    '02933112': ['cabinet'],
    '02942699': ['camera', 'photographic camera'],
    '02958343': ['car', 'auto', 'automobile', 'machine', 'motorcar'],
    '03207941': ['dishwasher', 'dish washer', 'dishwashing machine'],
    '03337140': ['file', 'file cabinet', 'filing cabinet'],
    '03624134': ['knife'],
    '03642806': ['laptop', 'laptop computer'],
    '03710193': ['mailbox', 'letter box'],
    '03761084': ['microwave', 'microwave oven'],
    '03928116': ['piano', 'pianoforte', 'forte-piano'],
    '03938244': ['pillow'],
    '03948459': ['pistol', 'handgun', 'side arm', 'shooting iron'],
    '04004475': ['printer', 'printing machine'],
    '04099429': ['rocket', 'projectile'],
    '04256520': ['sofa', 'couch', 'lounge'],
    '04554684': ['washer', 'automatic washer', 'washing machine'],
    '04090263': ['rifle'],
    '02946921': ['can', 'tin', 'tin can'],
    '04330267': ['stove'],
    '02843684': ['birdhouse'],
    '03513137': ['helmet'],
    '02992529': ['cellular telephone', 'cellular phone', 'cellphone', 'cell', 'mobile phone'],
    '03991062': ['pot', 'flowerpot'],
    '04074963': ['remote control', 'remote'],
    '03790512': ['motorcycle', 'bike'],
    '04225987': ['skateboard'],
    '03593526': ['jar'],
    '02954340': ['cap'],
    '03467517': ['guitar'],
    '04460130': ['tower'],
    '03759954': ['microphone', 'mike'],
    '03325088': ['faucet', 'spigot'],
    '03797390': ['mug'],
    '03046257': ['clock'],
    '02747177': ['ashcan', 'trash can', 'garbage can', 'wastebin', 'ash bin', 'ash-bin',
                 'ashbin', 'dustbin', 'trash barrel', 'trash bin'],
    '02818832': ['bed'],
    '03085013': ['computer keyboard', 'keypad'],
    '02876657': ['bottle'],
    '04468005': ['train', 'railroad train'],
    '03261776': ['earphone', 'earpiece', 'headphone', 'phone'],
    '02834778': ['bicycle', 'bike', 'wheel', 'cycle'],
    '02858304': ['boat']
}

# Label to Synset mapping (for ShapeNet core classes)
label_to_synset = {label: synset for synset, labels in synset_to_labels.items() for label in labels}

def _convert_categories(categories):
    for c in categories:
        if c not in synset_to_labels.keys() and c not in label_to_synset.keys():
            warnings.warn('Some or all of the categories requested are not part of \
                ShapeNetCore. Data loading may fail if these categories are not avaliable.')
    synsets = [label_to_synset[c] if c in label_to_synset.keys()
               else c for c in categories]
    return synsets

class ShapeNetV1(Dataset):
    r"""ShapeNetV1 Dataset class for meshes.

    The `__getitem__` method will return:

        * if output_dict=True: a dictionary with the following key-value pairs:

            * 'mesh': containing a :class:`kaolin.rep.SurfaceMesh` returned by :func:`kaolin.io.obj.import_mesh`.
            * 'name': the model name (i.e the subfolder name)
            * 'path': the full path to the .off
            * 'synset': the synset associated to the category
            * 'labels': the labels associated to the category (see ``synset_to_labels``)

        * if output_dict=False (deprecated): a :class:`KaolinDatasetItem` with the fields:

            * ``data``: containing a :class:`kaolin.rep.SurfaceMesh` returned by :func:`kaolin.io.obj.import_mesh`.
            * ``attributes``: containing a dictionary with the following key-value pairs:

                * 'name': the model name (i.e the subfolder name)
                * 'path': the full path to the .off
                * 'synset': the synset associated to the category
                * 'labels': the labels associated to the category (see ``synset_to_labels``)

    .. deprecated:: 0.13.0
       output_dict=False is deprecated.
       Datasets should always output a dictionary to be compatible with :class:`ProcessedDataset`.

    Args:
        root (str): path to ShapeNet root directory
        categories (list): List of categories to load from ShapeNet. This list may
                           contain synset ids, class label names (for ShapeNetCore classes),
                           or a combination of both. Default: all supported categories.
        train (bool):
            If True, return the training set, otherwise the test set.
            Default: True.
        split (float):
            Fraction of the dataset to be used for training (>=0 and <=1).
            Default: 0.7
        with_materials (bool):
            If True, load and return materials. Default: True.
        transform (Callable):
            A function/transform that takes in a dictionary or :class:`KaolinDatasetItem`
            and returns a transformed version.
        output_dict (bool):
            If True, __getitem__ output a dictionary, else :class:`KaolinDatasetItem` (deprecated)
            Default: False.
    """
    SUPPORTED_SYNSETS = {
        '04330267',
        '02843684',
        '02871439',
        '03513137',
        '02880940',
        '03001627',
        '03761084',
        '02992529',
        '03991062',
        '04074963',
        '03790512',
        '04225987',
        '02924116',
        '04379243',
        '03593526',
        '02828884',
        '03938244',
        '02954340',
        '03467517',
        '02834778',
        '04099429',
        '04460130',
        '03759954',
        '02858304',
        '03691459',
        '04256520',
        '04530566',
        '04004475',
        '02691156',
        '03211117',
        '03948459',
        '03636649',
        '03325088',
        '02946921',
        '02958343',
        '03797390',
        '02808440',
        '03046257',
        '02747177',
        '03928116',
        '02818832',
        '03337140',
        '04090263',
        '04401088',
        '03207941',
        '03624134',
        '02773838',
        '03085013',
        '03642806',
        '03710193',
        '02933112',
        '02876657',
        '04468005',
        '02801938',
        '03261776',
        '04554684',
        '02942699'
    }

    def __init__(self, root: str, categories: list = None, train: bool = True,
                 split: float = .7, with_materials=True, transform=None, output_dict=False):
        self.root = Path(root)
        self.transform = transform
        self.paths = []
        self.synset_idxs = []
        if categories is None:
            self.synsets = sorted(self.SUPPORTED_SYNSETS)
        else:
            self.synsets = _convert_categories(categories)
            for s in self.synsets:
                assert s in self.SUPPORTED_SYNSETS, \
                    f"{s} is not supported in ShapeNetV1"
        self.labels = [synset_to_labels[s] for s in self.synsets]
        self.with_materials = with_materials
        if not output_dict:
            warnings.warn("output_dict=False is deprecated, "
                          "datasets __getitem__ should always output a dictionary "
                          "to be compatible with :func:`ProcessedDatasetV2`",
                          DeprecationWarning, stacklevel=2)
        self.output_dict = output_dict

        # loops through desired classes
        for i, syn in enumerate(self.synsets):
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
        obj_location = self.paths[index] / 'model.obj'
        mesh = import_mesh(str(obj_location), with_materials=self.with_materials,
                           error_handler=ignore_error_handler)
        return mesh

    def get_attributes(self, index):
        synset_idx = self.synset_idxs[index]
        attributes = {
            'name': self.names[index],
            'path': self.paths[index] / 'model.obj',
            'synset': self.synsets[synset_idx],
            'labels': self.labels[synset_idx]
        }
        return attributes

    def get_cache_key(self, index):
        return self.names[index]

class ShapeNetV2(Dataset):
    r"""ShapeNetV2 Dataset class for meshes.

    The `__getitem__` method will return:

        * if output_dict=True: a dictionary with the following key-value pairs:

            * 'mesh': containing a :class:`kaolin.rep.SurfaceMesh` returned by :func:`kaolin.io.obj.import_mesh`.
            * 'name': the model name (i.e the subfolder name)
            * 'path': the full path to the .off
            * 'synset': the synset associated to the category
            * 'labels': the labels associated to the category (see ``synset_to_labels``)

        * if output_dict=False (deprecated): a :class:`KaolinDatasetItem` with the fields:

            * ``data``: containing a :class:`kaolin.rep.SurfaceMesh` returned by :func:`kaolin.io.obj.import_mesh`.
            * ``attributes``: containing a dictionary with the following key-value pairs:

                * 'name': the model name (i.e the subfolder name)
                * 'path': the full path to the .off
                * 'synset': the synset associated to the category
                * 'labels': the labels associated to the category (see ``synset_to_labels``)

    .. deprecated:: 0.13.0
       output_dict=False is deprecated.
       Datasets should always output a dictionary to be compatible with :class:`ProcessedDataset`.

    Args:
        root (str): path to ShapeNet root directory
        categories (list):
            List of categories to load from ShapeNet. This list may
            contain synset ids, class label names (for ShapeNetCore classes),
            or a combination of both. Default: all supported categories.
        train (bool):
            If True, return the training set, otherwise the test set.
            Default: True.
        split (float):
            fraction of the dataset to be used for training (>=0 and <=1).
            Default: 0.7
        with_materials (bool):
            If True, load and return materials. Default: True.
        transform (Callable):
            A function/transform that takes in a dictionary or :class:`KaolinDatasetItem`
            and returns a transformed version.
        output_dict (bool):
            If True, __getitem__ output a dictionary, else :class:`KaolinDatasetItem` (deprecated)
            Default: False.
    """

    SUPPORTED_SYNSETS = {
        '04330267',
        '02843684',
        '02871439',
        '03513137',
        '02880940',
        '03001627',
        '03761084',
        '02992529',
        '03991062',
        '04074963',
        '03790512',
        '04225987',
        '02924116',
        '04379243',
        '03593526',
        '02828884',
        '03938244',
        '02954340',
        '03467517',
        '04099429',
        '04460130',
        '03759954',
        '03691459',
        '04256520',
        '04530566',
        '04004475',
        '02691156',
        '03211117',
        '03948459',
        '03636649',
        '03325088',
        '02946921',
        '02958343',
        '03797390',
        '02808440',
        '03046257',
        '02747177',
        '03928116',
        '02818832',
        '03337140',
        '04090263',
        '04401088',
        '03207941',
        '03624134',
        '02773838',
        '03085013',
        '03642806',
        '03710193',
        '02933112',
        '02876657',
        '04468005',
        '02801938',
        '03261776',
        '04554684',
        '02942699'
    }

    def __init__(self, root: str, categories: list = None, train: bool = True,
                 split: float = .7, with_materials=True, transform=None, output_dict=False):
        self.root = Path(root)
        self.transform = transform
        self.paths = []
        self.synset_idxs = []
        if categories is None:
            self.synsets = list(self.SUPPORTED_SYNSETS)
        else:
            self.synsets = _convert_categories(categories)
            for s in self.synsets:
                assert s in self.SUPPORTED_SYNSETS, \
                    f"{s} is not supported in ShapeNetV2"
        self.labels = [synset_to_labels[s] for s in self.synsets]
        self.with_materials = with_materials
        if not output_dict:
            warnings.warn("output_dict=False is deprecated, "
                          "datasets __getitem__ should always output a dictionary "
                          "to be compatible with :func:`ProcessedDatasetV2`",
                          DeprecationWarning, stacklevel=2)
        self.output_dict = output_dict

        # loops through desired classes
        for i, syn in enumerate(self.synsets):
            class_target = self.root / syn
            if not class_target.exists():
                raise ValueError(
                    'Class {0} ({1}) was not found at location {2}.'.format(
                        syn, self.labels[i], str(class_target)))

            # find all objects in the class
            if syn == '02958343':  # this class have empty model folders
                models = sorted([el.parent.parent
                                 for el in class_target.glob('*/models/model_normalized.obj')])
            else:
                models = sorted(class_target.glob('*'))
            stop = int(len(models) * split)
            if train:
                models = models[:stop]
            else:
                models = models[stop:]
            self.paths += models
            self.synset_idxs += [i] * len(models)

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
        obj_location = self.paths[index] / 'models/model_normalized.obj'
        mesh = import_mesh(str(obj_location), with_materials=self.with_materials,
                           error_handler=ignore_error_handler)
        return mesh

    def get_attributes(self, index):
        synset_idx = self.synset_idxs[index]
        attributes = {
            'name': self.names[index],
            'path': self.paths[index] / 'models/model_normalized.obj',
            'synset': self.synsets[synset_idx],
            'labels': self.labels[synset_idx]
        }
        return attributes

    def get_cache_key(self, index):
        return self.names[index]
