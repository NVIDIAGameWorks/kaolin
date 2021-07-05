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

from kaolin.io.dataset import KaolinDataset
from kaolin.io.obj import import_mesh, ignore_error_handler

synset_to_labels = {
    '03790512': ['motorcycle', 'bike'],
    '02808440': ['bathtub', 'bathing tub', 'bath', 'tub'],
    '02871439': ['bookshelf'],
    '03761084': ['microwave', 'microwave oven'],
    '04530566': ['vessel', 'watercraft'],
    '02691156': ['airplane', 'aeroplane', 'plane'],
    '04379243': ['table'],
    '03337140': ['file', 'file cabinet', 'filing cabinet'],
    '04256520': ['sofa', 'couch', 'lounge'],
    '03636649': ['lamp'],
    '03928116': ['piano', 'pianoforte', 'forte-piano'],
    '04004475': ['printer', 'printing machine'],
    '03593526': ['jar'],
    '04330267': ['stove'],
    '04554684': ['washer', 'automatic washer', 'washing machine'],
    '03948459': ['pistol', 'handgun', 'side arm', 'shooting iron'],
    '03001627': ['chair'],
    '03797390': ['mug'],
    '02801938': ['basket', 'handbasket'],
    '03710193': ['mailbox', 'letter box'],
    '03938244': ['pillow'],
    '03624134': ['knife'],
    '02954340': ['cap'],
    '02773838': ['bag', 'traveling bag', 'travelling bag', 'grip', 'suitcase'],
    '02747177': ['ashcan', 'trash can', 'garbage can', 'wastebin', 
                 'ash bin', 'ash-bin', 'ashbin', 'dustbin', 'trash barrel', 'trash bin'],
    '04460130': ['tower'],
    '02933112': ['cabinet'],
    '02876657': ['bottle'],
    '03991062': ['pot', 'flowerpot'],
    '02843684': ['birdhouse'],
    '02818832': ['bed'],
    '02958343': ['car', 'auto', 'automobile', 'machine', 'motorcar'],
    '03642806': ['laptop', 'laptop computer'],
    '03085013': ['computer keyboard', 'keypad'],
    '04074963': ['remote control', 'remote'],
    '02924116': ['bus', 'autobus', 'coach', 'charabanc', 'double-decker', 
                 'jitney', 'motorbus', 'motorcoach', 'omnibus', 'passenger vehi'],
    '04225987': ['skateboard'],
    '03261776': ['earphone', 'earpiece', 'headphone', 'phone'],
    '02880940': ['bowl'],
    '03325088': ['faucet', 'spigot'],
    '03211117': ['display', 'video display'],
    '04468005': ['train', 'railroad train'],
    '03691459': ['loudspeaker', 'speaker', 'speaker unit', 'loudspeaker system', 'speaker system'],
    '04090263': ['rifle'],
    '02946921': ['can', 'tin', 'tin can'],
    '04099429': ['rocket', 'projectile'],
    '03467517': ['guitar'],
    '04401088': ['telephone', 'phone', 'telephone set'],
    '03046257': ['clock'],
    '03759954': ['microphone', 'mike'],
    '03513137': ['helmet'],
    '02834778': ['bicycle', 'bike', 'wheel', 'cycle'],
    '03207941': ['dishwasher', 'dish washer', 'dishwashing machine'],
    '02828884': ['bench'],
    '02942699': ['camera', 'photographic camera']}

# Label to Synset mapping (for ShapeNet core classes)
label_to_synset = {label: synset for synset, labels in synset_to_labels.items() for label in labels}

def _convert_categories(categories):
    if not (c in synset_to_label.keys() + label_to_synset.keys()
            for c in categories):
        warnings.warn('Some or all of the categories requested are not part of \
            Shrec16. Data loading may fail if these categories are not avaliable.')
    synsets = [label_to_synset[c] if c in label_to_synset.keys()
               else c for c in categories]
    return synsets

class SHREC16(KaolinDataset):
    r"""Dataset class for SHREC16, used for the "Large-scale 3D shape retrieval
    from ShapeNet Core55" contest at Eurographics 2016.
    More details about the challenge and the dataset are available
    `here <https://shapenet.cs.stanford.edu/shrec16/>`_.

    The `__getitem__` method will return a `KaolinDatasetItem`, with its `data`
    field containing a `kaolin.io.obj.return_type`.

    Args:
        root (str): Path to the root directory of the dataset.
        categories (list): List of categories to load (each class is
                           specified as a string, and must be a valid `SHREC16`
                           category). If this argument is not specified, all categories
                           are loaded by default.
        split (str): String to indicate what split to load, among ["train", "val", "test"].
                     Default: "train".
    """

    def __init__(self, root: str, categories: list = None, split: str = "train"):

        self.root = Path(root)
        self.paths = []
        self.synset_idxs = []

        if split == "test":
            # Setting synsets and labels to None if in test split
            self.synsets = [None]
            self.labels = [None]
        else:
            if categories is None:
                self.synsets = list(synset_to_labels.keys())
            else:
                self.synsets = _convert_categories(categories)
            self.labels = [synset_to_labels[s] for s in self.synsets]

        # loops through desired classes
        if split == "test":
            class_target = self.root / "test_allinone"
            # find all objects in the class
            models = sorted(class_target.glob('*'))

            self.paths += models
            self.synset_idxs += [0] * len(models)

        else:
            for i in range(len(self.synsets)):
                syn = self.synsets[i]

                if split == "train":
                    class_target = self.root / "train" / syn
                elif split == "val":
                    class_target = self.root / "val" / syn
                else:
                    raise ValueError(f'Split must be either train, test or val, '
                                     f'got {split} instead.')

                if not class_target.exists():
                    raise ValueError(
                        'Class {0} ({1}) was not found at location {2}.'.format(
                            syn, self.labels[i], str(class_target)))

                # find all objects in the class
                models = sorted(class_target.glob('*'))

                self.paths += models
                self.synset_idxs += [i] * len(models)

        self.names = [os.path.join(p.parent.name, p.name) for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def get_data(self, index):
        obj_location = self.paths[index]
        mesh = import_mesh(str(obj_location), error_handler=ignore_error_handler)
        return mesh

    def get_attributes(self, index):
        synset_idx = self.synset_idxs[index]
        attributes = {
            'name': self.names[index],
            'path': self.paths[index],
            'synset': self.synsets[synset_idx],
            'label': self.labels[synset_idx]
        }
        return attributes

    def get_cache_key(self, index):
        return self.names[index]
