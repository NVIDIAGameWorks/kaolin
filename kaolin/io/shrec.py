# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

synset_to_label = {
    '02691156': 'airplane', 
    '02747177': 'ashcan', 
    '02773838': 'bag', 
    '02801938': 'basket', 
    '02808440': 'bathtub', 
    '02818832': 'bed', 
    '02828884': 'bench', 
    '02834778': 'bicycle', 
    '02843684': 'birdhouse', 
    '02871439': 'bookshelf', 
    '02876657': 'bottle', 
    '02880940': 'bowl', 
    '02924116': 'bus', 
    '02933112': 'cabinet', 
    '02942699': 'camera', 
    '02946921': 'can', 
    '02954340': 'cap', 
    '02958343': 'car', 
    '03001627': 'chair', 
    '03046257': 'clock', 
    '03085013': 'computer keyboard', 
    '03207941': 'dishwasher', 
    '03211117': 'display', 
    '03261776': 'earphone', 
    '03325088': 'faucet', 
    '03337140': 'file', 
    '03467517': 'guitar', 
    '03513137': 'helmet', 
    '03593526': 'jar', 
    '03624134': 'knife', 
    '03636649': 'lamp', 
    '03642806': 'laptop', 
    '03691459': 'loudspeaker', 
    '03710193': 'mailbox', 
    '03759954': 'microphone', 
    '03761084': 'microwave', 
    '03790512': 'motorcycle', 
    '03797390': 'mug', 
    '03928116': 'piano', 
    '03938244': 'pillow', 
    '03948459': 'pistol', 
    '03991062': 'pot', 
    '04004475': 'printer', 
    '04074963': 'remote control', 
    '04090263': 'rifle', 
    '04099429': 'rocket', 
    '04225987': 'skateboard', 
    '04256520': 'sofa', 
    '04330267': 'stove', 
    '04379243': 'table', 
    '04401088': 'telephone', 
    '04460130': 'tower', 
    '04468005': 'train', 
    '04530566': 'vessel', 
    '04554684': 'washer', 
    '04591713': 'wine bottle'}
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
        split (str): String to indicate whether to load train, test or val set.
    """

    def __init__(self, root: str, categories: list = None, split: str = "train"):

        self.root = Path(root)
        self.paths = []
        self.synset_idxs = []
        self.synsets = _convert_categories(categories)
        self.labels = [synset_to_label[s] for s in self.synsets]

        # loops through desired classes
        for i in range(len(self.synsets)):
            syn = self.synsets[i]

            if split == "train":
                class_target = self.root / "train" / syn
            elif split == "test":
                class_target = self.root / "test" / syn
            else:
                class_target = self.root / "val" / syn

            if not class_target.exists():
                raise ValueError(
                    'Class {0} ({1}) was not found at location {2}.'.format(
                        syn, self.labels[i], str(class_target)))

            # find all objects in the class
            models = sorted(class_target.glob('*'))

            self.paths += models
            self.synset_idxs += [i] * len(models)

        self.names = [p.name for p in self.paths]

    def __len__(self):
        return len(self.paths)

    def get_data(self, index):
        obj_location = self.paths[index]
        mesh = import_mesh(str(obj_location))
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
