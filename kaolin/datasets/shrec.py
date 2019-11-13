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

import torch
import os
import glob
from torch.utils.data import Dataset


from kaolin.rep import TriangleMesh


class SHREC16(Dataset):
    r"""Class to help in loading the SHREC16 dataset.

    SHREC16 is the dataset used for the "Large-scale 3D shape retrieval
    from ShapeNet Core55" contest at Eurographics 2016.

    More details about the challenge and the dataset are available
    `here <https://shapenet.cs.stanford.edu/shrec16/>`_.

    Args:
        root (str): Path to the root directory of the dataset.
        categories (list): List of categories to load (each class is
            specified as a string, and must be a valid `SHREC16`
            category).
        mode (str, choices=['train', 'test']): Whether to load the
            'train' split or the 'test' split
    Returns:
        dict: Dictionary with keys: 'vertices' : vertices , 'faces' : faces

    """

    def __init__(self, root: str, categories: list = ['alien'],
                 mode: list = 'train'):

        super(SHREC16, self).__init__()

        if mode not in ['train', 'test']:
            raise ValueError('Argument \'mode\' must be one of \'train\''
                             'or \'test\'. Got {0} instead.'.format(mode))

        VALID_CATEGORIES = [
            'alien', 'ants', 'armadillo', 'bird1', 'bird2', 'camel',
            'cat', 'centaur', 'dinosaur', 'dino_ske', 'dog1', 'dog2',
            'flamingo', 'glasses', 'gorilla', 'hand', 'horse', 'lamp',
            'laptop', 'man', 'myScissor', 'octopus', 'pliers', 'rabbit',
            'santa', 'shark', 'snake', 'spiders', 'two_balls', 'woman'
        ]

        for category in categories:
            if category not in VALID_CATEGORIES:
                raise ValueError(f'Specified category {category} is not valid. '
                                 'Valid categories are {VALID_CATEGORIES}')

        self.mode = mode
        self.root = root
        self.categories = categories
        self.num_samples = 0
        self.paths = []
        self.categories = []
        for cl in self.categories:
            clsdir = os.path.join(root, cl, self.mode)
            cur = glob.glob(clsdir + '/*.obj')

            self.paths = self.paths + cur
            self.categories += [cl] * len(cur)
            self.num_samples += len(cur)
            if len(cur) == 0:
                raise RuntimeWarning('No .obj files could be read '
                                     f'for category \'{cl}\'. Skipping...')

    def __len__(self):
        """Returns the length of the dataset. """
        return self.num_samples

    def __getitem__(self, idx):
        """Returns the sample at index idx. """

        # Read in the list of vertices and faces
        # from the obj file.
        obj_location = self.paths[idx]
        mesh = TriangleMesh.from_obj(obj_location)
        category = self.categories[idx]
        # Return these tensors as a dictionary.
        data = dict()
        attributes = dict()
        data['vertices'] = mesh.vertices
        data['faces'] = mesh.faces
        attributes['rep'] = 'Mesh'
        attributes['name'] = obj_location
        attributes['class']: cagetory
        return {'attributes': attributes, 'data': data}
