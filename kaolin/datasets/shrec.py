# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

from typing import Iterable

import os
import glob

from ..rep import TriangleMesh

from .base import KaolinDataset


class SHREC16(KaolinDataset):
    r"""Dataset class for SHREC16, used for the "Large-scale 3D shape retrieval
    from ShapeNet Core55" contest at Eurographics 2016.

    More details about the challenge and the dataset are available
    `here <https://shapenet.cs.stanford.edu/shrec16/>`_.

    Args:
        root (str): Path to the root directory of the dataset.
        categories (list): List of categories to load (each class is
            specified as a string, and must be a valid `SHREC16`
            category). If this argument is not specified, all categories
            are loaded by default.
        train (bool): If True, return the train split, else return the test
            split (default: True).
    Returns:
        .. code-block::

           dict: {
                attributes: {path: str, category: str, label: int},
                data: kaolin.rep.TriangleMesh
           }

        path: The filepath to the .obj file on disk.
        category: A human-readable string describing the loaded sample.
        label: An integer (in the range :math:`[0, \text{len(categories)}]`)
            and can be used for training classifiers for example.
        vertices: Vertices of the loaded mesh (:math:`(*, 3)`), where :math:`*`
            indicates a positive integer.
        faces: Faces of the loaded mesh (:math:`(*, 3)`), where :math:`*`
            indicates a positive integer.

    Example:
        >>> dataset = SHREC16(root='/path/to/SHREC16/', categories=['alien', 'ants'], train=False)
        >>> sample = dataset[0]
        >>> sample["attributes"]["path"]
        /path/to/SHREC16/alien/test/T411.obj
        >>> sample["attributes"]["category"]
        alien
        >>> sample["attributes"]["label"]
        0
        >>> sample["data"].vertices.shape
        torch.Size([252, 3])
        >>> sample["data"].faces.shape
        torch.Size([500, 3])

    """
    _VALID_CATEGORIES = [
        "alien",
        "ants",
        "armadillo",
        "bird1",
        "bird2",
        "camel",
        "cat",
        "centaur",
        "dinosaur",
        "dino_ske",
        "dog1",
        "dog2",
        "flamingo",
        "glasses",
        "gorilla",
        "hand",
        "horse",
        "lamp",
        "laptop",
        "man",
        "myScissor",
        "octopus",
        "pliers",
        "rabbit",
        "santa",
        "shark",
        "snake",
        "spiders",
        "two_balls",
        "woman",
    ]

    def initialize(
        self,
        root: str,
        categories: Iterable = None,
        train: bool = True,
    ):

        if not categories:
            categories = SHREC16._VALID_CATEGORIES
        for category in categories:
            if category not in SHREC16._VALID_CATEGORIES:
                raise ValueError(
                    f"Specified category {category} is not valid. "
                    f"Valid categories are {SHREC16._VALID_CATEGORIES}"
                )

        self.root = root
        self.categories_to_load = categories
        self.train = train
        self.num_samples = 0
        self.paths = []
        self.category_names = []
        self.labels = []
        for i, cl in enumerate(self.categories_to_load):
            clsdir = os.path.join(root, cl, "train" if self.train else "test")
            cur = glob.glob(clsdir + "/*.obj")

            self.paths = self.paths + cur
            self.category_names += [cl] * len(cur)
            self.labels += [i] * len(cur)
            self.num_samples += len(cur)
            if len(cur) == 0:
                raise RuntimeWarning(
                    "No .obj files could be read " f"for category '{cl}'. Skipping..."
                )

    def __len__(self):
        """Returns the length of the dataset. """
        return self.num_samples

    def _get_data(self, idx):
        obj_location = self.paths[idx]
        mesh = TriangleMesh.from_obj(obj_location)
        return mesh

    def _get_attributes(self, idx):
        attributes = {
            "path": self.paths[idx],
            "category": self.category_names[idx],
            "label": self.labels[idx],
        }
        return attributes
