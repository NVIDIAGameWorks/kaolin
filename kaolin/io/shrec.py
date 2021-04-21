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

VALID_CATEGORIES = [
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

class SHREC16(KaolinDataset):
    r"""Dataset class for SHREC16, used for the "Large-scale 3D shape retrieval
    from ShapeNet Core55" contest at Eurographics 2016.
    More details about the challenge and the dataset are available
    `here <https://shapenet.cs.stanford.edu/shrec16/>`_.

    The `__getitem__` method will return a `KaolinDatasetItem`, with its data field 
    containing a namedtuple returned by :func:`kaolin.io.obj.import_mesh`.

    Args:
        root (str): Path to the root directory of the dataset.
        categories (list): List of categories to load (each class is
                           specified as a string, and must be a valid `SHREC16`
                           category). If this argument is not specified, all categories
                           are loaded by default.
        train (bool): If True, return the train split, else return the test.

    Returns:
        .. code-block::
           dict: {
                attributes: {path: str, category: str, label: int},
                data: kaolin.rep.TriangleMesh
           }
        path: The filepath to the .obj file on disk.
        name: The file name of the .obj file on disk.
        label: A human-readable string describing the loaded sample.
    
    Example:
        >>> dataset = SHREC16(root='/path/to/SHREC16/', categories=['alien', 'ants'], train=False)
        >>> sample = dataset[0]
        >>> sample["attributes"]["path"]
        /path/to/SHREC16/alien/test/T411.obj
        >>> sample["attributes"]["label"]
        alien
    """

    def __init__(self, root: str, categories: Iterable = None, train: bool = True):

        if not categories:
            categories = VALID_CATEGORIES

        self.root = Path(root)
        self.paths = []
        self.labels = []

        for i, category in enumerate(categories):
            if category not in VALID_CATEGORIES:
                raise ValueError(
                    f"Specified category {category} is not valid. "
                    f"Valid categories are {VALID_CATEGORIES}"
                )

            clsdir = os.path.join(root, category, "train" if train else "test")
            curr_models = glob.glob(clsdir + "/*.obj")

            self.paths += curr_models
            self.labels += [] * len(cur)

            if len(cur) == 0:
                raise RuntimeWarning(
                    "No .obj files could be read " f"for category '{cl}'. Skipping..."
                )
        
        self.names = [p.names for p in self.paths]

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.paths)

    def get_data(self, idx):
        """Returns a Usd Mesh from the path."""
        obj_location = self.paths[idx]
        mesh = import_mesh(str(obj_location))
        return mesh

    def get_attributes(self, idx):
        attributes = {
            "path": self.paths[idx],
            "name": self.names[idx],
            "label": self.labels[idx],
        }

        return attributes

    def get_cache_key(self, index):
        return self.names[index]