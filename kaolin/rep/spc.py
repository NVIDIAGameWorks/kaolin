# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
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


__all__ = [
    'Spc'
]

import torch
from ..ops.batch import list_to_packed

class Spc(object):
    """Data class holding all :ref:`Structured Point Cloud (SPC)<spc>` information.

    This class supports batching through :ref:`packed<packed>` representation:
    a single Spc object can pack multiple SPC structures of variable sizes.

    SPC data structures are represented through the combination various tensors detailed below:

    ``octrees`` compress the information required to build a full SPC.
    In practice, they are a low level structure which also constitute the
    :ref:`core part<spc_octree>` of the SPC data structure.

    ``octrees`` are kept as a torch.ByteTensor, where each byte represents a single octree parent cell,
    and each bit represents the occupancy of a child octree cell.
    e.g: 8 bits for 8 cells.

    Bits describe the octree cells in Morton Order::

         . . . . . . . .
         | .   3  .  7  | .                    3   7
         |   . . . . . . . .           ===>    1   5
         |   | .   1  . | 5   .
         |   |   . . . . . . . .
         |   |    |     |       |              2   6
          . .|. . | . . .       |      ===>    0   4
            .| 2  |.  6   .     |
              . . | . . . . .   |
                . | 0  .  4   . |
                  . . . . . . . .

    If a cell is occupied, an additional cell byte may be generated in the next level,
    up till the argument ``level``.

    For example, a ``SPC.octrees`` field may, look as follows::

            tensor([255, 128,  64,  32,  16,   8,   4,   2,  23], dtype=torch.uint8)

    Here "octrees" represents an octree of 9 nodes.
    The binary representation should be interpreted as follows::

            Level #1, Path*,      11111111    (All cells are occupied, therefore 8 bytes are allocated for level 2)
            Level #2, Path*-1,    10000000
            Level #2, Path*-2,    01000000
            Level #2, Path*-3,    00100000
            Level #2, Path*-4,    00010000
            Level #2, Path*-5,    00001000
            Level #2, Path*-6,    00000100
            Level #2, Path*-7,    00000010
            Level #2, Path*-8,    00010111

    ``lengths`` is a tensor of integers required to support batching. Since we assume a packed representation,
    all octree cells are shaped as a single stacked 1D tensor. ``lengths`` specifies the number of cells (bytes) each
    octree uses.

    ``features`` represent an optional per-point feature vector.
    When ``features`` is not ``None``, a feature is kept for each point at the highest-resolution level in the octree.

    ``max_level`` is an integer which specifies how many recursive levels an octree should have.

    ``point_hierarchies``, ``pyramid``, ``exsum`` are auxilary structures, which are generated upon request and
    enable efficient indexing to SPC entries.
    """

    KEYS = {'octrees', 'lengths', 'max_level', 'pyramids', 'exsum', 'point_hierarchies'}

    def __init__(self, octrees, lengths, max_level=None, pyramids=None,
                 exsum=None, point_hierarchies=None, features=None):
        assert (isinstance(octrees, torch.Tensor) and octrees.dtype == torch.uint8 and
                octrees.ndim == 1), "octrees must be a 1D ByteTensor."
        assert (isinstance(lengths, torch.Tensor) and lengths.dtype == torch.int and
                lengths.ndim == 1), "lengths must be a 1D IntTensor."
        assert (max_level is None) or (isinstance(max_level, int)), \
            "max_level must an int."

        if pyramids is not None:
            assert isinstance(pyramids, torch.Tensor) and pyramids.dtype == torch.int, \
                "pyramids must be an IntTensor."
            assert (pyramids.ndim == 3 and
                    pyramids.shape[0] == lengths.shape[0] and
                    pyramids.shape[1] == 2 and
                    ((max_level is None) or (pyramids.shape[2] == max_level + 2))), \
                "pyramids must be of shape (batch_size, 2, max_level + 2)."
            assert not pyramids.is_cuda, "pyramids must be on cpu."

        if exsum is not None:
            assert isinstance(exsum, torch.Tensor) and exsum.dtype == torch.int, \
                "exsum must be an IntTensor."
            assert (exsum.ndim == 1 and
                    exsum.shape[0] == octrees.shape[0] + lengths.shape[0]), \
                "exsum must be of shape (num_bytes + batch_size)."
            assert exsum.device == octrees.device, \
                "exsum must be on the same device than octrees."

        if point_hierarchies is not None:
            assert isinstance(point_hierarchies, torch.Tensor) and \
                point_hierarchies.dtype == torch.short, \
                "point_hierarchies must be a ShortTensor."
            assert (point_hierarchies.ndim == 2 and
                    point_hierarchies.shape[1] == 3), \
                "point_hierarchies must be of shape (num_nodes, 3)."
            assert point_hierarchies.device == octrees.device, \
                "point_hierarchies must be on the same device than octrees."

        if features is not None:
            assert isinstance(features, torch.Tensor), \
                "features must be a torch.Tensor"
            assert features.device == octrees.device, \
                "features must be on the same device as octrees."

        self.octrees = octrees
        self.lengths = lengths
        self._max_level = max_level
        self._pyramids = pyramids
        self._exsum = exsum
        self._point_hierarchies = point_hierarchies
        self.features = features

    @classmethod
    def make_dense(cls, level, device='cuda'):
        """Creates a dense, fully occupied Spc object.
        The Spc will have ``level`` levels of detail.

        Args:
            level (int):
                Number of levels to use for the dense Spc.
            device (torch.device):
                Torch device to keep the spc octree

        Return:
            (kaolin.rep.Spc): a new fully occupied ``Spc``.
        """
        from ..ops.spc import create_dense_spc
        octree, lengths = create_dense_spc(level, device)  # Create a single entry batch
        return Spc(octrees=octree, lengths=lengths)

    @classmethod
    def from_features(cls, feature_grids, masks=None):
        """Creates a sparse Spc object from the feature grid.

        Args:
            feature_grids (torch.Tensor):
                The sparse 3D feature grids, of shape
                :math:`(\text{batch_size}, \text{feature_dim}, X, Y, Z)`
            masks (optional, torch.BoolTensor):
                The topology mask, showing where are the features,
                of shape :math:`(\text{batch_size}, X, Y, Z)`.
                Default: A feature is determined when not full of zeros.

        Returns:
            (torch.ByteTensor, torch.IntTensor, torch.Tensor):
                a tuple containing:

                    - The octree, of size :math:`(\text{num_nodes})`

                    - The lengths of each octree, of size :math:`(\text{batch_size})`

                    - The coalescent features, of same dtype than ``feature_grids``,
                      of shape :math:`(\text{num_features}, \text{feature_dim})`.
        Return:
            (kaolin.rep.Spc): a ``Spc``, with length of :math:`(\text{batch_size})`,
            an octree of size octree, of size :math:`(\text{num_nodes})`, and the features field
            of the same dtype as ``feature_grids`` and of shape :math:`(\text{num_features}, \text{feature_dim})`.
        """
        from ..ops.spc import feature_grids_to_spc
        octrees, lengths, coalescent_features = feature_grids_to_spc(feature_grids, masks=masks)
        return Spc(octrees=octrees, lengths=lengths, features=coalescent_features)

    # TODO(cfujitsang): could be interesting to separate into multiple functions
    def _apply_scan_octrees(self):
        # to break circular dependency
        from ..ops.spc import scan_octrees
        max_level, pyramids, exsum = scan_octrees(self.octrees, self.lengths)
        self._max_level = max_level
        self._pyramids = pyramids
        self._exsum = exsum

    def _apply_generate_points(self):
        # to break circular dependency
        from ..ops.spc import generate_points
        self._point_hierarchies = generate_points(self.octrees, self.pyramids, self.exsum)

    @property
    def max_level(self):
        if self._max_level is None:
            self._apply_scan_octrees()
        return self._max_level

    @property
    def pyramids(self):
        if self._pyramids is None:
            self._apply_scan_octrees()
        return self._pyramids

    @property
    def exsum(self):
        if self._exsum is None:
            self._apply_scan_octrees()
        return self._exsum

    @property
    def point_hierarchies(self):
        if self._point_hierarchies is None:
            self._apply_generate_points()
        return self._point_hierarchies

    @classmethod
    def from_list(cls, octrees_list):
        """Generate an Spc from a list of octrees.

        Args:
            octrees_list (list of torch.ByteTensor):
                list containing multiple 1D torch.ByteTensor,
                each representing an octree.

        Return:
            (kaolin.rep.Spc): a new ``Spc``.
        """
        octrees, lengths = list_to_packed(
            [octree.reshape(-1, 1) for octree in octrees_list])
        return cls(octrees.reshape(-1).contiguous(), lengths.reshape(-1).int())

    def to(self, device, non_blocking=False,
           memory_format=torch.preserve_format):
        _octrees = self.octrees.to(device=device,
                                   non_blocking=non_blocking,
                                   memory_format=memory_format)

        # torch tensor.to() return the self if the type is identical
        if _octrees.data_ptr() == self.octrees.data_ptr():
            return self
        else:
            if self._exsum is not None:
                _exsum = self._exsum.to(device=device,
                                        non_blocking=non_blocking,
                                        memory_format=memory_format)
            else:
                _exsum = None

            if self._point_hierarchies is not None:
                _point_hierarchies = self.point_hierarchies.to(
                    device=device,
                    non_blocking=non_blocking,
                    memory_format=memory_format)
            else:
                _point_hierarchies = None

            return Spc(_octrees, self.lengths, self._max_level, self._pyramids,
                       _exsum, _point_hierarchies)

    def cuda(self, device='cuda', non_blocking=False,
             memory_format=torch.preserve_format):
        return self.to(device=device, non_blocking=non_blocking,
                       memory_format=memory_format)

    def cpu(self, memory_format=torch.preserve_format):
        return self.to(device='cpu', memory_format=memory_format)

    @property
    def batch_size(self):
        return self.lengths.shape[0]

    def to_dict(self, keys=None):
        if keys is None:
            return {k: getattr(self, k) for k in self.KEYS}
        else:
            return {k: getattr(self, k) for k in keys}

    def num_points(self, lod: int):
        """
        Returns how many points the SPC holds at a given level of detail.

        Args:
            lod (int):
                Index of a level of detail.
                Level 0 is considered the root and always holds a single point,
                level 1 holds up to :math:`(\text{num_points}=8)` points,
                level 2 holds up to :math:`(\text{num_points}=8^{2})`, and so forth.
        Return:
            (torch.Tensor): The number of points each SPC entry holds for the given level of detail.
        """
        return self.pyramids[:, 0, lod]
