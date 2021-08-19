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
    """Class containing all the Structured point clouds information"""

    KEYS = {'octrees', 'lengths', 'max_level', 'pyramids', 'exsum', 'point_hierarchies'}

    def __init__(self, octrees, lengths, max_level=None, pyramids=None,
                 exsum=None, point_hierarchies=None):
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

        self.octrees = octrees
        self.lengths = lengths
        self._max_level = max_level
        self._pyramids = pyramids
        self._exsum = exsum
        self._point_hierarchies = point_hierarchies

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
