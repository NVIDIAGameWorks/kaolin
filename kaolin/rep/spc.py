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
    ```

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

    ```

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

    `` parents`` provides an optional quick mapping between SPC cells and their parents.

    ``pyramids_dual``, ``point_hierarchies_dual``, ``trinkets`` are fields used to map between the SPC primary octree,
    which keeps features at cell centers, and an optional dual-octree, which keeps features at the 8 corners of each
    cell. The dual octree is only generated upon request (by accessing such fields).
    """

    KEYS = {'octrees', 'lengths', 'max_level', 'pyramids', 'exsum', 'point_hierarchies',
            'parents', 'pyramids_dual', 'point_hierarchies_dual', 'trinkets'}

    def __init__(self, octrees, lengths, max_level=None, pyramids=None,
                 exsum=None, point_hierarchies=None, features=None, parents=None,
                 pyramids_dual=None, point_hierarchies_dual=None, trinkets=None):

        self.validate_fields(octrees=octrees, lengths=lengths, max_level=max_level, pyramids=pyramids,
                             exsum=exsum, point_hierarchies=point_hierarchies, features=features, parents=parents,
                             pyramids_dual=pyramids_dual, point_hierarchies_dual=point_hierarchies_dual,
                             trinkets=trinkets)
        self.octrees = octrees
        self.lengths = lengths
        self._max_level = max_level
        self._pyramids = pyramids
        self._exsum = exsum
        self._point_hierarchies = point_hierarchies
        self._parents = parents
        self.features = features

        # In dual-octree mode features exist on cell corners rather than cell center
        self._pyramids_dual = pyramids_dual
        self._point_hierarchies_dual = point_hierarchies_dual
        self._trinkets = trinkets

    def validate_fields(self, octrees, lengths, max_level=None, pyramids=None,
                        exsum=None, point_hierarchies=None, features=None, parents=None,
                        pyramids_dual=None, point_hierarchies_dual=None, trinkets=None):
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
                "exsum must be on the same device as octrees."

        if point_hierarchies is not None:
            assert isinstance(point_hierarchies, torch.Tensor) and \
                   point_hierarchies.dtype == torch.short, \
                "point_hierarchies must be a ShortTensor."
            assert (point_hierarchies.ndim == 2 and
                    point_hierarchies.shape[1] == 3), \
                "point_hierarchies must be of shape (num_nodes, 3)."
            assert point_hierarchies.device == octrees.device, \
                "point_hierarchies must be on the same device as octrees."

        if parents is not None:
            assert isinstance(parents, torch.Tensor) and \
                   point_hierarchies.dtype == torch.int, \
                "parents must be an IntTensor."
            assert point_hierarchies is not None, "parents mapping must correspond to point_hierarchies"
            assert (parents.ndim == 1 and
                    parents.shape[0] == point_hierarchies.shape[0]), \
                "parents must be of shape (num_nodes,)."
            assert parents.device == octrees.device, \
                "parents must be on the same device as octrees."

        if features is not None:
            assert isinstance(features, torch.Tensor), \
                "features must be a torch.Tensor"
            assert features.device == octrees.device, \
                "features must be on the same device as octrees."

        if pyramids_dual is not None:
            assert isinstance(pyramids_dual, torch.Tensor) and pyramids_dual.dtype == torch.long, \
                "pyramids_dual must be an LongTensor."
            assert (pyramids_dual.ndim == 3 and
                    pyramids_dual.shape[0] == lengths.shape[0] and
                    pyramids_dual.shape[1] == 2 and
                    ((max_level is None) or (pyramids_dual.shape[2] == max_level + 2))), \
                "pyramids_dual must be of shape (batch_size, 2, max_level + 2)."
            assert not pyramids_dual.is_cuda, "pyramids_dual must be on cpu."

        if point_hierarchies_dual is not None:
            assert isinstance(point_hierarchies_dual, torch.Tensor) and \
                   point_hierarchies_dual.dtype == torch.short, \
                "point_hierarchies_dual must be a ShortTensor."
            assert (point_hierarchies_dual.ndim == 2 and
                    point_hierarchies_dual.shape[1] == 3), \
                "point_hierarchies_dual must be of shape (num_nodes, 3)."
            assert point_hierarchies_dual.device == octrees.device, \
                "point_hierarchies_dual must be on the same device as octrees."

        if trinkets is not None:
            assert isinstance(trinkets, torch.Tensor) and \
                   trinkets.dtype == torch.int, \
                "trinkets must be an IntTensor."
            assert (trinkets.ndim == 2 and trinkets.shape[1] == 8), \
                "trinkets must be of shape (num_nodes, 3)."
            assert trinkets.device == octrees.device, \
                "trinkets must be on the same device as octrees."

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

    def _apply_create_dual(self):
        r"""Initializes the basic fields required for dual-octree ops (which saves features at 8 corners of voxels).
        """
        # to break circular dependency
        from ..ops.spc import unbatched_make_dual
        if self.batch_size == 1:
            self._point_hierarchies_dual, self._pyramids_dual = unbatched_make_dual(self.point_hierarchies,
                                                                                    self.pyramids[0])
        else:   # Unbatched op currently runs in for-loop
            points_hierarchical_dual_list, pyramids_dual_list = list(), list()
            for spc in iter(self):
                points_hierarchical_dual, pyramids_dual = unbatched_make_dual(spc.point_hierarchies, spc.pyramids[0])
                points_hierarchical_dual_list.append(points_hierarchical_dual)
                pyramids_dual_list.append(pyramids_dual)
            self._point_hierarchies_dual = torch.cat(points_hierarchical_dual_list, dim=0)
            self._pyramids_dual = torch.stack(pyramids_dual_list, dim=0)

    def _apply_create_trinkets_parents(self):
        # to break circular dependency
        from ..ops.spc import unbatched_make_trinkets
        point_hierarchies_dual, pyramids_dual = self.dual   # Force creation of dual fields
        if self.batch_size == 1:
            self._trinkets, self._parents = unbatched_make_trinkets(self.point_hierarchies, self.pyramids[0],
                                                                    point_hierarchies_dual, pyramids_dual[0])
        else:
            trinkets_list, parents_list = list(), list()
            for spc in iter(self):  # Unbatched op currently runs in for-loop
                trinkets, parents = unbatched_make_trinkets(spc.point_hierarchies, spc.pyramids[0],
                                                            spc.point_hierarchies_dual, spc.pyramids_dual[0])
                trinkets_list.append(trinkets)
                parents_list.append(parents)
            self._trinkets, self._parents = torch.cat(trinkets_list, dim=0), torch.cat(parents_list, dim=0)

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

    @property
    def parents(self):
        r""" Returns the parents mapping of each point in the octree, and creates it if needed.
        `parents` is a quick mapping between SPC cells and their parents in the previous LOD.

        seealso:: :func:`kaolin.ops.unbatched_make_trinkets`

        Return:
            torch.IntTensor:
                - Indirection pointers to the parents of shape :math:`(\text{num_points})`.
        """
        if self._parents is None:
            self._apply_create_trinkets_parents()
        return self._parents

    @property
    def dual(self):
        r"""Returns the dual of the octree, and creates it if needed.
        Each node of the primary octree (represented as the :ref:`point_hierarchies <spc_points>`)
        can be thought of as voxels with 8 corners. The dual of the octree represents the corners
        of the primary octree nodes as another tree of nodes with a hierarchy of points and a pyramid.
        The mapping from the primary octree nodes to the nodes in the dual tree can be obtained through
        the ``trinkets`` property.

        ```
                       [Primary octree]                        [Dual octree]
                       . . . . . . . .                        X . . .X. . . X
                       | .   X  .  X  | .                     | .      .     | .
                       |   . . . . . . . .           ===>     |   X . . X . . . X
                       |   | .   X  . | X   .                 X   | .      . |     .
                       |   |   . . . . . . . .                |   |   X . . .X. . . X
                       |   |    |     |       |               |   X    |     |       |
                        . .|. . | . . .       |      ===>      X .|. . X . . X       |
                          .| X  |.  X   .     |                  .|    |.      .     X
                            . . | . . . . .   |                    X . | . X . . X   |
                              . | X  .  X   . |                      . |    .      . |
                                . . . . . . . .                        X . . X . . . X
        ```

        seealso:: :func:`kaolin.ops.spc.unbatched_make_dual`

        Return:
            (torch.ShortTensor, torch.LongTensor):
                - The point hierarchy of the dual octree of shape :math:`(\text{num_dual_points}, 3)`.
                - The dual pyramid of shape :math:`(2, \text{max_level}+2)`
        """
        if self._pyramids_dual is None or self._point_hierarchies_dual is None:
            self._apply_create_dual()
        return self._point_hierarchies_dual, self._pyramids_dual

    @property
    def trinkets(self):
        r"""Returns the trinkets for the dual octree, and creates the trinkets mapping if needed.
        The trinkets are indirection pointers (in practice, indices) from the nodes of the primary octree
        to the nodes of the dual octree. The nodes of the dual octree represent the corners of the voxels
        defined by the primary octree. The trinkets are useful for accessing values stored on the corners
        (like for example a signed distance function) and interpolating them from the nodes of the primary
        octree.

        seealso::
            :func:`kaolin.ops.spc.unbatched_make_trinkets`, :func:`kaolin.ops.spc.unbatched_interpolate_trilinear`

        Return:
            torch.IntTensor:
                - The trinkets of shape :math:`(\text{num_points}, 8)`.
        """
        if self._trinkets is None:
            self._apply_create_trinkets_parents()
        return self._trinkets

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

        def _field_to(field, device, non_blocking, memory_format):
            if field is not None:
                return field.to(device=device, non_blocking=non_blocking, memory_format=memory_format)
            else:
                return None

        _octrees = _field_to(self.octrees, device, non_blocking, memory_format)

        # torch tensor.to() return the self if the type is identical
        if _octrees.data_ptr() == self.octrees.data_ptr():
            return self
        else:
            return Spc(octrees=_octrees,
                       lengths=self.lengths,
                       max_level=self._max_level,
                       pyramids=self._pyramids,
                       exsum=_field_to(self._exsum, device, non_blocking, memory_format),
                       point_hierarchies=_field_to(self.point_hierarchies, device, non_blocking, memory_format) ,
                       features=_field_to(self.features, device, non_blocking, memory_format),
                       parents=_field_to(self._parents, device, non_blocking, memory_format),
                       pyramids_dual=_field_to(self._pyramids_dual, device, non_blocking, memory_format),
                       trinkets=_field_to(self._trinkets, device, non_blocking, memory_format))

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

    def _packed_fields_boundaries(self):
        r""" A convenience method for iteration over packed tensors: this function returns a tensor per field
        marking the index where each entry starts.

        Return:
            (torch.LongTensor, (torch.LongTensor, torch.LongTensor, torch.LongTensor))
                - Packed boundaries of octree field.
                - Packed boundaries of num_points based fields (i.e: point_hierarchies).
                - Packed boundaries of exsum field.
                - Packed boundaries of dual num_points based fields (i.e: point_hierarchies_dual).
            Note: None is returned if no field was initialized for a given packed field type.
        """
        # packed octree bounds are determined by length field
        octree_bounds = torch.cat((self.lengths.new_zeros(1), torch.cumsum(self.lengths, dim=0)))
        octree_bounds = octree_bounds.long()

        # optional fields: bounds are determined by pyramids
        pts_bounds = None
        if any([x is not None for x in (self._point_hierarchies, self._parents, self.features)]):
            pts_bounds = torch.cat((self.pyramids.new_zeros(1), torch.cumsum(self.pyramids[:, 1, -1], dim=0)))
            pts_bounds = pts_bounds.long()

        # optional field: exsum bounds are determined by length + 1
        exsum_bounds = None
        if self._exsum is not None:
            exsum_bounds = octree_bounds + torch.range(start=0, end=self.batch_size)
            exsum_bounds = exsum_bounds.long()

        # optional fields: bounds are determined by dual pyramids
        dual_pts_bounds = None
        if any([x is not None for x in (self._point_hierarchies_dual, self._trinkets)]):
            dual_pts_bounds = torch.cat((self._pyramids_dual.new_zeros(1),
                                         torch.cumsum(self._pyramids_dual[:, 1, -1], dim=0)))
            dual_pts_bounds = dual_pts_bounds.long()

        return octree_bounds, pts_bounds, exsum_bounds, dual_pts_bounds

    def _packed_fields_iterators(self):
        r"""
            Returns a tuple of iterators for the boundaries of the Spc packed fields.
            Each iterator returns a slice object, to be used to directly index the required information
            for a specific entry.
        """
        def _slice_generator(_bounds):
            for start, end in zip(_bounds[:-1], _bounds[1:]):
                yield slice(start, end)

        packed_bounds = self._packed_fields_boundaries()
        packed_iters = [_slice_generator(bounds) if bounds is not None else None for bounds in packed_bounds]

        # packed_iters: octree_bounds_iter, pts_bounds_iter, exsum_bounds_iter, dual_pts_bounds_iter
        return packed_iters

    def __iter__(self):
        r""" Returns an iterator for iterating over each of the SPC entries packed in this instance.
        The number of items this iterator returns is `self.batch_size`.
        Each item this iterator returns is a SPC instance of a single item batch.
        This iterator can also be used as a convenience function for quick access to unbatched SPC operations.

        Return:
            (kaolin.rep.Spc):
                Each iteration returns a Structured Point Cloud (SPC) object, holding a single-item batch.
        """
        octree_bounds_iter, pts_bounds_iter, exsum_bounds_iter, dual_pts_bounds_iter = self._packed_fields_iterators()

        for idx in range(self.batch_size):
            octree = self.octrees[next(octree_bounds_iter)]
            length = self.lengths[idx:idx+1]
            pyramids = self._pyramids[idx:idx+1] if self._pyramids is not None else None
            pyramids_dual = self._pyramids_dual[idx:idx + 1] if self._pyramids_dual is not None else None
            exsum = self.exsum[next(exsum_bounds_iter)] if exsum_bounds_iter is not None else None

            point_hierarchies, parents, features, trinkets = None, None, None, None
            if pts_bounds_iter is not None:
                pt_slice = next(pts_bounds_iter)
                point_hierarchies = self.point_hierarchies[pt_slice] if self._point_hierarchies is not None else None
                parents = self.parents[pt_slice] if self._parents is not None else None
                features = self.features[pt_slice] if self.features is not None else None
                trinkets = self.trinkets[pt_slice] if self._trinkets is not None else None

            point_hierarchies_dual = None
            if dual_pts_bounds_iter is not None:
                point_hierarchies_dual = self._point_hierarchies_dual[next(dual_pts_bounds_iter)] \
                    if self._point_hierarchies_dual is not None else None

            yield Spc(octrees=octree, lengths=length, max_level=self._max_level, pyramids=pyramids,
                      exsum=exsum, point_hierarchies=point_hierarchies, parents=parents, features=features,
                      pyramids_dual=pyramids_dual, point_hierarchies_dual=point_hierarchies_dual,
                      trinkets=trinkets)

    def __getitem__(self, idx):
        octree_bounds, pts_bounds, exsum_bounds, dual_pts_bounds = self._packed_fields_boundaries()
        octree_slice = slice(octree_bounds[idx], octree_bounds[idx+1])
        octree = self.octrees[octree_slice]
        length = self.lengths[idx:idx + 1]
        pyramids = self._pyramids[idx:idx + 1] if self._pyramids is not None else None
        pyramids_dual = self._pyramids_dual[idx:idx + 1] if self._pyramids_dual is not None else None
        exsum = self.exsum[exsum_bounds[idx]:exsum_bounds[idx+1]] if exsum_bounds is not None else None

        point_hierarchies, parents, features, trinkets = None, None, None, None
        if pts_bounds is not None:
            pt_slice = slice(pts_bounds[idx], pts_bounds[idx+1])
            point_hierarchies = self.point_hierarchies[pt_slice] if self._point_hierarchies is not None else None
            parents = self.parents[pt_slice] if self._parents is not None else None
            features = self.features[pt_slice] if self.features is not None else None
            trinkets = self.trinkets[pt_slice] if self._trinkets is not None else None

        point_hierarchies_dual = None
        if dual_pts_bounds is not None:
            dual_slice = slice(dual_pts_bounds[idx], dual_pts_bounds[idx+1])
            point_hierarchies_dual = self._point_hierarchies_dual[dual_slice] \
                if self._point_hierarchies_dual is not None else None

        return Spc(octrees=octree, lengths=length, max_level=self._max_level, pyramids=pyramids,
                   exsum=exsum, point_hierarchies=point_hierarchies, parents=parents, features=features,
                   pyramids_dual=pyramids_dual, point_hierarchies_dual=point_hierarchies_dual,
                   trinkets=trinkets)