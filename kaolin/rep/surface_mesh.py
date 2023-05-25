# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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
from __future__ import annotations
import copy
import logging
import kaolin
import torch

from enum import Enum
from itertools import chain
from typing import Sequence, Union, Optional

# TODO: this causes circular import. Fix!
# from kaolin.utils.testing import check_tensor, tensor_info
import kaolin.utils.testing


logger = logging.getLogger(__name__)


class SurfaceMesh(object):
    r"""
    This container class manages data attributes (pytorch tensors) of a homogeneous surface mesh
    (i.e. with all faces of an equal number of vertices, such as triangle mesh), or a batch
    of meshes following three :class:`Batching` strategies. ``SurfaceMesh`` allows
    converting between these batching strategies, and automatically computes some
    attributes (such as face normals) on access (see :ref:`supported attributes <rubric mesh attributes>`).
    This data type does not extend to volumetric tetrahedral meshes at this time and has limited
    support for materials.

    .. _rubric mesh overview:

    .. rubric:: Overview

    To construct a ``SurfaceMesh`` object, pass ``vertices`` and ``faces`` (can be 0-length) tensors and
    any other :ref:`supported attributes <rubric mesh attributes>`; batching strategy will be automatically
    determined from the inputs::

        vertices = torch.rand((B, V, 3), dtype=torch.float32, device=device)
        faces = torch.randint(0, V - 1, (F, 3), dtype=torch.long)
        mesh = SurfaceMesh(faces, vertices)

    To load a ``SurfaceMesh`` object::

        from kaolin.io import obj, usd
        mesh = obj.load_mesh(path)
        mesh2 = usd.load_mesh(path2)

    Examine mesh properties::

        >>> print(mesh)  # Note auto-computable attributes
        SurfaceMesh object with batching strategy NONE
                    vertices: [42, 3] (torch.float32)[cpu]
                       faces: [80, 3] (torch.int64)[cpu]
               face_vertices: if possible, computed on access from: (faces, vertices)
                face_normals: if possible, computed on access from: (normals, face_normals_idx) or (vertices, faces)
              vertex_normals: if possible, computed on access from: (faces, face_normals)
                    face_uvs: if possible, computed on access from: (uvs, face_uvs_idx)

        >>> mesh.face_normals  # Causes attribute to be computed
        >>> print(mesh.describe_attribute("face_normals"))
                face_normals: [80, 3, 3] (torch.float32)[cpu]


    To get a sense for what tensors the mesh can contain for different batching strategies see
    :ref:`table below <rubric mesh attributes>`, or run::

        >>> print(SurfaceMesh.attribute_info_string(SurfaceMesh.Batching.FIXED))
        Expected SurfaceMesh contents for batching strategy FIXED
                       faces: (torch.IntTensor)   of shape ['F', 'FSz']
            face_normals_idx: (torch.IntTensor)   of shape ['B', 'F', 'FSz']
                face_uvs_idx: (torch.IntTensor)   of shape ['B', 'F', 'FSz']
        material_assignments: (torch.IntTensor)   of shape ['B', 'F']
                    vertices: (torch.FloatTensor) of shape ['B', 'V', 3]
               face_vertices: (torch.FloatTensor) of shape ['B', 'F', 'FSz', 3]
                     normals: (torch.FloatTensor) of shape ['B', 'VN', 3]
                face_normals: (torch.FloatTensor) of shape ['B', 'F', 'FSz', 3]
              vertex_normals: (torch.FloatTensor) of shape ['B', 'V', 3]
                         uvs: (torch.FloatTensor) of shape ['B', 'U', 2]
                    face_uvs: (torch.FloatTensor) of shape ['B', 'F', 'FSz', 2]
                   materials: non-tensor attribute

    .. note::
        This class is using python `logging`, so set up logging to get diagnostics::

            import logging
            import sys
            logging.basicConfig(level=logging.INFO, stream=sys.stdout)


    .. _rubric mesh attributes:

    .. rubric:: Supported Attributes:

    ``SurfaceMesh`` supports the following attributes, which can be provided to
    the constructor or set on the object. See :ref:`supported batching strategies <rubric mesh batching>`.

    +------------------------------+------------------+--------------------+--------------------+-------------+
    | **Attribute**                | ``Batching.NONE``| ``Batching.FIXED`` | ``Batching.LIST``  | Computable? |
    +------------------------------+------------------+--------------------+--------------------+-------------+
    | vertices                     | V x 3            | B x V x 3          | [V_i x 3]          | N           |
    +------------------------------+------------------+--------------------+--------------------+-------------+
    | face_vertices                | F x FSz x 3      | B x F x FSz x 3    | [F_i x FSz_i x 3]  | Y           |
    +------------------------------+------------------+--------------------+--------------------+-------------+
    | normals                      | VN x 3           | B x VN x 3         | [VN_i x 3]         | N           |
    +------------------------------+------------------+--------------------+--------------------+-------------+
    | face_normals                 | F x FSz x 3      | B x F x FSz x 3    | [F_i x FSz_i x 3]  | Y           |
    +------------------------------+------------------+--------------------+--------------------+-------------+
    | vertex_normals               | V x 3            | B x V x 3          | [V_i x 3]          | Y           |
    +------------------------------+------------------+--------------------+--------------------+-------------+
    | uvs                          | U x 2            | B x U x 2          | [U_i x 2]          | N           |
    +------------------------------+------------------+--------------------+--------------------+-------------+
    | face_uvs                     | F x FSz x 2      | B x F x FSz x 2    | [F_i x FSz_i x 2]  | Y           |
    +------------------------------+------------------+--------------------+--------------------+-------------+
    | faces                        | F x FSz          | F x FSz            | [F_i x FSize]      | N           |
    +------------------------------+------------------+--------------------+--------------------+-------------+
    | face_normals_idx             | F x FSz          | B x F x FSz        | [F_i x FSz]        | N           |
    +------------------------------+------------------+--------------------+--------------------+-------------+
    | face_uvs_idx                 | F x FSz          | B x F x FSz        | [F_i x FSz]        | N           |
    +------------------------------+------------------+--------------------+--------------------+-------------+
    | material_assignments         | F                | B x F              | [F_i]              | N           |
    +------------------------------+------------------+--------------------+--------------------+-------------+
    | materials (non-tensor)       | list             | list of lists      | list of lists      | N           |
    +------------------------------+------------------+--------------------+--------------------+-------------+

    *Legend:* **B** - batch size, **V** - number of vertices, **VN** - number of vertex normals,
    **U** - number of UV coordinates, **F** - number of faces, **FSz** - number of vertices per face,
    **{?}_i** - count for the ith mesh, **[...]** - list of tensors of shapes.

     .. note::

       ``SurfaceMesh`` will not sanity check consistency of manually set attributes.

    .. method:: SurfaceMesh.__init__(vertices, faces, normals = None, uvs = None, face_uvs_idx = None, face_normals_idx = None, material_assignments = None, materials = None, vertex_normals = None, face_normals = None, face_uvs = None, face_vertices = None, strict_checks = True, unset_attributes_return_none = True, allow_auto_compute = True)

    Initializes the object, while automatically detecting the batching strategy
    (see above for expected tensor dimensions).
    The ``vertices`` and ``faces`` tensors are required, but the number of faces/vertices can be 0.
    Any or all of the other attributes can be also provided or set later. See :ref:`table above <rubric mesh attributes>`
    for expected tensor dimensions for different :ref:`batching <rubric mesh batching>` strategies.

    Args:
        vertices:  xyz locations of vertices.
        faces: indices into ``vertices`` array for each vertex of each face; this is
           the only **fixed topology** item for ``Batching.FIXED``.
        face_vertices: xyz locations for each vertex of each face; can be set directly
           or is **auto-computable** by indexing ``vertices`` with ``faces``.
        normals: xyz normal values, indexed by ``face_normals_idx``.
        face_normals: xyz normal values for each face; can be set directly
           or is **auto-computable** by 1) indexing ``normals`` with ``face_normals_idx``,
           or (if either is missing and mesh is triangular) by 2) using
           vertex locations.
        vertex_normals: xyz normal values, corresponding to vertices; can be set directly
           or is **auto-computable** by averaging ``face_normals`` of faces incident
           to a vertex.
        uvs: uv texture coordinates, indexed by ``face_uvs_idx``.
        face_uvs: uv coordinate values for each vertex of each face; can be set directly
           or is **auto-computable** by indexing ``uvs`` with ``face_uvs_idx``.
        face_normals_idx: indices into ``normals`` for each vertex in each face.
        face_uvs_idx: indices into ``uvs`` for each vertex of each face.
        material_assignments: indices into ``materials`` list for each face.
        materials: raw materials as output by the io reader.
        strict_checks: if ``True``, will raise exception if any tensors passed to
           the construcor have unexpected shapes (see :ref:`shapes matrix <rubric mesh attributes>` above);
           note that checks are less strict for ``Batching.LIST`` batching (default: ``True``).
        unset_attributes_return_none: if set to ``False`` exception will be raised when
           accessing attributes that are not set (or cannot be computed), if ``True`` will simply
           return ``None`` (default: ``True``).
        allow_auto_compute: whether to allow auto-computation of attributes on mesh
           attribute access; see :ref:`supported attributes <rubric mesh attributes>`
           (default: ``True``).
    """

    class Batching(str, Enum):
        """Batching strategies supported by the ``SurfaceMesh``."""
        # Note: for python>3.11 can use StrEnum instead
        NONE = "NONE"    #: a single unbatched mesh
        FIXED = "FIXED"  #: a batch of meshes with fixed topology (i.e. same faces array)
        LIST = "LIST"    #: a list of meshes of any topologies

    __material_attributes = ['materials']
    __settings_attributes = ['allow_auto_compute',
                             'unset_attributes_return_none']
    __misc_attributes = ['batching'] + __settings_attributes + __material_attributes
    __float_tensor_attributes = [
        'vertices',
        'face_vertices',
        'normals',
        'face_normals',
        'vertex_normals',
        'uvs',
        'face_uvs',
    ]
    __int_tensor_attributes = [
        'faces',
        'face_normals_idx',
        'face_uvs_idx',
        'material_assignments'
    ]
    __tensor_attributes = __float_tensor_attributes + __int_tensor_attributes

    # Keeping as separate list as things can diverge
    __fixed_topology_attributes = [
        'faces'
    ]

    # This means we cannot set attributes other than these
    __slots__ = __misc_attributes + __tensor_attributes

    @staticmethod
    def assert_supported(attr):
        if attr not in SurfaceMesh.__slots__:
            raise AttributeError(f'SurfaceMesh does not support attribute named "{attr}"')

    def __init__(self, vertices: Union[torch.FloatTensor, list],
                 faces: Union[torch.LongTensor, list],
                 normals: Optional[Union[torch.FloatTensor, list]] = None,
                 uvs: Optional[Union[torch.FloatTensor, list]] = None,
                 face_uvs_idx: Optional[Union[torch.LongTensor, list]] = None,
                 face_normals_idx: Optional[Union[torch.LongTensor, list]] = None,
                 material_assignments: Optional[Union[torch.Tensor, list]] = None,
                 materials: Optional[list] = None,
                 vertex_normals: Optional[Union[torch.FloatTensor, list]] = None,
                 face_normals: Optional[Union[torch.FloatTensor, list]] = None,
                 face_uvs: Optional[Union[torch.FloatTensor, list]] = None,
                 face_vertices: Optional[Union[torch.FloatTensor, list]] = None,
                 strict_checks: bool = True,
                 unset_attributes_return_none: bool = True,
                 allow_auto_compute: bool = True):
        r"""Initializes the surface mesh object, while automatically detecting a batching strategy
         (see :ref:`supported attributes <rubric mesh attributes>` for expected tensor dimensions).
         The `vertices` and `faces` tensors are required, but the number of faces/vertices can be 0.
         Any or all of the other attributes can be also provided or set later.

         .. note::

            ``SurfaceMesh`` will not sanity check consistency of manually set attributes. That is left
            to the user.
        """
        self.unset_attributes_return_none = unset_attributes_return_none
        self.allow_auto_compute = allow_auto_compute

        assert torch.is_tensor(vertices) or type(vertices) is list, f'unsupported vertices type {type(vertices)}'
        assert torch.is_tensor(faces) or type(faces) is list, f'unsupported faces type {type(faces)}'
        if type(vertices) is list or type(faces) is list or len(faces.shape) == 3:
            batching = SurfaceMesh.Batching.LIST
        elif len(vertices.shape) == 3:
            batching = SurfaceMesh.Batching.FIXED
        else:
            batching = SurfaceMesh.Batching.NONE

        self.vertices = vertices
        self.faces = faces
        self.normals = normals
        self.uvs = uvs
        self.face_uvs_idx = face_uvs_idx
        self.face_normals_idx = face_normals_idx
        self.material_assignments = material_assignments
        self.materials = materials
        self.vertex_normals = vertex_normals
        self.face_normals = face_normals
        self.face_uvs = face_uvs
        self.face_vertices = face_vertices
        super().__setattr__('batching', batching)

        ok = self.check_sanity()
        if strict_checks and not ok:
            raise ValueError(f'Illegal inputs passed to SurfaceMesh constructor; check log')

    def check_sanity(self):
        r"""Checks that tensor attribute sizes are consistent for the current batching strategy.
        Will log any inconsistencies.

        Return:
            (bool): true if sane, false if not
        """
        attributes = self.get_attributes(only_tensors=True)

        # Set some known values from current attributes
        known_sizes = {'batchsize': len(self)}

        if 'vertices' in attributes and torch.is_tensor(self.vertices) and self.vertices.numel() > 0:
            if self.batching == SurfaceMesh.Batching.NONE:
                known_sizes['numverts'] = self.vertices.shape[0]
            elif self.batching == SurfaceMesh.Batching.FIXED:
                known_sizes['numverts'] = self.vertices.shape[1]
        if 'faces' in attributes and torch.is_tensor(self.vertices) and self.faces.numel() > 0:
            if self.batching in [SurfaceMesh.Batching.NONE, SurfaceMesh.Batching.FIXED]:
                known_sizes['numfaces'] = self.faces.shape[0]
                known_sizes['facesize'] = self.faces.shape[1]

        res = True
        for attr in attributes:
            res = res and SurfaceMesh.__check_attribute(
                attr, getattr(self, attr), self.batching, batchsize=len(self), log_error=True,
                shape=SurfaceMesh.__expected_shape(attr, self.batching, **known_sizes))
        return res

    @classmethod
    def attribute_info_string(cls, batching: SurfaceMesh.Batching):
        r"""Outputs information about expected mesh contents and tensor sizes, given a batching strategy.
        Only includes tensor and material attributes.

        Args:
            batching (SurfaceMesh.Batching): batching strategy

        Return:
            (str): multi-line string of attributes and their shapes
        """
        def _get_shape(_attr):
            if batching == SurfaceMesh.Batching.LIST:
                return cls.__expected_shape(_attr, batching, batchsize='B', numverts='V_i', numfaces='F_i',
                                            facesize='FSz_i', numnormals='VN_i', numuvs='U_i')
            else:
                return cls.__expected_shape(_attr, batching, batchsize='B', numverts='V', numfaces='F',
                                            facesize='FSz', numnormals='VN', numuvs='U')

        def _format_type(type_str):
            if batching == SurfaceMesh.Batching.LIST:
                return f'[{type_str}]'
            else:
                return f'({type_str})'

        shape_str = 'shapes' if batching == SurfaceMesh.Batching.LIST else 'shape'
        res = [f'Expected SurfaceMesh contents for batching strategy {batching}']
        for attr in cls.__int_tensor_attributes:
            res.append(f'{attr : >20}: {_format_type("torch.IntTensor")}   of {shape_str} {_get_shape(attr)}')
        for attr in cls.__float_tensor_attributes:
            res.append(f'{attr : >20}: {_format_type("torch.FloatTensor")} of {shape_str} {_get_shape(attr)}')
        for attr in sorted(cls.__material_attributes):
            res.append(f'{attr : >20}: non-tensor attribute')
        return '\n'.join(res)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def describe_attribute(self, attr, print_stats=False, detailed=False):
        r"""Outputs an informative string about an attribute; the same method
        used for all attributes in ``to_string``.

         Args:
            print_stats (bool): if to print statistics about values in each tensor
            detailed (bool): if to include additional information about each tensor

        Return:
            (str): multi-line string with attribute information
        """
        SurfaceMesh.assert_supported(attr)

        if not self.has_attribute(attr):
            return 'None'

        val = super().__getattribute__(attr)
        res = ''
        if attr in SurfaceMesh.__tensor_attributes:
            if self.batching == SurfaceMesh.Batching.LIST or type(val) is list:
                res = '\n'.join([f'{attr : >20}: ['] +
                              [kaolin.utils.testing.tensor_info(
                                  x, name=f'{idx : >23}', print_stats=print_stats, detailed=detailed)
                               for idx, x in enumerate(val)] + ['{:>23}'.format(']')])
            else:
                res = kaolin.utils.testing.tensor_info(
                        val, name=f'{attr : >20}', print_stats=print_stats, detailed=detailed)
        elif attr == 'materials':
            if self.batching != SurfaceMesh.Batching.NONE:
                res = '\n'.join(['{: >20}: ['.format('materials')] +
                              [f'{idx : >23}: list of length {len(x)}'
                               for idx, x in enumerate(val)] + ['{:>23}'.format(']')])
            else:
                res = '{: >20}:'.format('materials') + f' list of length {len(val)}'
        else:
            res = '{: >20}: {}'.format(attr, val)
        return res

    def to_string(self, print_stats=False, detailed=False):
        r"""Returns information about tensor attributes currently contained in the mesh as a multi-line string.

        Args:
            print_stats (bool): if to print statistics about values in each tensor
            detailed (bool): if to include additional information about each tensor

        Return:
            (str): multi-line string with attribute information
        """
        attributes = self.get_attributes(only_tensors=True)
        res = [f'SurfaceMesh object with batching strategy {self.batching}']
        for attr in attributes:
            res.append(self.describe_attribute(attr, print_stats=print_stats, detailed=detailed))

        if self.has_attribute('materials'):
            res.append(self.describe_attribute('materials', print_stats=print_stats, detailed=detailed))

        for attr, req in self._get_computable_attributes().items():
            # req is list of lists of required attribute names
            res.append(
                f'{attr : >20}: if possible, computed on access from: ' +
                ' or '.join(['(' + ', '.join(x) + ')' for x in req]))

        return '\n'.join(res)

    def as_dict(self, only_tensors=False):
        """ Returns currently set items as a dictionary. Does not auto-compute any items, but returns raw values.

        Args:
            only_tensors (bool): if true, will only include tensor attributes

        Return:
            (dict): currently set attributes as a dictionary
        """
        # TODO: add options for usd export
        attr = self.get_attributes(only_tensors=only_tensors)
        return {a: getattr(self, a) for a in attr}

    def get_attributes(self, only_tensors=False):
        r"""Returns names of all attributes that are currently set.

        Args:
            only_tensors: if true, will only include tensor attributes

        Return:
           (list): list of string names
        """
        res = []
        options = SurfaceMesh.__tensor_attributes if only_tensors else SurfaceMesh.__slots__
        for attr in options:
            if self.has_attribute(attr):
                res.append(attr)
        return res

    def has_attribute(self, attr: str):
        r"""Checks if a given attribute is present without trying to compute it, if not.

        Args:
            attr: attribute name

        Return:
            (bool): True if attribute is set and is not None
        """
        try:
            super().__getattribute__(attr)
            return True
        except Exception as e:
            return False

    def __deepcopy__(self, memo):
        attr = self.get_attributes()

        kwargs = {a: copy.deepcopy(getattr(self, a), memo) for a in attr}
        del kwargs['batching']
        return SurfaceMesh(**kwargs, strict_checks=False)

    def __copy__(self):
        attr = self.get_attributes()
        kwargs = {}
        for a in attr:
            val = super().__getattribute__(a)
            if torch.is_tensor(val):
                new_val = val
            elif type(val) is list and len(val) > 0 and torch.is_tensor(val[0]):
                new_val = [x for x in val]
            else:
                new_val = copy.copy(val)
            kwargs[a] = new_val
        del kwargs['batching']
        return SurfaceMesh(**kwargs, strict_checks=False)

    def __setattr__(self, attr, value):
        # TODO: Should we add error checks here?
        if attr == 'batching':
            self.set_batching(value)
        elif value is None:
            if self.has_attribute(attr):
                print(f'Deleting {attr}')
                super().__delattr__(attr)
        else:
            super().__setattr__(attr, value)

    def __len__(self):
        if self.batching == SurfaceMesh.Batching.LIST:
            return len(self.vertices)
        elif self.batching == SurfaceMesh.Batching.NONE:
            return 1
        elif self.batching == SurfaceMesh.Batching.FIXED:
            return self.vertices.shape[0]
        else:
            raise NotImplementedError(f'Unsupported batching {self.batching}')

    @staticmethod
    def __expected_shape(name, batching: SurfaceMesh.Batching, batchsize=None,
                         numverts=None, numfaces=None, facesize=None, numnormals=None,
                         numuvs=None):
        r"""Returns expected shape for an attribute, given batching strategy. Additional values
        can be passed in for a more precise check.

        Return:
            (list): expected shape as [None, None...] or with values set from provided inputs
        """
        B = batchsize
        V = numverts
        VN = numnormals
        F = numfaces
        U = numuvs
        FSz = facesize
        Any = None

        shapes = {'vertices':                      [V, 3],
                  'normals':                       [VN, 3],
                  'uvs':                           [U, 2],
                  'vertex_normals':                [V, 3],
                  'face_normals':                  [F, FSz, 3],
                  'face_uvs':                      [F, FSz, 2],
                  'face_vertices':                 [F, FSz, 3],
                  'faces':                         [F, FSz],
                  'material_assignments':          [F],
                  'face_normals_idx':              [F, FSz],
                  'face_uvs_idx':                  [F, FSz]}

        if name not in shapes:
            raise NotImplementedError(f'Cannot get expected shape for attribute {name}')

        if batching in [SurfaceMesh.Batching.NONE, SurfaceMesh.Batching.LIST]:
            return shapes[name]
        elif batching == SurfaceMesh.Batching.FIXED:
            if name in SurfaceMesh.__fixed_topology_attributes:
                return shapes[name]
            else:
                return [B] + shapes[name]
        else:
            raise NotImplementedError(f'Unsupported batching {batching}')

    @staticmethod
    def __check_attribute(name, value, batching, batchsize, log_error=True, **check_tensor_kwargs):
        r"""Checks if a given attribute has expected properties.

        Args:
            name (str): name of the attribute
            value (torch.Tensor): value of the attribute
            batching (SurfaceMesh.Batching): batching strategy
            log_error (bool): if set (default), will log informative message
            check_tensor_kwargs: additional inputs to `kaolin.utils.testing.check_tensor`

        Return:
            (bool): True if tensor passed the test, False otherwise
        """
        def _maybe_log(msg):
            if log_error:
                logger.error(msg)

        check_tensor_kwargs['throw'] = False
        if batching == SurfaceMesh.Batching.LIST:
            if type(value) is not list:
                _maybe_log(f'Attribute {name} must have type list for batching type LIST, but got {type(value)}')
                return False
            if len(value) != batchsize:
                _maybe_log(f'Attribute {name} length {len(value)} does not match batchsize {batchsize} '
                           'for batching type LIST')
                return False
            for i, v in enumerate(value):
                if not torch.is_tensor(v):
                    _maybe_log(f'Expected tensor for {name}[i], but got {type(v)}')
                    return False
                if not kaolin.utils.testing.check_tensor(v, **check_tensor_kwargs):
                    _maybe_log(f'Attribute {name}[i] for batching type LIST has unexpected '
                               f'value {kaolin.utils.testing.tensor_info(v)} vs {check_tensor_kwargs}')
                    return False
        elif batching in [SurfaceMesh.Batching.NONE, SurfaceMesh.Batching.FIXED]:
            if not torch.is_tensor(value):
                _maybe_log(f'Expected tensor for {name}, but got {type(value)}')
                return False
            if not kaolin.utils.testing.check_tensor(value, **check_tensor_kwargs):
                _maybe_log(f'Attribute {name} for batching type {batching} has unexpected '
                           f'value {kaolin.utils.testing.tensor_info(value)} vs {check_tensor_kwargs}')
                return False
        else:
            raise NotImplementedError(f'Unsupported batching {batching}')
        return True

    @staticmethod
    def convert_attribute_batching(val: Union[torch.Tensor, list],
                         from_batching: SurfaceMesh.Batching, to_batching: SurfaceMesh.Batching,
                         is_tensor: bool = True, fixed_topology: bool = False,
                         batch_size: int = None):
        r"""Converts tensors between different :class:`SurfaceMesh.Batching` strategies. The input value
        is expected to respect the provided ``from_batching``. Will fail if conversion cannot be done

        Approximate summary of conversions for tensor values:
            * ``NONE`` -> ``LIST``: return ``[val]``
            * ``NONE`` -> ``FIXED``: return ``val.unsqueeze(0)`` unless `fixed_topology`
            * ``LIST`` -> ``NONE``: return ``val[0]``, fails if list longer than `1`
            * ``LIST`` -> ``FIXED``: return ``torch.stack(val)`` (or ``val[0]`` if `fixed_topology`)
            * ``FIXED`` -> ``NONE``: return ``val.squeeze(0)`` unless `fixed_topology`, fails if list longer than `1`
            * ``FIXED`` -> ``LIST``: return ``[val[i, ...] for i ...]`` (or ``[val for i ...]`` if `fixed_topology`)

        Non-tensor values are stored as lists for ``FIXED`` and ``LIST`` batching.

        .. note::
            This method is only useful for converting batching of custom attributes
            and is not needed if only working with attributes natively supported
            by the ``SurfaceMesh``.

        Args:
            val: value to convert, must be consistent with ``from_batching``
            from_batching: batching type to convert from
            to_batching: batching type to convert to
            is_tensor: if the converted value is a tensor attribute (and not e.g. unstructured value to store in lists)
            fixed_topology: if the attribute should be the same across items in a ``FIXED`` batching
            batch_size: desirable batch size; must be consistent with ``val`` (will be guessed in most cases, but when
                converting ``fixed_topology`` items to e.g. ``LIST`` batching, this value is needed)
        """
        batch_size_guess = None
        if from_batching == SurfaceMesh.Batching.LIST:
            batch_size_guess = len(val)
        elif from_batching == SurfaceMesh.Batching.NONE:
            batch_size_guess = 1
        elif from_batching == SurfaceMesh.Batching.FIXED:
            if is_tensor and not fixed_topology:
                batch_size_guess = val.shape[0]
            elif not is_tensor:
                batch_size_guess = len(val)

        if batch_size is not None:
            if batch_size_guess is not None and batch_size != batch_size_guess:
                raise ValueError(f'Provided batch size {batch_size} disagrees with value {batch_size_guess} guessed from input {val}')
        else:
            if batch_size_guess is None:
                batch_size_guess = 1
            batch_size = batch_size_guess

        if from_batching == to_batching:
            return val
        elif batch_size == 0:
            return val  # TODO: support empty batches
        elif not is_tensor:
            # Material and other non-tensor attributes kept as lists for LIST and FIXED batching
            if to_batching == SurfaceMesh.Batching.NONE:
                if batch_size == 1:
                    val = val[0]
                else:
                    raise ValueError(f'Cannot return unbatched non-tensor attribute from batch of length {batch_size}')
            elif from_batching == SurfaceMesh.Batching.NONE:
                val = [val]
        elif type(val) is list or torch.is_tensor(val):
            if to_batching == SurfaceMesh.Batching.NONE:
                if batch_size != 1:
                    raise ValueError(f'Cannot return unbatched tensor attribute from batch of length {batch_size}')
                if from_batching == SurfaceMesh.Batching.LIST:
                    val = val[0]
                elif from_batching == SurfaceMesh.Batching.FIXED:
                    if not fixed_topology:
                        val = val.squeeze(0)
                else:
                    raise NotImplementedError(f'Unsupported batching {from_batching}')
            elif to_batching == SurfaceMesh.Batching.FIXED:
                if from_batching == SurfaceMesh.Batching.NONE:
                    if not fixed_topology:
                        val = val.unsqueeze(0)
                elif from_batching == SurfaceMesh.Batching.LIST:
                    if fixed_topology:
                        for i in range(1, batch_size):
                            assert torch.allclose(val[0], val[i]), f'Fixed topology attribute must be equivalent for all meshes'
                        val = val[0]
                    else:
                        val = torch.stack(val)
                else:
                    raise NotImplementedError(f'Unsupported_batching {from_batching}')
            elif to_batching == SurfaceMesh.Batching.LIST:
                if from_batching == SurfaceMesh.Batching.NONE:
                    val = [val]
                elif from_batching == SurfaceMesh.Batching.FIXED:
                    if fixed_topology:
                        val = [val for i in range(batch_size)]
                    else:
                        val = [val[i, ...] for i in range(batch_size)]
                else:
                    raise NotImplementedError(f'Unsupported_batching {from_batching}')
            else:
                raise NotImplementedError(f'Unsupported_batching {to_batching}')
        return val

    def getattr_batched(self, attr: str, batching: SurfaceMesh.Batching):
        r"""Same as `getattr` or `mesh.attr`, but converts the attribute value
        to desired batching strategy before returning.

        All conversions are supported, except the following:
           * to NONE from FIXED or LIST batch with more than one mesh
           * to FIXED from LIST batch where fixed topology items are different

        Args:
            attr (str): attribute name
            batching (SurfaceMesh.Batching): desirable batching strategy.

        Return:
            attribute value
        """
        val = getattr(self, attr)
        is_material = attr in SurfaceMesh.__material_attributes
        is_tensor = attr in SurfaceMesh.__tensor_attributes

        if not is_material and not is_tensor:
            return val

        return SurfaceMesh.convert_attribute_batching(
            val, from_batching=self.batching, to_batching=batching,
            is_tensor=is_tensor, fixed_topology=(attr in SurfaceMesh.__fixed_topology_attributes),
            batch_size=len(self))

    def to_batched(self):
        r"""Convenience shorthand to convert unbatched mesh to FIXED topology batched mesh.
        Modifies the mesh in place and returns self.

        Return:
           (self)
        """
        return self.set_batching(batching=SurfaceMesh.Batching.FIXED)

    def set_batching(self, batching: SurfaceMesh.Batching, skip_errors=False):
        r"""Converts a mesh to a different batching strategy. Modifies the mesh
        in place and returns self.

        All conversions are supported, except the following:
           * to NONE from FIXED or LIST batch with more than one mesh
           * to FIXED from LIST batch where fixed topology items are different

        Args:
            batching (SurfaceMesh.Batching): desirable batching strategy.
            skip_errors: if true, will simply unset attributes that cannot
               be converted (useful if e.g. vertices are of fixed topology, but
               meshes have variable number of normals that cannot be stacked)

        Return:
            (self)
        """
        if self.batching == batching:
            return self

        if len(self) == 0:
            return self

        if batching == SurfaceMesh.Batching.NONE and len(self) != 1:
            raise ValueError(f'Cannot create an unbatched mesh from {len(self)} meshes')

        new_attr = {}
        attrs_to_process = self.get_attributes(only_tensors=True) + \
            [x for x in SurfaceMesh.__material_attributes if self.has_attribute(x)]
        for attr in attrs_to_process:
            try:
                val = self.getattr_batched(attr, batching)
            except Exception as e:  # TODO: what's the right error to catch?
                logger.error(f'Failed to convert attribute {attr} with error {e}')
                if skip_errors and attr not in ['vertices', 'faces']:  # required attrs
                    val = None
                else:
                    raise ValueError(f'Cannot convert {attr} to batching {batching} due to: {e}')
            new_attr[attr] = val

        # Set attributes (to avoid messing up internal state while getting attributes in previous loop)
        for attr, val in new_attr.items():
            if val is None:
                delattr(self, attr)
            else:
                setattr(self, attr, val)

        super().__setattr__('batching', batching)
        return self

    @classmethod
    def cat(cls, meshes: Sequence[SurfaceMesh], fixed_topology: bool = True, skip_errors: bool = False):
        r"""Concatenates meshes or batches of meshes to create a FIXED (if fixed_topology)
        or LIST batched mesh. Only attributes present in all the meshes will be
        preserved.

        Args:
           meshes: meshes to concatenate; any batching is supported
           fixed_topology: if to create a FIXED batched mesh (input must comply to assumptions)
           skip_errors: if True, will not fail if some of the attributes fail to convert to target batching

        Return:
           (SurfaceMesh): new mesh
        """
        def _get_joint_attrs():
            _attrs = set(meshes[0].get_attributes())
            for i in range(1, len(meshes)):
                _attrs.intersection_update(meshes[i].get_attributes())
            return _attrs

        def _attr_from_meshes(in_attr):
            return list(chain.from_iterable(
                        [m.getattr_batched(in_attr, SurfaceMesh.Batching.LIST) for m in meshes]))

        batchable_attributes = SurfaceMesh.__tensor_attributes + SurfaceMesh.__material_attributes

        if len(meshes) == 0:
            raise ValueError('Zero length list provided to cat operation; at least 1 mesh input required')
        elif len(meshes) == 1:
            res = meshes[0]
        else:
            # Convert all meshes to LIST and create a LIST mesh
            # TODO: this could be more efficient for special cases
            attrs = _get_joint_attrs()
            args = {}
            for attr in attrs:
                if attr in batchable_attributes:
                    args[attr] = _attr_from_meshes(attr)
                elif attr in SurfaceMesh.__settings_attributes:
                    args[attr] = getattr(meshes[0], attr)  # Take first mesh's value

            if fixed_topology:
                # Handle indexed attributes that may not concatenate even for fixed topology meshes
                for indexed_attr in ['normals', 'uvs']:
                    if indexed_attr in args:
                        try:
                            stacked = torch.stack(args[indexed_attr])
                        except Exception as e:
                            logger.warning(f'Cannot cat {indexed_attr} arrays of given shapes; '
                                           f'trying to concatenate face_{indexed_attr} instead, due to: {e}')

                            # Delete indexed attribute and the index that can't be concatenated (e.g. uvs, face_uvs_idx)
                            del args[indexed_attr]
                            face_index_attr = f'face_{indexed_attr}_idx'
                            if face_index_attr in args:
                                del args[face_index_attr]

                            # Auto-compute full attribute value per face instead (this can be concatenated as long as
                            # the number of faces matches)
                            face_attr = f'face_{indexed_attr}'
                            try:
                                args[face_attr] = _attr_from_meshes(face_attr)
                            except Exception as e:
                                logger.warning(f'Cannot compute {face_attr} for all concatenated meshes: {e}')

            res = SurfaceMesh(**args)

        target_batching = SurfaceMesh.Batching.FIXED if fixed_topology else SurfaceMesh.Batching.LIST
        res.set_batching(target_batching, skip_errors=skip_errors)

        return res

    def _requires_grad(self, value):
        res = False
        if torch.is_tensor(value):
            return value.requires_grad
        elif self.batching == SurfaceMesh.Batching.LIST and type(value) is list:
            for v in value:
                if torch.is_tensor(v):
                   res = res or v.requires_grad
                else:
                    logger.warning(f'Unexpected type passed to requires_grad {v}')
        else:
            logger.warning(f'Unexpected type passed to requires_grad {value}')
        return res

    def _index_value_by_faces(self, values, face_idx):
        """
        Args:
            values: NV x C or
        """
        can_cache = not self._requires_grad(values) and not self._requires_grad(face_idx)
        res = None

        if self.batching == SurfaceMesh.Batching.NONE:
            res = kaolin.ops.mesh.index_vertices_by_faces(values.unsqueeze(0), face_idx).squeeze(0)
        elif self.batching == SurfaceMesh.Batching.FIXED:
            # only faces have fixed topology
            if len(face_idx.shape) == 2:
                # Use fixed topology method
                res = kaolin.ops.mesh.index_vertices_by_faces(values, face_idx)
            else:
                # TODO: add a more flexible index_values_by_face_idx utility
                res = torch.cat([
                    kaolin.ops.mesh.index_vertices_by_faces(values[i:i+1, ...], face_idx[i, ...])
                    for i in range(len(self))], dim=0)
        elif self.batching == SurfaceMesh.Batching.LIST:
            res = [kaolin.ops.mesh.index_vertices_by_faces(values[i].unsqueeze(0), face_idx[i]).squeeze(0)
                   for i in range(len(self))]
        else:
            raise NotImplementedError(f'Unsupported batching {self.batching}')
        return res, can_cache

    def _compute_face_normals_from_vertices(self, should_cache=None):
        args = {'unit': True}
        face_vertices = self.get_or_compute_attribute('face_vertices', should_cache=should_cache)
        can_cache = not self._requires_grad(face_vertices)

        if self.batching == SurfaceMesh.Batching.NONE:
            # When computed this way, there is only one normal per face
            res = kaolin.ops.mesh.face_normals(
                face_vertices.unsqueeze(0), **args).squeeze(0).unsqueeze(1).repeat((1, 3, 1))
        elif self.batching == SurfaceMesh.Batching.FIXED:
            res = kaolin.ops.mesh.face_normals(
                face_vertices, **args).unsqueeze(2).repeat((1, 1, 3, 1))
        elif self.batching == SurfaceMesh.Batching.LIST:
            res = [kaolin.ops.mesh.face_normals(
                face_vertices[i].unsqueeze(0), **args).squeeze(0).unsqueeze(1).repeat((1, 3, 1))
                    for i in range(len(self))]
        else:
            raise NotImplementedError(f'Unsupported batching {self.batching}')
        return res, can_cache

    def _compute_face_uvs(self):
        return self._index_value_by_faces(self.uvs, self.face_uvs_idx)

    def _compute_face_vertices(self):
        return self._index_value_by_faces(self.vertices, self.faces)

    def _compute_face_normals(self, should_cache=None):
        if self.has_attribute('normals') and self.has_attribute('face_normals_idx'):
            return self._index_value_by_faces(self.normals, self.face_normals_idx)
        elif self.has_attribute('face_vertices') or (self.has_attribute('vertices') and self.has_attribute('faces')):
            return self._compute_face_normals_from_vertices(should_cache=should_cache)
        else:
            raise RuntimeError(f'This is a bug, _compute_face_normals should never be called if not computable')

    def _compute_vertex_normals(self, should_cache=None):
        # for each vertex, accumulate normal for every face that has it
        face_normals = self.get_or_compute_attribute('face_normals', should_cache=should_cache)

        can_cache = not self._requires_grad(self.faces) and not self._requires_grad(face_normals)

        if self.batching == SurfaceMesh.Batching.NONE:
            res = kaolin.ops.mesh.compute_vertex_normals(self.faces, face_normals.unsqueeze(0),
                                                         num_vertices=self.vertices.shape[0]).squeeze(0)
        elif self.batching == SurfaceMesh.Batching.FIXED:
            res = kaolin.ops.mesh.compute_vertex_normals(self.faces, face_normals,
                                                         num_vertices=self.vertices.shape[1])
        elif self.batching == SurfaceMesh.Batching.LIST:
            res = [kaolin.ops.mesh.compute_vertex_normals(self.faces[i], face_normals[i].unsqueeze(0),
                                                          num_vertices=self.vertices[i].shape[0]).squeeze(0)
                   for i in range(len(self))]
        else:
            raise NotImplementedError(f'Unsupported batching {self.batching}')
        return res, can_cache

    def _compute_computable_attribute(self, attr, should_cache=None):
        if attr == 'vertex_normals':
            return self._compute_vertex_normals(should_cache=should_cache)
        elif attr == 'face_normals':
            return self._compute_face_normals(should_cache=should_cache)
        elif attr == 'face_uvs':
            return self._compute_face_uvs()
        elif attr == 'face_vertices':
            return self._compute_face_vertices()
        else:
            logger.error(f'This is a bug; {attr} detected as computable, but computation not implemented')
            return None, False

    def has_or_can_compute_attribute(self, attr: str):
        """Returns true if this attribute is set or has all the requirements to be computed. Note that actual
        computation may still fail at run time.

        Args:
            attr: attribute name to check
        Return:
            (bool): True if exists or likely to be computable.
        """
        SurfaceMesh.assert_supported(attr)
        return self._has_or_can_compute_attr(attr)

    def probably_can_compute_attribute(self, attr: str):
        """Checks that the attributes required for computing attribute exist and returns true if the
        attribute is likely to be computable (not that it is not possible to determine this for sure
        without actually computing the attribute, as there could be runtime errors that occur during
        computation).

        Args:
            attr: attribute name to check
        Return:
            (bool) True if likely to be computable
        """
        SurfaceMesh.assert_supported(attr)
        return self._can_compute_attr(attr)[0]

    def _has_or_can_compute_attr(self, attr, allowed_recursion=2):
        if self.has_attribute(attr):
            return True
        return self._can_compute_attr(attr, allowed_recursion=allowed_recursion)[0]

    def _can_compute_attr(self, attr, allowed_recursion=2):
        """ Returns if attribute is already set or has all requirements to be computed. Note that
        some requirements can be computed from each other, potentially causing this method to be called
        recursively in an inifinite loop. We limit depth of recursion to a reasonable value.
        """
        if allowed_recursion < 0:
            return False, ''

        computable = self._get_computable_attributes()
        if attr not in computable:
            return False, ''

        can_compute = False
        req_str = ''
        for requirements_list in computable[attr]:
            can_compute = True
            for required_attr in requirements_list:
                if not self._has_or_can_compute_attr(required_attr, allowed_recursion=allowed_recursion-1):
                    can_compute = False
                    break
            if can_compute:
                req_str = f'{requirements_list}'
                break
        if not can_compute:
            req_str = ' or '.join(str(x) for x in computable[attr])

        return can_compute, req_str

    def _check_compute_attribute(self, attr, should_cache=None):
        """Checks if missing attribute can be computed and attempts to compute it if it is."""
        SurfaceMesh.assert_supported(attr)

        throw_info_str = 'To return None instead of throwing, set mesh.unset_attributes_return_none=True'

        # See if we can compute the attribute and issue informative message
        can_compute, req_str = self._can_compute_attr(attr)
        if not can_compute:
            info_str = f'Attribute "{attr}" has not been set and does not have required attributes to be computed: {req_str}'
            if self.unset_attributes_return_none:
                logger.debug(info_str)
                return None
            raise AttributeError(f'{info_str}\n{throw_info_str}')

        # If we can compute, let's compute
        logger.debug(f'Automatically computing {attr} based on {req_str}')
        try:
            computed, auto_should_cache = self._compute_computable_attribute(attr, should_cache=should_cache)
            if should_cache or (should_cache is None and auto_should_cache):
                setattr(self, attr, computed)
            return computed
        except Exception as e:
            info_str = f'Attribute "{attr}" has not been set and failed to be computed due to: {e}'
            if self.unset_attributes_return_none:
                logger.warning(info_str)
                return None
            raise AttributeError(f'{info_str}\n{throw_info_str}')

    def get_or_compute_attribute(self, attr: str, should_cache: Optional[bool] = None):
        """ Gets or computes an attribute, while allowing explicit control of caching of the computed value.
        If attribute is not set and cannot be computed will either return ``None`` if
        ``mesh.unset_attributes_return_none`` or raise an exception.

        Args:
            attr: attribute name, see :ref:`attributes <rubric mesh attributes>`
            should_cache: if ``True``, will cache attribute if it was computed; if ``False``, will not cache;
                by default will decide if to cache based on ``requires_grad`` of variables used in computation
                (will not cache if any has ``requires_grad is True``).

        Return:
            attribute value
        """
        if self.has_attribute(attr):
            return getattr(self, attr)

        return self._check_compute_attribute(attr, should_cache=should_cache)

    def get_attribute(self, attr: str):
        """ Gets attribute without any auto-computation magic. If attribute is not set will either return ``None``
        if ``mesh.unset_attributes_return_none`` or raise an exception.

        Args:
            attr: attribute name, see :ref:`attributes <rubric mesh attributes>`

        Return:
            attribute value

        Raises:
            AttributeError: if attribute nate is not supported, or if attribute is not set
                and ``not mesh.unset_attributes_return_none``
        """
        if self.has_attribute(attr):
            return getattr(self, attr)

        SurfaceMesh.assert_supported(attr)

        throw_info_str = 'To return None instead of throwing, set mesh.unset_attributes_return_none=True'
        info_str = f'Attribute "{attr}" has not been set'
        if self.unset_attributes_return_none:
            logger.debug(info_str)
            return None
        raise AttributeError(f'{info_str}\n{throw_info_str}')

    def __getattr__(self, attr):
        # Note: this is only called if super().__getattribute__(attr) failed
        SurfaceMesh.assert_supported(attr)

        throw_info_str = 'To return None instead of throwing, set mesh.unset_attributes_return_none=True'

        # If auto-compute disallowed
        if not self.allow_auto_compute:
            info_str = f'Attribute "{attr}" has not been set and allow_auto_compute is off'
            if self.unset_attributes_return_none:
                logger.debug(info_str)
                return None
            raise AttributeError(f'{info_str}\n{throw_info_str}')

        return self._check_compute_attribute(attr)

    def _get_computable_attributes(self):
        r"""Returns attributes that are currently not set but could be computed using existing attributes."""
        exist = self.get_attributes(only_tensors=True)
        computable = {}
        for attr in SurfaceMesh.__tensor_attributes:
            if attr not in exist:
                if attr == 'vertex_normals':
                    computable[attr] = [['faces', 'face_normals']]
                elif attr == 'face_normals':
                    # There are two ways to compute face normals
                    computable[attr] = [['normals', 'face_normals_idx'], ['vertices', 'faces']]
                elif attr == 'face_uvs':
                    computable[attr] = [['uvs', 'face_uvs_idx']]
                elif attr == 'face_vertices':
                    computable[attr] = [['faces', 'vertices']]
        return computable

    def cuda(self, device=None, attributes=None):
        """Calls cuda on all or only on select tensor attributes, returns a copy of self.

        Args:
            device: device to set
            attributes (list of str): if set, will only call cuda() on select attributes
        Return:
            (SurfaceMesh) shallow copy, with the exception of attributes that were converted
        """
        return self._construct_apply(lambda t: t.cuda(device), attributes)

    def cpu(self, attributes=None):
        """Calls cpu() on all or only on select tensor attributes, returns a copy of self.

        Args:
            attributes (list of str): if set, will only call cpu() on select attributes
        Return:
            (SurfaceMesh) shallow copy, with the exception of attributes that were converted
        """
        return self._construct_apply(lambda t: t.cpu(), attributes)

    def float_tensors_to(self, float_dtype):
        """Converts all floating point tensors to the provided type; returns shallow copy.

        Args:
            float_dtype: torch dtype such as torch.float16, torch.float32
        Return:
            (SurfaceMesh) shallow copy, with the exception of attributes that were converted
        """
        attributes = set(self.get_attributes(only_tensors=True))
        attributes.intersection_update(SurfaceMesh.__float_tensor_attributes)
        return self._construct_apply(lambda t: t.to(float_dtype), attributes)

    def detach(self, attributes=None):
        """Detaches all or select attributes in a shallow copy of self.

        Args:
            attributes (list of str): if set, will only call cuda on select attributes
        Return:
            (SurfaceMesh) shallow copy, with the exception of attributes that were converted
        """
        return self._construct_apply(lambda t: t.detach(), attributes)

    def _construct_apply(self, func, attributes=None):
        """
        Creates a shallow copy of self, applies func() to all (or specified) tensor attributes in the copy,
        for example converting to cuda.
        """
        if attributes is None:
            attributes = self.get_attributes(only_tensors=True)

        my_copy = copy.copy(self)
        for attr in attributes:
            current_val = getattr(my_copy, attr)
            if self.batching == SurfaceMesh.Batching.LIST:
                updated_val = [func(x) for x in current_val]
            else:
                updated_val = func(current_val)
            setattr(my_copy, attr, updated_val)
        return my_copy

    def to(self, device, attributes=None):
        """Converts all or select tensor attributes to provided device; returns copy of self.

         Args:
            device (str, torch.device): device to call torch tensors' ``to`` method with
            attributes (list of str): if set, will only convert select attributes

         Return:
            (SurfaceMesh) shallow copy, with the exception of attributes that were converted
        """
        return self._construct_apply(lambda t: t.to(device), attributes)


# TODO: consider the following API improvements ------------------------------------------------------------------------
# 1. also implement __getitem__ to access element of a batch
#     # def __getitem__(self, idx):
#     #     """Indexes a specific subset of the meshes."""
#     #     return IndexedMesh(self, idx)
#     #     # mesh = mesh[indices].bake()
# 2. add basic operations for convenience, such as:
#     # def subdivide(self, in_place=True):
#     #     pass
#     # # The only operations taking in an unstructured list are exports
#     # def export_obj(self, TBD):
#     #     # Throws error if batched
#     #     pass
#     # def export_usd(self, TBD):
#     #     pass
#     # def add_to_timelapse(self, timelapse):
# 3. support custom face and vertex attributes, such as vertex colors
#         Custom attributes:
#            * custom_vertex_attributes - custom per-vertex attribute with any channel number
#            * custom_face_attributes - custom per-face attribute with any channel number
#            * custom_face_vertex_attributes - custom attribute per each vertex of each face
#         Shapes:
#         custom_vertex_attributes      | V x Any          | B x V x Any        | [V_i x Any_i]
#         custom_face_attributes        | F x Any2         | B x F x Any2       | [F_i x Any2_i]
#         custom_face_vertex_attributes | F x FSz x Any3   | B x F x FSz x Any3 | [F_i x FSz_i x Any3_i]
# 4. consider auto-computing face_{normals,uvs}_idx and {normals,uvs} from face_{normals,uvs} -- i.e.
#    reverse indexing, in order to support scenarios, where indexing is required by the API, such as
#    nvdiffrast, avoiding clunky conditionals like this:
#    if 'normals' in mesh.get_attributes(only_tensors=True):
#         im_world_normals = nvdiffrast.torch.interpolate(
#             mesh.normals, rast[0], mesh.face_normals_idx.int())[0]
#     else:
#         im_world_normals = nvdiffrast.torch.interpolate(
#             mesh.face_normals.reshape(batch_size, -1, 3), rast[0],
#             torch.arange(mesh.faces.shape[0] * 3, device='cuda', dtype=torch.int).reshape(-1, 3)
# 5. support hihg-level rendering function for mesh batches
