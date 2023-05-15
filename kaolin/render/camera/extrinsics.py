# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
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
from typing import Union, Tuple, Iterable, Sequence, List, Dict
import functools
import numpy as np
import torch
from torch.types import _float, _bool
from . import extrinsics_backends
from .extrinsics_backends import ExtrinsicsParamsDefEnum, ExtrinsicsRep, _REGISTERED_BACKENDS


__all__ = [
    'CameraExtrinsics',
    'ExtrinsicsRep',
    'register_backend'
]

_HANDLED_TORCH_FUNCTIONS = dict()   # torch compatible functions are registered here
default_dtype = torch.get_default_dtype()
default_device = 'cpu'

def implements(torch_function):
    """Registers a torch function override for CameraExtrinsics"""
    @functools.wraps(torch_function)
    def decorator(func):
        _HANDLED_TORCH_FUNCTIONS[torch_function] = func
        return func
    return decorator

def register_backend(name: str):
    """Registers a representation backend class with a unique name.
    
    CameraExtrinsics can switch between registered representations dynamically (see :func:`switch_backend()`).
    """
    return extrinsics_backends.register_backend(name)

class CameraExtrinsics():
    r"""Holds the extrinsics parameters of a camera: position and orientation in space.

    This class maintains the view matrix of camera, used to transform points from world coordinates
    to camera / eye / view space coordinates.

    This view matrix maintained by this class is column-major, and can be described by the 4x4 block matrix:

    .. math::

        \begin{bmatrix}
            R & t \\
            0 & 1
        \end{bmatrix}

    where **R** is a 3x3 rotation matrix and **t** is a 3x1 translation vector for the orientation and position
    respectively.

    This class is batched and may hold information from multiple cameras.

    :class:`CameraExtrinsics` relies on a dynamic representation backend to manage the tradeoff between various choices
    such as speed, or support for differentiable rigid transformations.
    Parameters are stored as a single tensor of shape :math:`(\text{num_cameras}, K)`,
    where K is a representation specific number of parameters.
    Transformations and matrices returned by this class support differentiable torch operations,
    which in turn may update the extrinsic parameters of the camera::

                                 convert_to_mat
            Backend                 ---- >            Extrinsics
        Representation R                             View Matrix M
        Shape (num_cameras, K),                    Shape (num_cameras, 4, 4)
                                    < ----
                                 convert_from_mat

    .. note::

        Unless specified manually with :func:`switch_backend`,
        kaolin will choose the optimal representation backend depending on the status of ``requires_grad``.
    .. note::

        Users should be aware, but not concerned about the conversion from internal representations to view matrices.
        kaolin performs these conversions where and if needed.

    Supported backends:

        - **"matrix_se3"**\: A flattened view matrix representation, containing the full information of
          special euclidean transformations (translations and rotations).
          This representation is quickly converted to a view matrix, but differentiable ops may cause
          the view matrix to learn an incorrect, non-orthogonal transformation.
        - **"matrix_6dof_rotation"**\: A compact representation with 6 degrees of freedom, ensuring the view matrix
          remains orthogonal under optimizations. The conversion to matrix requires a single Gram-Schmidt step.

        .. seealso::

            `On the Continuity of Rotation Representations in Neural Networks, Zhou et al. 2019
            <https://arxiv.org/abs/1812.07035>`_

    Unless stated explicitly, the definition of the camera coordinate system used by this class is up to the
    choice of the user.
    Practitioners should be mindful of conventions when pairing the view matrix managed by this class with a projection
    matrix.
    """
    DEFAULT_BACKEND = 'matrix_se3'
    DEFAULT_DIFFERENTIABLE_BACKEND = 'matrix_6dof_rotation'

    def __init__(self, backend: ExtrinsicsRep, shared_fields: dict = None):
        """
        Constructs the camera extrinsics with a representation backend.

        .. warning::

            !! ``__init__`` should not be called directly !!
            See other convenience constructors below to build the extrinsics given some parameters:

                * :func:`from_lookat`:
                  constructs the extrinsics module from look, at and pos vectors.
                * :func:`from_camera_pose`:
                  constructs the extrinsics module from the camera position & orientation.
                * :func:`from_view_matrix`:
                  constructs the extrinsics module from a 4x4 view matrix.

        Args:
            backend ExtrinsicsRep Representation backend
            shared_fields Dictionary of values that should be shared when "views" or shallow copies of the
                ``CameraExtrinsics`` class are created. Changes made to those fields are reflected in all copies.
        """
        self._backend = backend

        # _shared_fields ensures that views created on this instance will mirror any changes back
        # These fields can be accessed as simple properties
        if shared_fields is not None:
            # Another object have shared its fields, access them through the dict
            self._shared_fields = shared_fields
        else:
            self._shared_fields = dict(
                # 3x3 matrix for bookkeeping coordinate system changes performed
                base_change_matrix=torch.eye(3, device=self.device, dtype=self.dtype),

                # True only when a specific backend was explicitly requested by a user,
                # either during construction or by invoking switch_backend().
                # In this mode kaolin will not attempt to optimize for the best representation backend
                # to the current state but assume the user is responsible for setting one according to their needs
                user_requested_backend=False
            )

    @torch.no_grad()
    def _internal_switch_backend(self, backend_name: str):
        """
        Switches the representation backend to a different implementation.
        'backend_name' must be a registered backend.
        
        .. note::

            This function does not allow gradient flow, as it is error prone.
        """
        assert backend_name in _REGISTERED_BACKENDS,\
            f"CameraExtrinsics attempted to switch internal representation to an " \
            f"unregistered backend: {backend_name}. Valid values are registered backends: {self.available_backends()}"
        mat = self._backend.convert_to_mat()
        backend_cls = _REGISTERED_BACKENDS[backend_name]
        self._backend = backend_cls.from_mat(mat,
                                             dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        self._shared_fields = self._shared_fields.copy()    # Detach from shared fields of previous views

    @torch.no_grad()
    def switch_backend(self, backend_name: str):
        """Switches the representation backend to a different implementation.

        .. note::

            Manually switching the representation backend will hint kaolin it should turn off automatic backend
            selection. Users should normally use this manual feature if they're testing a new type of representation.
            For most use cases, it is advised to let kaolin choose the representation backend automatically,
            and avoid using this function explicitly.

        .. warning::

            This function does not allow gradient flow, as it is error prone.

        Args:
            backend_name (str):
                the backend to switch to, must be a registered backend.
                Values supported by default\: ``matrix_se3``, ``matrix_6dof_rotation`` (see class description).
        """
        self._internal_switch_backend(backend_name=backend_name)
        self._shared_fields['user_requested_backend'] = True

    @classmethod
    def _make_backend(cls, mat: torch.Tensor,
                      dtype: torch.dtype = default_dtype,
                      device: Union[torch.device, str] = None,
                      requires_grad: bool = False, backend_name: str = None):
        """
        Creates representation backend from given (C, 4, 4) view matrix and type parameters.
        """
        # Batchify
        if mat.ndim == 2:
            mat = mat.unsqueeze(0)

        # If a backend name is explicitly requested, use that one
        if backend_name is not None:
            assert backend_name in _REGISTERED_BACKENDS,\
                f'CameraExtrinsics tried to use backend: {backend_name},' \
                f'which is not registered. Available backends: {cls.available_backends()}'
        else:
            # If no backend was specified, then by default we choose one which is optimal for torch differentiable ops
            if requires_grad:
                backend_name = CameraExtrinsics.DEFAULT_DIFFERENTIABLE_BACKEND
            else:
                backend_name = CameraExtrinsics.DEFAULT_BACKEND
        backend_class = _REGISTERED_BACKENDS[backend_name]
        backend = backend_class.from_mat(mat, dtype, device, requires_grad)
        return backend

    @classmethod
    def _from_world_in_cam_coords(cls, rotation: torch.Tensor, translation: torch.Tensor,
                                  dtype: torch.dtype = default_dtype,
                                  device: Union[torch.device, str] = None,
                                  requires_grad: bool = False, backend_name: str = None) -> CameraExtrinsics:
        """Constructs the extrinsics from a rigid transformation describing
        how the world is transformed relative to the camera.

        Essentially, this constructor builds the extrinsic matrix directly from
        the world origin and directions of world axes in camera coordinates.

        Args:
            rotation (torch.Tensor):
                of shape [C]x3x3, for rotating the world to align with camera coordinates
            translation (torch.Tensor):
                of shape [C]x3 or [C]x3x1, for translating the world to align with camera coordinates
            device (str):
                The CameraExtrinsics object will manage torch tensors on this device
            requires_grad (bool):
                Sets the requires_grad field for the params tensor of the CameraExtrinsics
            backend (str):
                The backend used to manage the internal representation of the extrinsics, and how it is converted
                to a view matrix.
                Different representations are tuned to varied use cases:
                speed, differentiability w.r.t rigid transformations space, and so forth.
                Normally this should be left as ``None`` to let kaolin automatically select the optimal backend.
                Valid values: matrix_se3’, ‘matrix_6dof_rotation’ (see class description).
        """
        batch_dim = rotation.shape[0] if rotation.ndim > 2 else 1
        mat = torch.zeros((batch_dim, 4,4), dtype=rotation.dtype, device=rotation.device)
        mat[:, :3, :3] = rotation
        mat[:, :3, 3] = translation.squeeze(-1)
        mat[:, 3, 3] = 1
        backend = cls._make_backend(mat, dtype, device, requires_grad, backend_name)
        extrinsics = CameraExtrinsics(backend)
        extrinsics._shared_fields['user_requested_backend'] = backend_name is not None
        return extrinsics

    @staticmethod
    def _to_tensor_input(data: Union[np.ndarray, torch.Tensor], dtype: torch.dtype, device: Union[torch.device, str]):
        """ A convenience method allocate torch tensors from data of other numpy arrays / torch tensors """
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype, device=device)
        else:
            return torch.tensor(data, device=device, dtype=dtype)

    @classmethod
    def from_camera_pose(cls,
                         cam_pos: Union[np.ndarray, torch.Tensor],
                         cam_dir: Union[np.ndarray, torch.Tensor],
                         dtype: torch.dtype = default_dtype,
                         device: Union[torch.device, str] = None,
                         requires_grad: bool = False,
                         backend: str = None) -> CameraExtrinsics:
        r"""Constructs the extrinsics from the camera pose and orientation in world coordinates.

        Args:
            cam_pos (numpy.ndarray or torch.Tensor):
                the location of the camera center in world-coordinates,
                of shape :math:`(3,)`, :math:`(3, 1)`, :math:`(\text{num_cameras}, 3)` or
                :math:`(\text{num_cameras}, 3, 1)`
            cam_dir (numpy.ndarray or torch.Tensor):
                the camera's orientation with respect to the world,
                of shape :math:`(3, 3)` or :math:`(\text{num_cameras}, 3, 3)`
            dtype (optional, str):
                the dtype used for the tensors managed by the CameraExtrinsics.
                If dtype is None, :func:`torch.get_default_dtype()` will be used
            device (optional, str):
                the device on which the CameraExtrinsics object will manage its tensors.
                If device is None, the default torch device will be used
            requires_grad (bool):
                Sets the requires_grad field for the params tensor of the CameraExtrinsics
            backend (str):
                The backend used to manage the internal representation of the extrinsics, and how it is converted
                to a view matrix.
                Different representations are tuned to varied use cases:
                speed, differentiability w.r.t rigid transformations space, and so forth.
                Normally this should be left as ``None`` to let kaolin automatically select the optimal backend.
                Valid values: ``matrix_se3``, ``matrix_6dof_rotation`` (see class description).

        Returns:
            (CameraExtrinsics): the camera extrinsics
        """
        cam_pos = cls._to_tensor_input(cam_pos, device=device, dtype=dtype)
        cam_dir = cls._to_tensor_input(cam_dir, device=device, dtype=dtype)
        # The camera pose / orientation in world coordinates are converted to the world pose / axes in
        # camera coordinates:
        
        #   R_world = R_cam.T
        #   t_world = -R_world @ t_cam
        world_rotation = torch.transpose(cam_dir, -1, -2)
        if cam_pos.shape[-1] != 1:
            cam_pos = cam_pos.unsqueeze(-1)
        world_translation = -world_rotation @ cam_pos
        return cls._from_world_in_cam_coords(rotation=world_rotation, translation=world_translation,
                                             dtype=dtype, device=device, requires_grad=requires_grad,
                                             backend_name=backend)

    @classmethod
    def from_lookat(cls,
                    eye: Union[np.ndarray, torch.Tensor],
                    at: Union[np.ndarray, torch.Tensor],
                    up: Union[np.ndarray, torch.Tensor],
                    dtype: torch.dtype = default_dtype,
                    device: Union[torch.device, str] = None,
                    requires_grad: bool = False,
                    backend: str = None) -> CameraExtrinsics:
        r"""Constructs the extrinsic from camera position, camera up vector,
        and destination the camera is looking at.

        This constructor is compatible with glm's lookat function, which by default assumes a
        cartesian right-handed coordinate system (z axis positive direction points outwards from screen).

        Args:
            eye (numpy.ndarray or torch.Tensor):
                the location of the camera center in world-coordinates,
                of shape :math:`(3,)`, :math:`(3, 1)`, :math:`(\text{num_cameras}, 3)` or
                :math:`(\text{num_cameras}, 3, 1)`
            up (numpy.ndarray or torch.Tensor):
                the vector pointing up from the camera in world-coordinates,
                of shape :math:`(3,)`, :math:`(3, 1)`, :math:`(\text{num_cameras}, 3)`
                or :math:`(\text{num_cameras}, 3, 1)`
            at (numpy.ndarray or torch.Tensor) of [C]x3 or [C]x3x1,
                the direction the camera is looking at in world-coordinates,
                of shape :math:`(3,)`, :math:`(3, 1)`, :math:`(\text{num_cameras}, 3)`
                or :math:`(\text{num_cameras}, 3, 1)`
            dtype (optional, str):
                the dtype used for the tensors managed by the CameraExtrinsics.
                If dtype is None, the :func:`torch.get_default_dtype()` will be used
            device (optional, str):
                the device on which the CameraExtrinsics object will manage its tensors.
                If device is None, the default torch device will be used
            requires_grad (bool):
                Sets the requires_grad field for the params tensor of the CameraExtrinsics
            backend (str):
                The backend used to manage the internal representation of the extrinsics, and how it is converted
                to a view matrix.
                Different representations are tuned to varied use cases:
                speed, differentiability w.r.t rigid transformations space, and so forth.
                Normally this should be left as ``None`` to let kaolin automatically select the optimal backend.
                Valid values: ``matrix_se3``, ``matrix_6dof_rotation`` (see class description).

        Returns:
            (CameraExtrinsics): the camera extrinsics
        """
        eye = cls._to_tensor_input(eye, device=device, dtype=dtype)
        at = cls._to_tensor_input(at, device=device, dtype=dtype)
        up = cls._to_tensor_input(up, device=device, dtype=dtype)
        # Transform to tensors of (C, 3)
        eye = eye.squeeze(-1)
        at = at.squeeze(-1)
        up = up.squeeze(-1)
        if eye.ndim == 1:
            eye = eye.unsqueeze(0)
        if at.ndim == 1:
            at = at.unsqueeze(0)
        if up.ndim == 1:
            up = up.unsqueeze(0)

        # Follow OpenGL conventions: https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml
        backward = at - eye
        backward = torch.nn.functional.normalize(input=backward, dim=-1)
        right = torch.cross(backward, up, dim=-1)
        right = torch.nn.functional.normalize(input=right, dim=-1)
        up = torch.cross(right, backward, dim=-1)

        # For clarity: the extrinsic matrix maintained by this class is column major.
        # So far we constructed components that specify the camera position and orientation in world coordinates.
        # However, the view matrix is actually constructed using the world origin & axes in camera coordinates.
        # Hence we build components for constructing the inverse view matrix.
        # The view matrix can be obtained by inverting the matrix components we constructed:
        # (1) This amounts to transposing the rotation component,
        # (2) Negating the translation component in world coordinates, and multiplying it with the eye location).

        # Form a batched tensor where the two last dimensions are a matrix (form it transposed already):
        # [r1,   r2,   r3,     Right
        #  u1,   u2,   u3,     Up
        #  -b1,  -b2,  -b3]    Forward
        world_rotation = torch.stack((right, up, -backward), dim=1)

        # Translation component, according to cam location within the world
        world_translation = -world_rotation @ eye.unsqueeze(-1)
        return cls._from_world_in_cam_coords(rotation=world_rotation, translation=world_translation,
                                             dtype=dtype, device=device, requires_grad=requires_grad,
                                             backend_name=backend)

    @classmethod
    def from_view_matrix(cls,
                         view_matrix: Union[np.array, torch.Tensor],
                         dtype: torch.dtype = default_dtype,
                         device: Union[torch.device, str] = None,
                         requires_grad: bool = False,
                         backend: str = None) -> CameraExtrinsics:
        r"""Constructs the extrinsics from a given view matrix
        of shape :math:`(\text{num_cameras}, 4, 4)`.

        The matrix should be a column major view matrix, for converting vectors from world to camera coordinates
        (a.k.a: world2cam matrix):

        .. math::
        
            \begin{bmatrix}
                r1 & r2 & r3 & tx \\
                u1 & u2 & u3 & ty \\
                f1 & f2 & f3 & tz \\
                0 & 0 & 0 & 1
            \end{bmatrix}

        with:

            - **r**: Right - world x axis, in camera coordinates,
              also the camera right axis, in world coordinates
            - **u**: Up - world y axis, in camera coordinates,
              also the camera up axis, in world coordinates
            - **f**: Forward - world z axis, in camera coordinates,
              also the camera forward axis, in world coordinates
            - **t**: Position - the world origin in camera coordinates

        if you're using a different coordinate system, the axes may be permuted.

        .. seealso::

            :func:`change_coordinate_system()`

        Args:
            view_matrix (numpy.ndarray or torch.Tensor):
                view matrix, of shape :math:`(\text{num_cameras}, 4, 4)`
            dtype (optional, str):
                the dtype used for the tensors managed by the CameraExtrinsics.
                If dtype is None, the :func:`torch.get_default_dtype()` will be used
            device (optional, str):
                the device on which the CameraExtrinsics object will manage its tensors.
                If device is None, the default torch device will be used
            requires_grad (bool):
                Sets the requires_grad field for the params tensor of the CameraExtrinsics
            backend (str):
                The backend used to manage the internal representation of the extrinsics, and how it is converted
                to a view matrix.
                Different representations are tuned to varied use cases:
                speed, differentiability w.r.t rigid transformations space, and so forth.
                Normally this should be left as ``None`` to let kaolin automatically select the optimal backend.
                Valid values: ``matrix_se3``, ``matrix_6dof_rotation`` (see class description).

        Returns:
            (CameraExtrinsics): the camera extrinsics
        """
        view_matrix = cls._to_tensor_input(view_matrix, device=device, dtype=dtype)
        backend = cls._make_backend(view_matrix, dtype, device, requires_grad, backend)
        extrinsics = CameraExtrinsics(backend)
        extrinsics._shared_fields['user_requested_backend'] = backend is not None
        return extrinsics

    def change_coordinate_system(self, basis_change: Union[np.array, torch.Tensor]):
        r"""Applies a coordinate system change using the given 3x3 permutation & reflections matrix.

        For instance:

        (1) From a Y-up coordinate system (cartesian) to Z-up:

        .. math::

            \text{basis_change} = \begin{bmatrix}
                1 & 0 & 0 \\
                0 & 0 & -1 \\
                0 & 1 & 0
            \end{bmatrix}

        (2) From a right handed coordinate system (Z pointing outwards) to a left handed one (Z pointing inwards):

        .. math::
            \text{basis_change} = \begin{bmatrix}
                1 & 0 & 0 \\
                0 & 1 & 0 \\
                0 & 0 & -1
            \end{bmatrix}


        The basis_change is assumed to have a determinant of +1 or -1.

        .. seealso::

            :func:`blender_coords()` and :func:`opengl_coords()`

        Args:
            basis_change (numpy.ndarray or torch.Tensor):
                a composition of axes permutation and reflections, of shape :math:`(3, 3)`
        """
        # One prevalent form of performing coordinate change is swapping / negating the inverse view matrix rows.
        # That is - we want to alter the camera axes & position in WORLD coordinates.
        # Note it's enough however, to multiply the R component of the view matrix by the basis change matrix transpose
        # (recall we rotate about the world origin, which remains in place).
        #
        # Compare the inverse matrix before after basis change:
        #               Pre basis change:
        #     view_matrix =             inverse_view_matrix =           Rt is R transposed
        #       [ R | t ]                 [ Rt | -Rt @ t ]              @ denotes matrix column multiplication
        #       [ 0 | 1 ]                 [ 0  |   1     ]
        #
        #               Post basis change:
        #     view_matrix =             inverse_view_matrix =                    P is the basis change matrix
        #       [ R @ Pt | t ]                 [ P @ Rt | -(P @ Rt) @ t ]        Pt is the transposition of P
        #       [ 0      | 1 ]                 [ 0      |       1       ]
        #
        #                                =     [ P @ Rt | P @ (-Rt @ t) ]
        #                                      [ 0      |       1       ]
        basis_change = self._to_tensor_input(basis_change, device=self.device, dtype=self.dtype)

        # Cache basis change matrix to be able to revert later if desired
        self._base_change_matrix = self._base_change_matrix @ basis_change

        basis_change = basis_change.T
        basis_change = basis_change.repeat(len(self), 1, 1)
        self.R = self.R @ basis_change

    def reset_coordinate_system(self):
        """Resets the coordinate system back to the default one used by kaolin
            (right-handed cartesian: x pointing right, y pointing up, z pointing outwards)"""
        self.change_coordinate_system(self._base_change_matrix.T)

    @property
    def R(self) -> torch.Tensor:
        r"""A tensor whose columns represent the directions of world-axes in camera coordinates,
        of shape :math:`(\text{num_cameras}, 3, 3)`.

        This is the **R** submatrix of the extrinstic matrix:

        .. math::

            \begin{bmatrix}
                R & t \\
                0 & 1
            \end{bmatrix}

        defined as:

        .. math::

            R = \begin{bmatrix}
                r1 & r2 & r3 \\
                u1 & u2 & u3 \\
                f1 & f2 & f3
                \end{bmatrix}

        with:

            - **r**: Right - world x axis, in camera coordinates,
              also the camera right axis, in world coordinates
            - **u**: Up - world y axis, in camera coordinates,
              also the camera up axis, in world coordinates
            - **f**: Forward - world z axis, in camera coordinates,
              also the camera forward axis, in world coordinates

        .. seealso:

            :attr:`cam_forward`, :attr:`cam_up`, :attr:`cam_right` for camera axes
            in world coordinates.
        """
        return self.view_matrix()[:, :3, :3]

    @R.setter
    def R(self, val: torch.Tensor):
        """Sets a subset of the matrix whose columns represent
        the directions of world-axes in camera coordinates.
        """
        mat = self.view_matrix()
        mat[:, :3, :3] = val
        self._backend.update(mat)

    @property
    def t(self) -> torch.Tensor:
        r"""The position of world origin in camera coordinates,
        a torch.Tensor of shape :math:`(\text{num_cameras}, 3, 1)`

        This is the **t** vector of the extrinsic matrix:

        .. math::

            \begin{bmatrix}
                R & t \\
                0 & 1
            \end{bmatrix}

        .. seealso::

            :attr:`cam_pos` for the camera position in world coordinates.
        """
        return self.view_matrix()[:, :3, -1:]

    @t.setter
    def t(self, val: torch.Tensor):
        """Sets the position of world origin in camera coordinates."""

        mat = self.view_matrix()
        if val.ndim == 1:
            val = val.unsqueeze(-1)
        mat[:, :3, -1:] = val
        self._backend.update(mat)

    def __len__(self) -> int:
        """Returns the number of cameras batched in this instance."""
        return len(self._backend)

    def transform(self, vectors: torch.Tensor) -> torch.Tensor:
        r"""Apply rigid transformation of the camera extrinsics such that
        objects in world coordinates are transformed to camera space coordinates.

        The camera coordinates are cast to the precision of the vectors argument.

        Args:
            vectors (torch.Tensor):
                the vectors, of shape :math:`(\text{num_vectors}, 3)`
                or :math:`(\text{num_cameras}, \text{num_vectors}, 3)`

        Returns:
            (torch.Tensor): the transformed vector, of same shape than ``vectors``
        """
        assert self.dtype == vectors.dtype,\
            f"CameraExtrinsics of dtype {self.dtype} cannot transform vectors of dtype {vectors.dtype}"
        assert self.device == vectors.device, \
            f"CameraExtrinsics of device {self.device} cannot transform vectors of device {vectors.device}"
        num_cameras = len(self)          # C - number of cameras
        batch_size = vectors.shape[-2]   # B - number of vectors
        v = vectors.expand(num_cameras, batch_size, 3)[..., None]   # Expand as (C, B, 3, 1)
        R = self.R[:, None].expand(num_cameras, batch_size, 3, 3)   # Expand as (C, B, 3, 3)
        t = self.t[:, None].expand(num_cameras, batch_size, 3, 1)   # Expand as (C, B, 3, 1)
        transformed_v = R @ v + t
        return transformed_v.squeeze(-1)  # Return shape:  (C, B, 3)

    def inv_transform_rays(self, ray_orig: torch.Tensor, ray_dir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Transforms rays from camera space to world space (hence: "inverse transform").

        Apply rigid transformation of the camera extrinsics.
        The camera coordinates are cast to the precision of the vectors argument.

        Args:
            ray_orig (torch.Tensor):
                the origins of rays, of shape :math:`(\text{num_rays}, 3)` or
                :math:`(\text{num_cameras}, \text{num_rays}, 3)`
            ray_dir (torch.Tensor):
                the directions of rays, of shape :math:`(\text{num_rays}, 3)` or
                :math:`(\text{num_cameras}, \text{num_rays}, 3)`

        Returns:
            (torch.Tensor, torch.Tensor):
                the transformed ray origins and directions, of same shape than inputs
        """
        assert self.dtype == ray_orig.dtype == ray_dir.dtype,\
            f"CameraExtrinsics of dtype {self.dtype} cannot transform " \
            f"ray_orig/dir of dtype {ray_orig.dtype}, {ray_dir.dtype}"
        assert self.device == ray_orig.device == ray_dir.device, \
            f"CameraExtrinsics of device {self.device} cannot transform " \
            f"ray_orig/dir of device {ray_orig.device}, {ray_dir.device}"
        num_cameras = len(self)           # C - number of cameras
        batch_size = ray_dir.shape[-2]    # B - number of vectors
        d = ray_dir.expand(num_cameras, batch_size, 3)[..., None]   # Expand as (C, B, 3, 1)
        o = ray_orig.expand(num_cameras, batch_size, 3)[..., None]  # Expand as (C, B, 3, 1)
        R = self.R[:, None].expand(num_cameras, batch_size, 3, 3)   # Expand as (C, B, 3, 3)
        t = self.t[:, None].expand(num_cameras, batch_size, 3, 1)   # Expand as (C, B, 3, 1)
        R_T = R.transpose(2, 3)     # Transforms orientation from camera to world
        transformed_dir = R_T @ d   # Inverse rotation is transposition: R^(-1) = R^T
        transformed_orig = R_T @ (o - t)
        return transformed_orig.squeeze(-1), transformed_dir.squeeze(-1)  # Return shape:  (C, B, 3)

    def view_matrix(self) -> torch.Tensor:
        r"""Returns a column major view matrix for converting vectors from world to camera coordinates
        (a.k.a: world2cam matrix):

        .. math::
        
            \begin{bmatrix}
                r1 & r2 & r3 & tx \\
                u1 & u2 & u3 & ty \\
                f1 & f2 & f3 & tz \\
                0 & 0 & 0 & 1
            \end{bmatrix}

        with:

            - **r**: Right - world x axis, in camera coordinates,
              also the camera right axis, in world coordinates
            - **u**: Up - world y axis, in camera coordinates,
              also the camera up axis, in world coordinates
            - **f**: Forward - world z axis, in camera coordinates,
              also the camera forward axis, in world coordinates
            - **t**: Position - the world origin in camera coordinates

        if you're using a different coordinate system, the axes may be permuted.

        .. seealso::

            :func:`change_coordinate_system()`


        The matrix returned by this class supports pytorch differential operations

        .. note::

            practitioners are advised to choose a representation backend which
            supports differentiation of rigid transformations

        .. note::

            Changes modifying the returned tensor will also update the extrinsics parameters.

        Returns:
            (torch.Tensor):
                the view matrix, of shape :math:`(\text{num_cameras}, 4, 4)` (homogeneous coordinates)
        """
        return self._backend.convert_to_mat()

    def inv_view_matrix(self) -> torch.Tensor:
        r"""Returns the inverse of the view matrix used to convert vectors from camera to world coordinates
        (a.k.a: cam2world matrix). This matrix is column major:

        .. math::

            \begin{bmatrix}
                r1 & u1 & f1 & px \\
                r2 & u2 & f2 & py \\
                r3 & u3 & f3 & pz \\
                0 & 0 & 0 & 1
            \end{bmatrix}

        with:

            - **r**: Right - world x axis, in camera coordinates,
              also the camera right axis, in world coordinates
            - **u**: Up - world y axis, in camera coordinates,
              also the camera up axis, in world coordinates
            - **f**: Forward - world z axis, in camera coordinates,
              also the camera forward axis, in world coordinates
            - **t**: Position - the world origin in camera coordinates

        if you're using a different coordinate system, the axes may be permuted.

        .. seealso::

            :func:`change_coordinate_system()`

        Returns:
            (torch.Tensor):
                the inverse view matrix, of shape :math:`(\text{num_cameras}, 4, 4)`
        """

        inv_view = torch.eye(4, device=self.device, dtype=self.dtype).repeat(len(self), 1, 1)
        R_inv = self.R.transpose(1, 2)  # R^-1 = R^T
        inv_view[:, :3, :3] = R_inv
        inv_view[:, :3, -1:] = -R_inv @ self.t  # cam_center = -R^T @ t
        return inv_view

    def update(self, mat: torch.Tensor):
        r"""Updates extrinsics parameters to match the given view matrix.
        
        Args:
            mat (torch.Tensor):
                the new view matrix, of shape :math:`(\text{num_cameras}, 4, 4)`
        """
        self._backend.update(mat)

    def translate(self, t: torch.Tensor):
        r"""Translates the camera in world coordinates.
        The camera orientation axes will not change.

        Args:
            t (torch.Tensor):
                Amount of translation in world space coordinates,
                of shape :math:`(3,)` or :math:`(3, 1)` broadcasting over all the cameras,
                or :math:`(\text{num_cameras}, 3, 1)` for applying unique translation per camera.
        """
        assert self.dtype == t.dtype,\
            f"CameraExtrinsics of dtype {self.dtype} cannot translate with tensor of dtype {t.dtype}"
        assert self.device == t.device, \
            f"CameraExtrinsics of device {self.device} cannot translate with tensor of device {t.device}"
        if t.shape[-1] != 1:
            t = t[..., None]    # Add row dimension,
        self.t -= self.R @ t    # batch dim is broadcasted if needed

    def rotate(self,
               yaw: Union[float, torch.Tensor]=None,
               pitch: Union[float, torch.Tensor]=None,
               roll: Union[float, torch.Tensor]=None):
        r"""Executes an inplace rotation of the camera using the given yaw, pitch, and roll amounts.
        
        Input can be float / tensor float units will apply the same rotation on all cameras,
        where torch.Tensors allow for applying a per-camera rotation.
        Rotation is applied in camera space.

        Args:
            yaw (torch.Tensor or float):
                Amount of rotation in radians around normal direction of right-up plane
            pitch (torch.Tensor or float):
                Amount of rotation in radians around normal direction of right-forward plane
            roll (torch.Tensor or float):
                Amount of rotation in radians around normal direction of up-forward plane
        """
        if yaw is not None and not isinstance(yaw, torch.Tensor):
            yaw = torch.tensor([yaw], device=self.device, dtype=self.dtype)
        if pitch is not None and not isinstance(pitch, torch.Tensor):
            pitch = torch.tensor([pitch], device=self.device, dtype=self.dtype)
        if roll is not None and not isinstance(roll, torch.Tensor):
            roll = torch.tensor([roll], device=self.device, dtype=self.dtype)

        #   Yaw-Pitch-Roll (a.k.a Tait Bryan angles) affect the camera angles as follows:
        #    camera up (yaw)
        #    ^      camera forward (roll)
        #    |    ^
        #    |  /
        #    |/
        #    ----------->  camera right (pitch)
        rotation_mat = torch.eye(4, device=self.device, dtype=self.dtype)
        if yaw is not None:  # Rotate around "camera up" axis
            # Batch compatible version of
            # torch.tensor([
            #     [torch.cos(yaw),  0,  -torch.sin(yaw), 0],
            #     [0,               1,  0,               0],
            #     [torch.sin(yaw),  0,  torch.cos(yaw),  0],
            #     [0,               0,  0,               1]
            # ])
            rot_yaw = torch.eye(4, device=self.device, dtype=self.dtype).repeat(len(self), 1, 1)
            rot_yaw[:, 0, 0] = torch.cos(yaw)
            rot_yaw[:, 0, 2] = -torch.sin(yaw)
            rot_yaw[:, 2, 0] = torch.sin(yaw)
            rot_yaw[:, 2, 2] = torch.cos(yaw)
            rotation_mat = rot_yaw @ rotation_mat
        if pitch is not None:   # Rotate around "camera right" axis
            # Batch compatible version of
            # torch.tensor([
            #     [1, 0,                 0,                0],
            #     [0, torch.cos(pitch),  torch.sin(pitch), 0],
            #     [0, -torch.sin(pitch), torch.cos(pitch), 0],
            #     [0, 0,                 0,                1]
            # ])
            rot_pitch = torch.eye(4, device=self.device, dtype=self.dtype).repeat(len(self), 1, 1)
            rot_pitch[:, 1, 1] = torch.cos(pitch)
            rot_pitch[:, 1, 2] = torch.sin(pitch)
            rot_pitch[:, 2, 1] = -torch.sin(pitch)
            rot_pitch[:, 2, 2] = torch.cos(pitch)
            rotation_mat = rot_pitch @ rotation_mat
        if roll is not None:  # Rotate around "camera forward" axis
            # Batch compatible version of
            # torch.tensor([
            #     [torch.cos(roll),  -torch.sin(roll),   0, 0],
            #     [torch.sin(roll), torch.cos(roll),     0, 0],
            #     [0,                0,                  1, 0],
            #     [0,                0,                  0, 1]
            # ], device=self.device, dtype=self.dtype)
            rot_roll = torch.eye(4, device=self.device, dtype=self.dtype).repeat(len(self), 1, 1)
            rot_roll[:, 0, 0] = torch.cos(roll)
            rot_roll[:, 0, 1] = -torch.sin(roll)
            rot_roll[:, 1, 0] = torch.sin(roll)
            rot_roll[:, 1, 1] = torch.cos(roll)
            rotation_mat = rot_roll @ rotation_mat
        mat = rotation_mat.unsqueeze(0) @ self.view_matrix()
        self._backend.update(mat)

    def move_right(self, amount):
        """Translates the camera along the camera right axis.

        Args:
            amount (torch.Tensor or float):
                Amount of translation, measured in world coordinates
        """
        self.t -= self._world_x() * amount

    def move_up(self, amount):
        """Translates the camera along the camera up axis.

        Args:
            amount (torch.Tensor or float):
                Amount of translation, measured in world coordinates.
        """
        self.t -= self._world_y() * amount

    def move_forward(self, amount):
        """Translates the camera along the camera forward axis.

        Args:
            amount (torch.Tensor or float):
                Amount of translation, measured in world coordinates.
        """
        self.t -= self._world_z() * amount

    def _world_x(self) -> torch.Tensor:
        """Returns:
            (torch.Tensor): the world x axis in world coordinates.
        """
        right_col = torch.zeros_like(self.t)
        right_col[:, 0] = 1.0
        return right_col

    def _world_y(self) -> torch.Tensor:
        """Returns:
            (torch.Tensor): the world y axis in world coordinates.
        """
        up_col = torch.zeros_like(self.t)
        up_col[:, 1] = 1.0
        return up_col

    def _world_z(self) -> torch.Tensor:
        """Returns:
            (torch.Tensor): the world z axis in world coordinates.
        """
        forward_col = torch.zeros_like(self.t)
        forward_col[:, 2] = 1.0
        return forward_col

    def cam_pos(self) -> torch.Tensor:
        """Returns:
            (torch.Tensor): the camera position, in world coordinates
        """
        R_inv = self.R.transpose(1, 2)  # R^-1 = R^T
        return -R_inv @ self.t          # cam_pos = -R^T @ t

    def cam_right(self) -> torch.Tensor:
        """Returns:
            (torch.Tensor): the camera right axis, in world coordinates
        """
        return self.R.transpose(2, 1) @ self._world_x()

    def cam_up(self) -> torch.Tensor:
        """Returns:
            (torch.Tensor): the camera up axis, in world coordinates
        """
        return self.R.transpose(2, 1) @ self._world_y()

    def cam_forward(self) -> torch.Tensor:
        r""" Returns the camera forward axis -

        See: https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function/framing-lookat-function.html

        Returns:
            (torch.Tensor): the camera forward axis, in world coordinates."""
        return self.R.transpose(2, 1) @ self._world_z()

    def parameters(self) -> torch.Tensor:
        """Returns:
            (torch.Tensor):

                the extrinsics parameters buffer.
                This is essentially the underlying representation of the extrinsics,
                and is backend dependant.
        """
        return self._backend.params

    @property
    def device(self) -> torch.device:
        """the torch device of parameters tensor"""
        return self._backend.device

    @property
    def dtype(self) -> torch.dtype:
        """the torch dtype of parameters tensor"""
        return self._backend.dtype

    def to(self, *args, **kwargs) -> CameraExtrinsics:
        """An instance of this object with the parameters tensor on the given device.
        If the specified device is the same as this object, this object will be returned.
        Otherwise a new object with a copy of the parameters tensor on the requested device will be created.

        .. seealso::

            :func:`torch.Tensor.to`
        """
        converted_backend = self._backend.to(*args, **kwargs)
        if self._backend == converted_backend:
            return self
        else:
            extrinsics = CameraExtrinsics(converted_backend)
            extrinsics._base_change_matrix = self._base_change_matrix.clone()
            return extrinsics

    def cpu(self) -> CameraExtrinsics:
        return self.to('cpu')

    def cuda(self) -> CameraExtrinsics:
        return self.to('cuda')

    def half(self) -> CameraExtrinsics:
        return self.to(torch.float16)

    def float(self) -> CameraExtrinsics:
        return self.to(torch.float32)

    def double(self) -> CameraExtrinsics:
        return self.to(torch.float64)

    @property
    def requires_grad(self) -> bool:
        """True if the current extrinsics object allows gradient flow.

        .. note::

            All extrinsics backends allow gradient flow, but some are not guaranteed to maintain a rigid
            transformation view matrix.
        """
        return self._backend.requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        """Toggle gradient flow for the extrinsics.

        .. note::

            All extrinsics backends allow gradient flow, but some are not guaranteed to maintain a rigid
            transformation view matrix. By default, kaolin will switch the representation backend to one that
            supports differentiable rigid transformations. This behaviour is disabled if users explicitly choose
            the representation backend through :func:`switch_backend`.

        """
        if self.requires_grad != val and not self._shared_fields['user_requested_backend']:
            # If the user haven't requested a specific backend, automatically set the
            # one which best agrees with the new differentiability state.
            # For requires_grad = True, if the params tensor is a leaf node set a differentiable representation backend
            if val and self.parameters().is_leaf:
                backend = 'matrix_6dof_rotation'
            else:
                backend = 'matrix_se3'
            self._internal_switch_backend(backend)
        self._backend.requires_grad = val

    @property
    def backend_name(self) -> str:
        """the unique name used to register the currently used representation backend.

        Values available by default:

            - **"matrix_se3"**: A flattened view matrix representation, containing the full information of
              special eucilidean transformations (translations and rotations).
              This representation is quickly converted to a view matrix, but differentiable ops may cause
              the view matrix to learn an incorrect, non-orthogonal transformation.
            - **"matrix_6dof_rotation"**: A compact representation with 6 degrees of freedom,
              ensuring the view matrix remains orthogonal under optimizations.
              The conversion to matrix requires a single Gram-Schmidt step.
        """
        value_idx = list(_REGISTERED_BACKENDS.values()).index(type(self._backend))
        backend_name = list(_REGISTERED_BACKENDS.keys())[value_idx]
        return backend_name

    @property
    def _base_change_matrix(self):
        """the transformation matrix (permutation + reflections) used to change the coordinates system
        of this camera from the default cartesian one to another.

        This matrix is manipulated by: :func:`change_coordinate_system()`,

        seealso::
            :func:`reset_coordinate_system()`
        """
        return self._shared_fields.get('base_change_matrix')

    @_base_change_matrix.setter
    def _base_change_matrix(self, value):
        """Sets the transformation matrix (permutation + reflections) used to change the coordinates system
        of this camera from the default cartesian one to another.
        seealso::
            :func:`reset_coordinate_system()`
        """
        self._shared_fields['base_change_matrix'] = value

    @property
    def basis_change_matrix(self):
        """The transformation matrix (permutation + reflections) used to change the coordinates system
        of this camera from the default cartesian one to another.

        This matrix is manipulated by: :func:`change_coordinate_system()`,
        :func:`reset_coordinate_system()`
        """
        return self._base_change_matrix

    def gradient_mask(self, *args: Union[str, ExtrinsicsParamsDefEnum]) -> torch.Tensor:
        r"""Creates a gradient mask, which allows to backpropagate only through params designated as trainable.

        This function does not consider the ``requires_grad`` field when creating this mask.

        .. note::
            The 3 camera axes are always masked as trainable together.
            This design choice ensures that these axes, as well as the view matrix, remain orthogonal.

        Args:
            *args: A vararg list of the extrinsics params that should allow gradient flow.
                   This function also supports conversion of params from their string names.
                   (i.e: 't' will convert to ``ExtrinsicsParamsDefEnum.t``)

        Example:
            >>> # equivalent to:   mask = extrinsics.gradient_mask(ExtrinsicsParamsDefEnum.t)
            >>> mask = extrinsics.gradient_mask('t')
            >>> extrinsics.params.register_hook(lambda grad: grad * mask.float())
            >>> # extrinsics will now allow gradient flow only for the camera location

        Returns:
            (torch.BoolTensor): the gradient mask, of same shape than ``self.parameters()``
        """
        try:
            # Convert str args to ExtrinsicsParamsDefEnum subclass values
            args = [ExtrinsicsParamsDefEnum[a] if isinstance(a, str) else a for a in args]
        except KeyError as e:
            raise ValueError(f'Camera\'s set_trainable_params() received an unsupported arg: {e}')

        mask = torch.zeros_like(self.parameters()).bool()
        for param in args:
            # The indices of each extrinsic param are backend dependant
            indices = self._backend.param_idx(param)
            mask[:, indices] = 1.0
        return mask

    def __getitem__(self, item) -> CameraExtrinsics:
        r"""Returns a view on a specific cameras from the batch of cameras managed by this object.

        Returns:
           (CameraExtrinsics):

               A subset of camera's extrinsics from this batched object,
               of shape :math:`(\text{size_slice}, 4, 4)`"""
        return CameraExtrinsics(self._backend[item], self._shared_fields)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in _HANDLED_TORCH_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, CameraExtrinsics))
            for t in types
        ):
            return NotImplemented
        return _HANDLED_TORCH_FUNCTIONS[func](*args, **kwargs)

    @classmethod
    def available_backends(cls) -> Iterable[str]:
        """Returns:
            (iterable of str):

                list of available representation backends,
                to be used with :func:`switch_backend`
        """
        return _REGISTERED_BACKENDS.keys()

    @classmethod
    def cat(cls, cameras: Sequence[CameraExtrinsics]):  
        """Concatenate multiple CameraExtrinsics's.

        Assumes all cameras use the same coordinate system.
        (kaolin will not alert if not, the coordinate system will be selected as the first camera)

        Args:
            cameras (Sequence of CameraExtrinsics): the cameras extrinsics to concatenate.

        Returns:
            (CameraExtrinsics): The concatenated cameras extrinsics as a single CameraExtrinsics
        """
        if len(cameras) == 0:
            return None
        view_mats = [c.view_matrix() for c in cameras]
        batched_cams = torch.cat(view_mats, dim=0)
        extrinsics = CameraExtrinsics.from_view_matrix(batched_cams,
                                                       device=cameras[0].device,
                                                       dtype=cameras[0].dtype,
                                                       requires_grad=cameras[0].requires_grad,
                                                       backend=cameras[0].backend_name)
        extrinsics._base_change_matrix = cameras[0]._base_change_matrix
        return extrinsics

    def named_params(self) -> List[Dict[str, float]]:
        """Get a descriptive list of named parameters per camera.

        Returns:
            (list of dict): The named parameters.
        """
        named_params_per_camera = []
        params = self.parameters()
        R_idx = self._backend.param_idx(ExtrinsicsParamsDefEnum.R)
        t_idx = self._backend.param_idx(ExtrinsicsParamsDefEnum.t)

        # Collect the parameters of each of the cameras
        for camera_idx in range(len(self)):
            cam_params = dict(
                R=params[camera_idx, R_idx],
                t=params[camera_idx, t_idx]
            )
            named_params_per_camera.append(cam_params)
        return named_params_per_camera

    def __repr__(self) -> str:
        title = f"CameraExtrinsics of {len(self)} cameras, device: {self.device}, dtype: {self.dtype}, " \
                f"backend: {type(self._backend).__name__}.\n"
        coords = f"Coordinates basis: \n{self.basis_change_matrix}.\n"
        params_txt = f"Extrinsic params: {self.parameters()}\n"
        return ''.join([title] + [coords] + [params_txt])

    def __str__(self) -> str:
        return f"CameraExtrinsics of {len(self)} cameras, of coordinate system: \n{self.basis_change_matrix}. \n" + \
               '\n'.join([
                   f"Camera #{idx} View Matrix: \n{self.view_matrix()},\n" \
                   f"Camera #{idx} Inverse View Matrix: \n{self.inv_view_matrix()}\n"
                   for idx in range(len(self))
               ])

@implements(torch.allclose)
def allclose(input: CameraExtrinsics, other: CameraExtrinsics, rtol: _float = 1e-05, atol: _float = 1e-08,
            equal_nan: _bool = False) -> _bool:
    """:func:`torch.allclose` compatibility implementation for CameraExtrinsics.

    Args:
        input (Camera): first camera to compare
        other (Camera): second camera to compare
        atol (float, optional): absolute tolerance. Default: 1e-08
        rtol (float, optional): relative tolerance. Default: 1e-05
        equal_nan (bool, optional): if ``True``, then two ``NaN`` s will be considered equal.
                                    Default: ``False``

    Returns:
        (bool): Result of the comparison
    """
    return input.backend_name == other.backend_name and \
           torch.allclose(input.parameters(), other.parameters(), rtol=rtol, atol=atol, equal_nan=equal_nan)
