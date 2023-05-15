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
from abc import ABC, abstractmethod
from typing import Type, Dict, List, Union, Sequence
from enum import IntEnum
import functools
import copy
import torch
from torch.types import _float, _bool

__all__ = [
    "CameraFOV",
    "CameraIntrinsics",
    "up_to_homogeneous",
    "down_from_homogeneous"
]

_HANDLED_TORCH_FUNCTIONS = dict()
default_dtype = torch.get_default_dtype()


def implements(torch_function):
    """Registers a torch function override for CameraIntrinsics"""
    @functools.wraps(torch_function)
    def decorator(func):
        _HANDLED_TORCH_FUNCTIONS[torch_function] = func
        return func
    return decorator


def up_to_homogeneous(vectors: torch.Tensor):
    """Up-projects vectors to homogeneous coordinates of four dimensions.
    If the vectors are already in homogeneous coordinates, this function return the inputs.

    Args:
        vectors (torch.Tensor):
            the inputs vectors to project, of shape :math:`(..., 3)`

    Returns:
        (torch.Tensor): The projected vectors, of same shape than inputs but last dim to be 4
    """
    if vectors.shape[-1] == 4:
        return vectors
    return torch.cat([vectors, torch.ones_like(vectors[..., 0:1])], dim=-1)


def down_from_homogeneous(homogeneous_vectors: torch.Tensor):
    """(1) Performs perspective division by dividing each vector by its w coordinate.
    (2) Down-projects vectors from 4D homogeneous space to 3D space.

    Args:
        homogenenous_vectors: the inputs vectors, of shape :math:`(..., 4)`

    Returns:
        (torch.Tensor): the 3D vectors, of same shape than inputs but last dim to be 3    
    """
    return homogeneous_vectors[..., :-1] / homogeneous_vectors[..., -1:]

class CameraFOV(IntEnum):
    """Camera's field-of-view can be defined by either of the directions"""
    HORIZONTAL = 0
    VERTICAL = 1
    DIAGONAL = 2    # Used by wide fov lens (i.e. fisheye)


# Alias for parameters definition enum, subclasses should define this to specify order of intrinsics parameters stored
# See usage through: ClassIntrinsics.param_types()
IntrinsicsParamsDefEnum = IntEnum


class CameraIntrinsics(ABC):
    r"""Holds the intrinsics parameters of a camera: how it should project from camera space to
    normalized screen / clip space.

    The instrinsics are determined by the camera type, meaning parameters may differ according to the lens structure.
    Typical computer graphics systems commonly assume the intrinsics of a pinhole camera (see: :class:`PinholeIntrinsics` class).

    One implication is that some camera types do not use a linear projection (i.e: Fisheye lens).
    There are therefore numerous ways to use CameraIntrinsics subclasses:

        1. Access intrinsics parameters directly.
        This may typically benefit use cases such as ray generators.

        2. The :func:`transform()` method is supported by all CameraIntrinsics subclasses,
        both linear and non-linear transformations, to project vectors from camera space to normalized screen space.
        This method is implemented using differential pytorch operations.

        3. Certain CameraIntrinsics subclasses which perform linear projections, may expose the transformation matrix
        via dedicated methods.
        For example, :class:`PinholeIntrinsics` exposes a :func:`projection_matrix()` method.
        This may typically be useful for rasterization based rendering pipelines (i.e: OpenGL vertex shaders).

    This class is batched and may hold information from multiple cameras.

    Parameters are stored as a single tensor of shape :math:`(\text{num_cameras}, K)` where K is the number of
    intrinsic parameters.
    """

    def __init__(self, width: int, height: int, params: torch.Tensor, near: float, far: float):
        # Make batched
        if params.ndim == 1:
            params = params.unsqueeze(0)

        self.params: torch.Tensor = params    # Buffer for camera intrinsic params, shape (C, K)

        # _shared_fields ensures that views created on this instance will mirror any changes back
        # These fields can be accessed as simple properties
        self._shared_fields = dict(
            width=width,               # Screen resolution (x), int
            height=height,             # Screen resolution (y), int
            near=float(near),          # Near clipping plane, float
            far=float(far),            # Far clipping plane, float
            ndc_min=-1.0,              # Min value of NDC space
            ndc_max=1.0                # Max value of NDC space
        )

    def aspect_ratio(self) -> float:
        """Returns the aspect ratio of the cameras held by this object."""
        return self.width / self.height

    def projection_matrix(self):
        raise NotImplementedError('This projection of this camera type is non-linear in homogeneous coordinates '
                                  'and therefore does not support a projection matrix. Use self.transform() instead.')

    def viewport_matrix(self, vl=0, vr=None, vb=0, vt=None, min_depth=0.0, max_depth=1.0) -> torch.Tensor:
        r"""Constructs a viewport matrix which transforms coordinates from NDC space to pixel space.
        This is the general matrix form of glViewport, familiar from OpenGL.

        NDC coordinates are expected to be in:
        * [-1, 1] for the (x,y) coordinates.
        * [ndc_min, ndc_max] for the (z) coordinate.

        Pixel coordinates are in:
        * [vl, vr] for the (x) coordinate.
        * [vb, vt] for the (y) coordinate.
        * [0, 1] for the (z) coordinate (yielding normalized depth).

        When used in conjunction with a :func:`projection_matrix()`, a transformation from camera view space to
        window space can be obtained.

        Note that for the purpose of rendering with OpenGL shaders, this matrix is not required, as viewport
        transformation is already applied by the hardware.

        By default, this matrix assumes the NDC screen spaces have the y axis pointing up.
        Under this assumption, and a [-1, 1] NDC space,
        the default values of this method are compatible with OpenGL glViewport.

        .. seealso::

            glViewport() at https://registry.khronos.org/OpenGL-Refpages/gl4/html/glViewport.xhtml
            and https://en.wikibooks.org/wiki/GLSL_Programming/Vertex_Transformations#Viewport_Transformation

            projection_matrix() which converts coordinates from camera view space to NDC space.

        .. note::

            1. This matrix changes form depending on the NDC space used.
            2. Returned values are floating points, rather than integers
               (thus this method is compatible with antialising ops).

        Args:
            vl (int): Viewport left (pixel coordinates x) - where the viewport starts. Default is 0.
            vr (int): Viewport right (pixel coordinates x) - where the viewport ends. Default is camera width.
            vb (int): Viewport bottom (pixel coordinates y) - where the viewport starts. Default is 0.
            vt (int): Viewport top (pixel coordinates y) - where the viewport ends. Default is camera height.
            min_depth (float): Minimum of output depth range. Default is 0.0.
            max (float): Maximum of output depth range. Default is 1.0.

        Returns:
            (torch.Tensor): the viewport matrix, of shape :math:`(1, 4, 4)`.
        """
        if vr is None:
            vr = self.width
        if vt is None:
            vt = self.height
        vl = float(vl)
        vr = float(vr)
        vb = float(vb)
        vt = float(vt)

        # From NDC space
        ndc_min_x = -1.0
        ndc_min_y = -1.0
        ndc_min_z = self.ndc_min
        ndc_max_x = 1.0
        ndc_max_y = 1.0
        ndc_max_z = self.ndc_max
        ndc_width = ndc_max_x - ndc_min_x               # All ndc spaces assume x clip coordinates in [-1, 1]
        ndc_height = ndc_max_y - ndc_min_y              # All ndc spaces assume y clip coordinates in [-1, 1]
        ndc_depth = ndc_max_z - ndc_min_z               # NDC depth range, this is NDC space dependent

        # To screen space
        vw = vr - vl                                    # Viewport width
        vh = vt - vb                                    # Viewport height
        out_depth_range = max_depth - min_depth         # By default, normalized depth is assumed [0, 1]

        # Recall that for OpenGL NDC space and full screen viewport, the following matrix is given,
        # where vw, vh stand for screen width and height:
        #              [vw/2,  0.0,  0.0,  vw/2]  @  [ x ]   = ..   perspective   =  [(x/w + 1) * (vw/2)]
        #              [0.0,   vh/2, 0.0,  vh/2]     [ y ]          division         [(y/w + 1) * (vh/2)]
        #              [0.0,   0.0,  1/2,  1/2]      [ z ]          ------------>    [(z/w + 1) / 2]
        #              [0.0,   0.0,  0.0,  1.0]      [ w ]          (/w)             [  1.0  ]

        # The matrix is non differentiable, as viewport coordinates are a fixed standard set by the graphics api
        ndc_mat = self.params.new_tensor([
            [vw / ndc_width, 0.0,             0.0,                          -(ndc_min_x / ndc_width) * vw + vl],
            [0.0,            vh / ndc_height, 0.0,                          -(ndc_min_y / ndc_height) * vh + vb],
            [0.0,            0.0,             out_depth_range / ndc_depth,  -(ndc_min_z / ndc_depth) * out_depth_range + min_depth],
            [0.0,            0.0,             0.0,                          1.0]
        ])

        # Add batch dim, to allow broadcasting
        return ndc_mat.unsqueeze(0)

    @abstractmethod
    def transform(self, vectors: torch.Tensor) -> torch.Tensor:
        r"""Projects the vectors from view space / camera space to NDC (normalized device coordinates) space.
        The NDC space used by kaolin is a left-handed coordinate system which uses OpenGL conventions::

             Y      Z
             ^    /
             |  /
             |---------> X

        The coordinates returned by this class are not concerned with clipping, and therefore the range
        of values returned by this transformation is not numerically bounded between :math:`[-1, 1]`.

        To support a wide range of lens, this function is compatible with both linaer or non-linear transformations
        (which are not representable by matrices).
        CameraIntrinsics subclasses should always implement this method using pytorch differential operations.

        Args:
            vectors (torch.Tensor):
                the vectors to be transformed,
                can homogeneous of shape :math:`(\text{num_vectors}, 4)`
                or :math:`(\text{num_cameras}, \text{num_vectors}, 4)`
                or non-homogeneous of shape :math:`(\text{num_vectors}, 3)`
                or :math:`(\text{num_cameras}, \text{num_vectors}, 3)`

        Returns:
            (torch.Tensor): the transformed vectors, of same shape than ``vectors`` but last dim 3
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def param_types(cls) -> Type[IntrinsicsParamsDefEnum]:
        """
        Returns:
            (IntrinsicsParamsDefEnum):

                an enum describing each of the intrinsic parameters managed by the subclass.
                This enum also defines the order in which values are kept within the params buffer.
        """
        raise NotImplementedError

    def param_count(self) -> int:
        """
        Returns:
            (int): number of intrinsic parameters managed per camera
        """
        return len(self.param_types())

    def named_params(self) -> List[Dict[str, float]]:
        """Get a descriptive list of named parameters per camera.

        Returns:
            (list of dict): The named parameters.
        """
        param_names = self.param_types()._member_names_
        named_params_per_camera = []

        # Collect the parameters of each of the cameras
        for camera_idx in range(len(self)):
            cam_params = {p_name: self.params[camera_idx, p_idx].item() for p_idx, p_name in enumerate(param_names)}
            named_params_per_camera.append(cam_params)
        return named_params_per_camera

    @classmethod
    def _allocate_params(cls, *args,
                         num_cameras: int = 1,
                         device: Union[torch.device, str] = None,
                         dtype: torch.dtype = default_dtype) -> torch.Tensor:
        r"""Allocates the intrinsic parameters buffer of a single camera as a torch tensor.

        Args:
            *args: the values to be kept on the buffer
            num_cameras (optional, int): the number of cameras to allocated for. Default: 1
            device (optional, str):
                the torch device on which the parameters tensor should be allocated.
                Default: cpu

        Returns:
            (torch.Tensor): the allocated params tensor
        """
        assert len(args) == len(cls.param_types())  # Verify content matches subclass params enum definition
        params = torch.tensor(args, device=device, dtype=dtype)
        params = params.unsqueeze(0)
        params = params.repeat(num_cameras, 1)
        return params

    def _set_param(self, val: Union[float, torch.Tensor], param_idx: IntrinsicsParamsDefEnum):
        r"""Writes a value to the intrinsics parameters buffer of all cameras.

        Args:
            val (float or torch.Tensor): the new value to set in the intrinsics parameters buffer.
                If val is a float or a scalar tensor, this value will be set to all cameras.
                If val is a 1D torch.Tensor of size :math:`\text{num_cameras}`,
                each camera tracked by this class will be updated with the corresponding value.
            param_idx (IntrinsicsParamsDefEnum): index of the parameter to be set
        """
        if isinstance(val, float) or isinstance(val, int):  # All cameras use same value
            self.params[:, param_idx] = torch.full_like(self.params[:, 0], val)  # TODO(operel): can just use: =val
        elif val.ndim == 0:         # All cameras use same value
            self.params[:, param_idx] = val.unsqueeze(0).repeat(len(self), 1)    # TODO(operel): can just use broadcast
        elif val.ndim == 1:         # Each camera set with different value
            self.params[:, param_idx] = val

    @abstractmethod
    def zoom(self, amount):
        r"""Applies a zoom on the camera by adjusting the lens.

        Args:
            amount: Amount of adjustment
        """
        raise NotImplementedError

    def to(self, *args, **kwargs) -> CameraIntrinsics:
        """An instance of this object with the parameters tensor on the given device.
        If the specified device is the same as this object, this object will be returned.
        Otherwise a new object with a copy of the parameters tensor on the requested device will be created.

        .. seealso::

            :func:`torch.Tensor.to`
        """
        converted_params = self.params.to(*args, **kwargs)
        if self.params.dtype == converted_params.dtype and self.params.device == converted_params.device:
            return self
        else:
            new_instance = copy.deepcopy(self)
            new_instance.params = converted_params
            return new_instance

    def gradient_mask(self, *args: Union[str, IntrinsicsParamsDefEnum]) -> torch.Tensor:
        """Creates a gradient mask, which allows to backpropagate only through params designated as trainable.

        This function does not consider the requires_grad field when creating this mask.

        Args:
            *args: A vararg list of the intrinsic params that should allow gradient flow.
                   This function also supports conversion of params from their string names.
                   (i.e: 'focal_x' will convert to ``PinholeParamsDefEnum.focal_x``)

        Example:
            >>> # equivalent to:   mask = intrinsics.gradient_mask(IntrinsicsParamsDefEnum.focal_x,
            >>> #                                                  IntrinsicsParamsDefEnum.focal_y)
            >>> mask = intrinsics.gradient_mask('focal_x', 'focal_y')
            >>> intrinsics.params.register_hook(lambda grad: grad * mask.float())
            >>> # intrinsics will now allow gradient flow only for PinholeParamsDefEnum.focal_x and
            >>> # PinholeParamsDefEnum.focal_y.
        """
        try:
            # Get the enum type for this kind of intrinsics class, used to convert str args to
            # IntrinsicsParamsDefEnum subclass values
            param_def_enum = self.param_types()
            args = [param_def_enum[a] if isinstance(a, str) else a for a in args]
        except KeyError as e:
            raise ValueError(f'Camera\'s set_trainable_params() received an unsupported arg: {e}')

        mask = torch.zeros_like(self.params).bool()
        for param in args:
            mask[:, param.value] = 1.0
        return mask

    def clip_mask(self, depth: torch.Tensor) -> torch.BoolTensor:
        r"""Creates a boolean mask for clipping depth values which fall out of the view frustum.

        Args:
            depth (torch.Tensor): depth values

        Returns:
            (torch.BoolTensor):

                a mask, marking whether ``depth`` values are within the view frustum or not,
                of same shape than depth.
        """
        min_mask = depth.ge(min(self.near, self.far))
        max_mask = depth.le(max(self.near, self.far))
        return torch.logical_and(min_mask, max_mask)

    @property
    @abstractmethod
    def lens_type(self) -> str:
        raise NotImplementedError

    @property
    def device(self) -> str:
        """the torch device of parameters tensor"""
        return self.params.device

    @property
    def dtype(self):
        """the torch dtype of parameters tensor"""
        return self.params.dtype

    @property
    def requires_grad(self) -> bool:
        """True if the current intrinsics object allows gradient flow"""
        return self.params.requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        """Toggle gradient flow for the intrinsics"""
        self.params.requires_grad = val

    def parameters(self) -> torch.Tensor:
        """Returns:
            (torch.Tensor): the intrinsics parameters buffer
        """
        return self.params

    def cpu(self) -> CameraIntrinsics:
        return self.to('cpu')

    def cuda(self) -> CameraIntrinsics:
        return self.to('cuda')

    def half(self) -> CameraIntrinsics:
        return self.to(torch.float16)

    def float(self) -> CameraIntrinsics:
        return self.to(torch.float32)

    def double(self) -> CameraIntrinsics:
        return self.to(torch.float64)

    @classmethod
    def cat(cls, cameras: Sequence[CameraIntrinsics]):
        """Concatenate multiple CameraIntrinsics's.

        Assumes all cameras use the same width, height, near and far planes.

        Args:
            cameras (Sequence of CameraIntrinsics): the cameras to concatenate.

        Returns:
            (CameraIntrinsics): The concatenated cameras as a single CameraIntrinsics.
        """
        if len(cameras) == 0:
            return None
        params = [c.params for c in cameras]
        output = copy.deepcopy(cameras[0])
        output.params = torch.cat(params, dim=0)
        return output

    def set_ndc_range(self, ndc_min, ndc_max):
        """
        .. warning::

            This method is not implemented
        """
        # TODO(operel): comment out after properly testing for next version
        raise NotImplementedError('Currently only NDC space of [-1, 1] is supported.')
        # self._shared_fields['ndc_min'] = ndc_min
        # self._shared_fields['ndc_max'] = ndc_max

    def __getitem__(self, item) -> CameraIntrinsics:
        """Indexes a specific camera from the batch of cameras tracked by this class.

        Args:
            item (int or slice): Zero based camera index.

        Returns:
            (CameraIntrinsics):

                A new instance of this class viewing a single camera.
                The returned instance will track a parameters tensor, of shape :math:`(M, K)`,
                where K is the number of intrinsic parameters and M is the length of item.
                The parameters tensor of the new instance is a view of the current object parameters,
                and therefore changes to either will be reflected in both.
        """
        shallow = copy.copy(self)   # Gather all non-param fields
        params = self.params[item]
        if params.ndim < self.params.ndim:
            params = params.unsqueeze(0)
        shallow.params = params
        return shallow

    def __len__(self) -> int:
        """Returns Number of cameras tracked by this object """
        return self.params.shape[0]

    def __str__(self) -> str:
        named_params = self.named_params()
        title = f"{type(self).__name__} of {len(self)} cameras of resolution {self.width}x{self.height}.\n"
        entries = [f"Camera #{cam_idx}: {cam_params}\n" for cam_idx, cam_params in enumerate(named_params)]
        return ''.join([title] + entries)

    def __repr__(self) -> str:
        return f"{type(self).__name__} of {self.width}x{self.height}, params: {self.params.__repr__()}"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in _HANDLED_TORCH_FUNCTIONS or not all(
                issubclass(t, (torch.Tensor, CameraIntrinsics))
                for t in types
        ):
            return NotImplemented
        return _HANDLED_TORCH_FUNCTIONS[func](*args, **kwargs)

    @property
    def width(self) -> int:
        return self._shared_fields['width']

    @width.setter
    def width(self, value: int) -> None:
        self._shared_fields['width'] = value

    @property
    def height(self) -> int:
        return self._shared_fields['height']

    @height.setter
    def height(self, value: int) -> None:
        self._shared_fields['height'] = value

    @property
    def near(self) -> float:
        return self._shared_fields['near']

    @near.setter
    def near(self, value: float) -> None:
        self._shared_fields['near'] = value

    @property
    def far(self) -> float:
        return self._shared_fields['far']

    @far.setter
    def far(self, value: float) -> None:
        self._shared_fields['far'] = value

    @property
    def ndc_min(self) -> float:
        return self._shared_fields['ndc_min']

    @property
    def ndc_max(self) -> float:
        return self._shared_fields['ndc_max']


@implements(torch.allclose)
def allclose(input: CameraIntrinsics, other: CameraIntrinsics, rtol: _float = 1e-05, atol: _float = 1e-08,
            equal_nan: _bool = False) -> _bool:
    """:func:`torch.allclose` compatibility implementation for CameraIntrinsics.

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
    if type(input) != type(other):
        return False
    return torch.allclose(input.params, other.params, rtol=rtol, atol=atol, equal_nan=equal_nan) and \
           input.width == other.width and input.height == other.height
