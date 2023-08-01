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
import functools
from copy import deepcopy
import torch
import inspect
from typing import Sequence, List, Dict, Union, Tuple, Type, FrozenSet, Callable
from torch.types import _float, _bool
from .extrinsics import CameraExtrinsics, ExtrinsicsParamsDefEnum
from .intrinsics import CameraIntrinsics, IntrinsicsParamsDefEnum
from .intrinsics_ortho import OrthographicIntrinsics
from .intrinsics_pinhole import PinholeIntrinsics

__all__ = [
    'Camera',
    'allclose'
]


_EXTRINSICS_MODULES = [CameraExtrinsics, ]
"""Kaolin modules which support camera extrinsic related functionality """
_INTRINSICS_MODULES = [OrthographicIntrinsics, PinholeIntrinsics]
"""Kaolin modules which support camera intrinsic related functionality """

CameraModuleType = Union[Type[CameraExtrinsics], Type[CameraIntrinsics]]
"""An alias for all camera modules, identified as subtypes of CameraExtrinsics or CameraIntrinsics"""

_HANDLED_TORCH_FUNCTIONS = dict()
"""Registered torch functions the Camera class implements"""


def implements(torch_function):
    """Registers a torch function override for Camera"""
    @functools.wraps(torch_function)
    def decorator(func):
        _HANDLED_TORCH_FUNCTIONS[torch_function] = func
        return func
    return decorator


def _gather_constructors(*cam_modules: CameraModuleType) -> Dict[FrozenSet, Tuple[Callable, List]]:
    r"""Given a variable list of camera modules, returns a mapping of their constructors,
    used to disambiguate which ctor should be called according to kwargs.

    The mapping can be used to disambiguate which ctor should be called according to kwargs.

    Args:
        *cam_modules (CameraModuleType):
            A variable list of CameraExtrinsics or CameraIntrinsic classes or their subtypes. This function
            will gather constructors from these classes.

    Return:
        (dict):
        
           a mapping of their constructors in the following format:
           `unique_arg_identifiers -> (func, args)
           where:

               - `unique_arg_identifiers` is a hashable, frozenset of the mandatory args that uniquely identify each
               constructor.
               - `func` is a reference to the ctor class function
               - `args` is the full list of kwargs required by the ctor

    Example:
        >>> _gather_constructors(PinholeIntrinsics)
        {
            frozenset(('width', 'height', 'focal_x')):
                (PinholeIntrinsics.from_focal, [width, height, focal_x, focal_y, x0, y0, ..])
            frozenset(('width', 'height', 'fov')):
                PinholeIntrinsics.from_fov, [width, height, fov, fov_direction, x0, y0, ..])
        }
    """
    ctors = []
    for cam_module in cam_modules:
        # Constructors are @classmethod with a 'from_' prefix
        is_ctor = lambda x: inspect.ismethod(x) and x.__name__.startswith('from_')

        # Get all methods that satisfy the constructor predicate.
        ctors.extend(inspect.getmembers(cam_module, predicate=is_ctor))
    # The return value is a tuple per entry, take the 2nd element with the method reference
    ctors = [c[1] for c in ctors]

    def _ctor_funcs_to_args(func):
        argspec = inspect.getfullargspec(func)
        args = argspec.args[1:]
        if 'cls' in args:
            args.remove('cls')
        mandatory_args = args[:-len(argspec.defaults)]
        key = frozenset(mandatory_args)
        return key, (func, args)
    return dict(map(_ctor_funcs_to_args, ctors))


class Camera:
    r"""Camera is a one-stop class for all camera related differentiable / non-differentiable transformations.

    Camera objects are represented by *batched* instances of 2 submodules:

        - :class:`CameraExtrinsics`: The extrinsics properties of the camera (position, orientation).
          These are usually embedded in the view matrix, used to transform vertices from world space to camera space.
        - :class:`CameraIntrinsics`: The intrinsics properties of the lens
          (such as field of view / focal length in the case of pinhole cameras).
          Intrinsics parameters vary between different lens type,
          and therefore multiple CameraIntrinsics subclasses exist,
          to support different types of cameras: pinhole / perspective, orthographic, fisheye, and so forth.
          For pinehole and orthographic lens, the intrinsics are embedded in a projection matrix.
          The intrinsics module can be used to transform vertices from camera space to Normalized Device Coordinates.

    .. note::

        To avoid tedious invocation of camera functions through
        ``camera.extrinsics.someop()`` and ``camera.intrinsics.someop()``, kaolin overrides the ``__get_attributes__``
        function to forward any function calls of ``camera.someop()`` to
        the appropriate extrinsics / intrinsics submodule.

    The entire pipeline of transformations can be summarized as (ignoring homogeneous coordinates)::

        World Space                                         Camera View Space
             V         ---CameraExtrinsics.transform()--->         V'          ---CameraIntrinsics.transform()---
        Shape~(B, 3)            (view matrix)                  Shape~(B, 3)                                     |
                                                                                                                |
                                                                               (linear lens: projection matrix) |
                                                                                      + homogeneus -> 3D        |
                                                                                                                V
                                                                                     Normalized Device Coordinates (NDC)
                                                                                                Shape~(B, 3)

        When using view / projection matrices, conversion to homogeneous coordinates is required.
        Alternatively, the `transform()` function takes care of such projections under the hood when needed.

    How to apply transformations with kaolin's Camera:

        1. Linear camera types, such as the commonly used pinhole camera,
           support the :func:`view_projection_matrix()` method.
           The returned matrix can be used to transform vertices through pytorch's matrix multiplication, or even be
           passed to shaders as a uniform.
        2. All Cameras are guaranteed to support a general :func:`transform()` function
           which maps coordinates from world space to Normalized Device Coordinates space.
           For some lens types which perform non linear transformations,
           the :func:`view_projection_matrix()` is non-defined.
           Therefore the camera transformation must be applied through
           a dedicated function. For linear cameras,
           :func:`transform()` may use matrices under the hood.
        3. Camera parameters may also be queried directly.
           This is useful when implementing camera params aware code such as ray tracers.

    How to control kaolin's Camera:

        - :class:`CameraExtrinsics`: is packed with useful methods for controlling the camera position and orientation:
          :func:`translate() <CameraExtrinsics.translate()>`,
          :func:`rotate() <CameraExtrinsics.rotate()>`,
          :func:`move_forward() <CameraExtrinsics.move_forward()>`,
          :func:`move_up() <CameraExtrinsics.move_up()>`,
          :func:`move_right() <CameraExtrinsics.move_right()>`,
          :func:`cam_pos() <CameraExtrinsics.cam_pos()>`,
          :func:`cam_up() <CameraExtrinsics.cam_up()>`,
          :func:`cam_forward() <CameraExtrinsics.cam_forward()>`,
          :func:`cam_up() <CameraExtrinsics.cam_up()>`.
        - :class:`CameraIntrinsics`: exposes a lens :func:`zoom() <CameraIntrinsics.zoom()>`
          operation. The exact functionality depends on the camera type.

    How to optimize the Camera parameters:

        - Both :class:`CameraExtrinsics`: and :class:`CameraIntrinsics` maintain
          :class:`torch.Tensor` buffers of parameters which support pytorch differentiable operations.
        - Setting ``camera.requires_grad_(True)`` will turn on the optimization mode.
        - The :func:`gradient_mask` function can be used to mask out gradients of specific Camera parameters.

        .. note::

            :class:`CameraExtrinsics`: supports multiple representions of camera parameters
            (see: :func:`switch_backend <CameraExtrinsics.switch_backend()>`).
            Specific representations are better fit for optimization
            (e.g.: they maintain an orthogonal view matrix).
            Kaolin will automatically switch to using those representations when gradient flow is enabled
            For non-differentiable uses, the default representation may provide better
            speed and numerical accuracy.

    Other useful camera properties:

        - Cameras follow pytorch in part, and support arbitrary ``dtype`` and ``device`` types through the
          :func:`to()`, :func:`cpu()`, :func:`cuda()`, :func:`half()`, :func:`float()`, :func:`double()`
          methods and :func:`dtype`, :func:`device` properties.
        - :class:`CameraExtrinsics`: and :class:`CameraIntrinsics`: individually support the :func:`requires_grad`
          property.
        - Cameras implement :func:`torch.allclose` for comparing camera parameters under controlled numerical accuracy.
          The operator ``==`` is reserved for comparison by ref.
        - Cameras support batching, either through construction, or through the :func:`cat()` method.

        .. note::

            Since kaolin's cameras are batched, the view/projection matrices are of shapes :math:`(\text{num_cameras}, 4, 4)`,
            and some operations, such as :func:`transform()` may return values as shapes of :math:`(\text{num_cameras}, \text{num_vectors}, 3)`.

    Concluding remarks on coordinate systems and other confusing conventions:

        - kaolin's Cameras assume column major matrices, for example, the inverse view matrix (cam2world) is defined as:

          .. math::

              \begin{bmatrix}
                  r1 & u1 & f1 & px \\
                  r2 & u2 & f2 & py \\
                  r3 & u3 & f3 & pz \\
                  0 & 0 & 0 & 1
              \end{bmatrix}

          This sometimes causes confusion as the view matrix (world2cam) uses a transposed 3x3 submatrix component,
          which despite this transposition is still column major (observed through the last `t` column):

          .. math::

              \begin{bmatrix}
                  r1 & r2 & r3 & tx \\
                  u1 & u2 & u3 & ty \\
                  f1 & f2 & f3 & tz \\
                  0 & 0 & 0 & 1
              \end{bmatrix}

        - kaolin's cameras do not assume any specific coordinate system for the camera axes. By default, the
          right handed cartesian coordinate system is used. Other coordinate systems are supported through
          :func:`change_coordinate_system() <CameraExtrinsics.change_coordinate_system()>`
          and the ``coordinates.py`` module::

                Y
                ^
                |
                |---------> X
               /
             Z

        - kaolin's NDC space is assumed to be left handed (depth goes inwards to the screen).
          The default range of values is [-1, 1].
    """

    _extrinsics_constructors = _gather_constructors(*_EXTRINSICS_MODULES)
    """Minimal arguments required to disambiguate & invoke the different extrinsics constructors.
    (unique_arg_identifiers) -> (func, args)
    """

    _intrinsics_constructors = _gather_constructors(*_INTRINSICS_MODULES)
    """Minimal arguments required to disambiguate & invoke the different extrinsics constructors.
    (unique_arg_identifiers) -> (func, args)
    """

    def __init__(self, extrinsics: CameraExtrinsics, intrinsics: CameraIntrinsics):
        r"""Constructs a new camera module from the pre-constructed extrinsics and intrinsics components.

        .. seealso::

            :func:`Camera.from_args`

        Args:
            extrinsics (CameraExtrinsics):
                A component containing the extrinsic information of the Camera, used to construct a view matrix

            intrinsics (CameraIntrinsics):
                A component containing the intrinsic information of the Camera, used to transform from camera
                space to NDC space.

        """
        assert len(extrinsics) == len(intrinsics)
        assert extrinsics.device == intrinsics.device
        self.extrinsics: CameraExtrinsics = extrinsics
        self.intrinsics: CameraIntrinsics = intrinsics

    @classmethod
    def from_args(cls, **kwargs):
        r"""A convenience constructor for the camera class, which takes all extrinsics & intrinsics arguments
        at once, and disambiguates them to construct a complete camera object.

        The correct way of using this constructor is by specifying the camera args as \**kwargs, for example::

            # Construct a pinhole camera with perspective projection
            Camera.from_args(
                eye=torch.tensor([10.0, 0.0, 0.0]),
                at=torch.tensor([0.0, 0.0, 0.0]),
                up=torch.tensor([0.0, 1.0, 0.0]),
                fov=30 * np.pi / 180,   # alternatively focal_x, optionally specify: focal_y, x0, y0
                width=800, height=800,
                near=1e-2, far=1e2,
                dtype=torch.float64,
                device='cuda'
            )
            # Construct an orthographic camera
            Camera.from_args(
                eye=np.array([10.0, 0.0, 4.0]),
                at=np.array([0.0, 0.0, 0.0]),
                up=np.array([0.0, 1.0, 0.0]),
                width=800, height=800,
                near=-800, far=800,
                fov_distance=1.0,
                dtype=torch.float32,
                device='cuda'
            )
            # Construct a pinhole camera
            Camera.from_args(
                view_matrix=torch.tensor([[1.0, 0.0, 0.0, 0.5],
                                          [0.0, 1.0, 0.0, 0.5],
                                          [0.0, 0.0, 1.0, 0.5],
                                          [0.0, 0.0, 0.0, 1.0]]),
                focal_x=1000,
                width=1600, height=1600,
            )

        Args:
            **kwargs (dict of str, *):
                keywords specifying the parameters of the camera.
                Valid options are a combination of extrinsics, intrinsics and general properties:

                    * Extrinsic params: ``eye``, ``at``, ``up`` / ``view_matrix`` / ``cam_pos``, ``cam_dir``
                    * Perspective intrinsic params: ``fov`` / ``focal_x``,
                      optionally: ``x0``, ``y0``, ``focal_y``, ``fov_direction``
                    * Orthographic intrinsic params: ``fov_distance``
                      optionally: ``x0``, ``y0``
                    * General intrinsic dimensions: ``width``, ``height``, optionally: ``near``, ``far``
                    * Tensor params properties - optionally: ``device``, ``dtype``
        """
        call_args = frozenset(kwargs)
        extrinsic_key = [k for k in Camera._extrinsics_constructors.keys() if k.issubset(call_args)]
        intrinsic_key = [k for k in Camera._intrinsics_constructors.keys() if k.issubset(call_args)]

        if len(extrinsic_key) != 1:
            raise ValueError('Camera construction failed due to ambiguous parameters: '
                             f'{list(kwargs.keys())}')
        extrinsic_key = extrinsic_key[0]
        extrinsic_ctor, extrinsic_args = Camera._extrinsics_constructors[extrinsic_key]

        if len(intrinsic_key) == 0:
            # Protect against empty match
            intrinsic_key = None
        elif len(intrinsic_key) == 1:
            # call args should ideally only match a single extrinsics & intrinsics constructor key
            intrinsic_key = intrinsic_key[0]
        else:
            # If more than one intrinsics constructor matches the args, check the other direction:
            # are all given callargs contained in the combined extrinsics & intrinsics ctors.
            def _is_callargs_subset_of_ctor(key):
                _, intrinsic_args = Camera._intrinsics_constructors[key]
                candidate_call_args = set(extrinsic_args).union(set(intrinsic_args))
                return call_args.issubset(candidate_call_args)
            intrinsic_key = list(filter(_is_callargs_subset_of_ctor, intrinsic_key))

            # Finally, check if all remaining matches are a subset of a single constructor.
            # If so, choose this constructor as it is the one most explicitly referred.
            if len(intrinsic_key) > 0:
                longest_match = max(intrinsic_key, key=lambda s: len(s))
                is_all_matches_subset_of_longest = all([s.issubset(longest_match) for s in intrinsic_key])
                intrinsic_key = longest_match if is_all_matches_subset_of_longest else None

        if intrinsic_key is None:
            raise ValueError(f'Camera construction failed due to ambiguous parameters: {list(kwargs.keys())}')
        intrinsic_ctor, intrinsic_args = Camera._intrinsics_constructors[intrinsic_key]

        extrinsic_args = {k: v for k, v in kwargs.items() if k in call_args.intersection(extrinsic_args)}
        tensors_devices = set([arg.device for arg in extrinsic_args.values() if isinstance(arg, torch.Tensor)])
        if 'device' not in extrinsic_args and len(tensors_devices) > 1:
            raise ValueError(f'Camera construction with tensors args on different devices is not allowed '
                             f'without explicitly specifying the Camera "device". Please '
                             f'review the Camera input args: {list(kwargs.keys())}')

        extrinsics = extrinsic_ctor(**extrinsic_args)

        _intrinsic_args = {k: v for k, v in kwargs.items() if k in call_args.intersection(intrinsic_args)}
        # Make sure dtype and device are consistent with extrinsics
        _intrinsic_args['device'] = extrinsics.device
        _intrinsic_args['dtype'] = extrinsics.dtype
        # Support broadcasting of intrinsics in case extrinsics are batched
        # (for intrinsic ctors supporting a number of cameras param)
        if 'num_cameras' in intrinsic_args and 'num_cameras' not in _intrinsic_args:
            _intrinsic_args['num_cameras'] = len(extrinsics)
        intrinsics = intrinsic_ctor(**_intrinsic_args)
        return Camera(extrinsics=extrinsics, intrinsics=intrinsics)

    def parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the full parameters set of the camera,
        divided to extrinsics and intrinsics parameters

        Returns:
            (torch.Tensor, torch.Tensor):
                the extrinsics and the intrinsics parameters.
        """
        return self.extrinsics.parameters(), self.intrinsics.parameters()

    def gradient_mask(self, *args: Union[str, ExtrinsicsParamsDefEnum, IntrinsicsParamsDefEnum]) -> torch.Tensor:
        """Creates a gradient mask, which allows to backpropagate only through
        params designated as trainable.

        This function does not consider the :attr:`requires_grad` field when creating this mask.

        .. note::

            The 3 extrinsics camera axes are always masked as trainable together.
            This design choice ensures that these axes, as well as the view matrix, remain orthogonal.

        Args:
            *args :
                A vararg list of the extrinsic and intrinsic params that should allow gradient flow.
                This function also supports conversion of params from their string names.
                (i.e: 't' will convert to ``PinholeParamsDefEnum.t``).

        Returns:
            (torch.BoolTensor, torch.BoolTensor):
                the gradient masks, of same shapes than
                ``self.extrinsics.parameters()`` and ``self.intrinsics.parameters()``.

        Example:
            >>> extrinsics_mask, intrinsics_mask = camera.gradient_mask('t', 'focal_x', 'focal_y')
            >>> # equivalent to the args:
            >>> # ExtrinsicsParamsDefEnum.t, IntrinsicsParamsDefEnum.focal_x, IntrinsicsParamsDefEnum.focal_y
            >>> extrinsics_params, intrinsic_params = camera.params()
            >>> extrinsics_params.register_hook(lambda grad: grad * extrinsics_mask.float())
            >>> # extrinsics will now allow gradient flow only for the camera location
            >>> intrinsic_params.register_hook(lambda grad: grad * intrinsics_mask.float())
            >>> # intrinsics will now allow gradient flow only for the focal length
        """
        args = set(args)
        str_args = set([a for a in args if isinstance(a, str)])
        all_extrinsic_params_str = [e.name for e in ExtrinsicsParamsDefEnum]
        all_extrinsic_args = set([a for a in args if isinstance(a, ExtrinsicsParamsDefEnum)])
        all_intrinsic_args = set([a for a in args if isinstance(a, IntrinsicsParamsDefEnum)])
        extrinsic_args = str_args.intersection(all_extrinsic_params_str).union(all_extrinsic_args)
        intrinsic_args = str_args.difference(all_extrinsic_params_str).union(all_intrinsic_args)
        return self.extrinsics.gradient_mask(*extrinsic_args), self.intrinsics.gradient_mask(*intrinsic_args)

    @property
    def width(self) -> int:
        """Camera image plane width (pixel resolution)"""
        return self.intrinsics.width

    @width.setter
    def width(self, value: int) -> None:
        """Camera image plane width (pixel resolution)"""
        self.intrinsics.width = value

    @property
    def height(self) -> int:
        """Camera image plane height (pixel resolution)"""
        return self.intrinsics.height

    @height.setter
    def height(self, value: int) -> None:
        """Camera image plane height (pixel resolution)"""
        self.intrinsics.height = value

    @property
    def lens_type(self) -> str:
        r"""A textual description of the camera lens type. (i.e 'pinhole', 'ortho')
        """
        return self.intrinsics.lens_type

    @property
    def device(self) -> torch.device:
        """torch device of parameters tensors"""
        assert self.extrinsics.device == self.intrinsics.device, \
            'Camera extrinsics and intrinsics use different devices'
        return self.extrinsics.device

    @property
    def dtype(self) -> torch.dtype:
        """torch dtype of parameters tensors"""
        assert self.extrinsics.dtype == self.intrinsics.dtype, \
            'Camera extrinsics and intrinsics use different dtypes'
        return self.extrinsics.dtype

    def requires_grad_(self, val: bool):
        """Toggle gradient flow for both extrinsics and intrinsics params.

        .. note::

            To read the requires_grad attribute access the extrinsics / intrinsics components
            explicitly, as their requires_grad status may differ.
        """
        self.extrinsics.requires_grad = val
        self.intrinsics.requires_grad = val

    def to(self,  *args, **kwargs) -> Camera:
        return Camera(extrinsics=self.extrinsics.to( *args, **kwargs), intrinsics=self.intrinsics.to( *args, **kwargs))

    def cpu(self) -> Camera:
        return Camera(extrinsics=self.extrinsics.cpu(), intrinsics=self.intrinsics.cpu())

    def cuda(self) -> Camera:
        return Camera(extrinsics=self.extrinsics.cuda(), intrinsics=self.intrinsics.cuda())

    def half(self) -> Camera:
        return Camera(extrinsics=self.extrinsics.half(), intrinsics=self.intrinsics.half())

    def float(self) -> Camera:
        return Camera(extrinsics=self.extrinsics.float(), intrinsics=self.intrinsics.float())

    def double(self) -> Camera:
        return Camera(extrinsics=self.extrinsics.double(), intrinsics=self.intrinsics.double())

    def transform(self, vectors: torch.Tensor):
        r"""Applies extrinsic and instrinsic projections consecutively,
        thereby projecting the vectors from world to NDC space.

        Args:
            vectors (torch.Tensor):
                the vectors to transform,
                of shape :math:`(\text{batch_size}, 3)` or
                :math:`(\text{num_cameras}, \text{batch_size}, 3)`.

        Returns:
            (torch.Tensor):
            
                The vectors projected to NDC space, of the same shape as ``vectors``,
                transform can be broadcasted.
        """
        post_view = self.extrinsics.transform(vectors)
        post_proj = self.intrinsics.transform(post_view)
        # Broadcast for single camera in batch: Reshape output to (B, 3) or (C, B, 3) according to input vectors
        if len(self) == 1:
            post_proj = post_proj.reshape(vectors.shape)
        return post_proj

    def view_projection_matrix(self):
        """Return the composed view projection matrix.

        .. note::

            Works only for cameras with linear projection transformations.

        Returns:
            (torch.Tensor): The view projection matrix, of shape :math:`(\text{num_cameras}, 4, 4)`
        """
        view = self.extrinsics.view_matrix()
        projection = self.intrinsics.projection_matrix()
        return torch.bmm(projection, view)

    @classmethod
    def cat(cls, cameras: Sequence[Camera]):
        """Concatenate multiple Camera's.

        Assumes all cameras use the same width, height, near and far planes.

        Args:
            cameras (Sequence of Camera): the cameras to concatenate.

        Returns:
            (Camera): The concatenated cameras as a single Camera.
        """
        return Camera(extrinsics=CameraExtrinsics.cat([c.extrinsics for c in cameras]),
                      intrinsics=CameraIntrinsics.cat([c.intrinsics for c in cameras]))

    def __getattr__(self, item):
        """Allows for an easier API - camera attributes are routed to intrinsic / extrinsic components
        """
        if item.startswith('__') and item.endswith('__'):
            raise AttributeError

        extrinsic_attr = hasattr(self.extrinsics, item)
        intrinsic_attr = hasattr(self.intrinsics, item)
        assert not intrinsic_attr or not extrinsic_attr, \
            "Camera cannot implicitly route attribute to extrinsic or intrinsic components " + \
            f"as both have similar named attribute {item}"

        if extrinsic_attr:
            return getattr(self.extrinsics, item)
        elif intrinsic_attr:
            return getattr(self.intrinsics, item)
        else:
            raise AttributeError

    def __setattr__(self, item, value):
        """Allows for an easier API - camera attributes are routed to intrinsic / extrinsic components
        """
        if item in ("extrinsics", "intrinsics") or (item.startswith('__') and item.endswith('__')):
            super().__setattr__(item, value)
        else:
            extrinsics = getattr(self, "extrinsics")
            intrinsics = getattr(self, "intrinsics")
            extrinsic_attr = hasattr(extrinsics, item)
            intrinsic_attr = hasattr(intrinsics, item)

            if extrinsic_attr and intrinsic_attr:
                raise AttributeError(
                    f'Attribute "{item}" is defined in both CameraExtrinsics and '
                    'CameraIntrinsics classes. Therefore implicitly setting a new value '
                    'through the Camera is ambiguous.')
            elif extrinsic_attr:
                setattr(extrinsics, item, value)
            elif intrinsic_attr:
                setattr(intrinsics, item, value)
            else:
                super().__setattr__(item, value)    # Set attribute for this class

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __eq__(self, other):
        if not isinstance(other, Camera):
            return False
        return self.extrinsics == other.extrinsics and self.intrinsics == other.intrinsics

    def __getitem__(self, item):
        return Camera(extrinsics=self.extrinsics[item], intrinsics=self.intrinsics[item])

    def __len__(self):
        return len(self.extrinsics)  # Assumed to be identical to length of intrinsics

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in _HANDLED_TORCH_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, Camera))
            for t in types
        ):
            return NotImplemented
        return _HANDLED_TORCH_FUNCTIONS[func](*args, **kwargs)

    def named_params(self) -> List[Dict[str, float]]:
        """Get a descriptive list of named parameters per camera.

        Returns:
            (list of dict): The named parameters.
        """
        return [dict(e, **i) for e, i in zip(self.extrinsics.named_params(), self.intrinsics.named_params())]

    def __str__(self) -> str:
        return self.extrinsics.__str__() + '\n' + self.intrinsics.__str__()

    def __repr__(self) -> str:
        return self.extrinsics.__repr__() + '\n' + self.intrinsics.__repr__()


@implements(torch.allclose)
def allclose(input: Camera, other: Camera, rtol: _float = 1e-05, atol: _float = 1e-08,
            equal_nan: _bool = False) -> _bool:
    """This function checks if the camera extrinsics and intrinsics,
    are close using :func:`torch.allclose`.

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
    return torch.allclose(input.extrinsics, other.extrinsics, rtol=rtol, atol=atol, equal_nan=equal_nan) and \
           torch.allclose(input.intrinsics, other.intrinsics, rtol=rtol, atol=atol, equal_nan=equal_nan)
