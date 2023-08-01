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
from typing import Optional, Type, Union
import numpy as np
import torch
from .intrinsics import CameraIntrinsics, IntrinsicsParamsDefEnum, CameraFOV, \
    up_to_homogeneous, down_from_homogeneous, default_dtype

__all__ = [
    'PinholeIntrinsics'
]

class PinholeParamsDefEnum(IntrinsicsParamsDefEnum):
    x0 = 0          # Principal point offset (x), by default assumes offset from the canvas center
    y0 = 1          # Principal point offset (y), by default assumes offset from the canvas center
    focal_x = 2     # Focal length (x), measured in pixels
    focal_y = 3     # Focal length (y), measured in pixels
    # Following common practice, the axis skew of pinhole cameras is always assumed to be zero

class PinholeIntrinsics(CameraIntrinsics):
    r"""Holds the intrinsics parameters of a pinhole camera:
    how it should project from camera space to normalized screen / clip space.
    The intrinsics parameters are used to define the lens attributes of the perspective projection matrix.

    The pinhole camera explicitly exposes the projection transformation matrix.
    This may typically be useful for rasterization based rendering pipelines (i.e: OpenGL).
    See documentation of CameraIntrinsics for numerous ways of how to use this class.

    Kaolin assumes a left handed NDC coordinate system: after applying the projection matrix,
    the depth increases inwards into the screen.

    The complete perspective matrix can be described by the following factorization:

    .. math::

        \text{FullProjectionMatrix} = &\text{Ortho} \times \text{Depth Scale} \times \text{Perspective} \\
        = &\begin{bmatrix}
            2/(r-l) & 0 & 0 & tx \\
            0 & 2/(t-b) & 0 & ty \\
            0 & 0 & -2/(f-n) & tz \\
            0 & 0 & 0 & 1
        \end{bmatrix} \\
        \times &\begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & B & A \\
            0 & 0 & 0 & -1
        \end{bmatrix} \\
        \times &\begin{bmatrix}
            \text{focal_x} & 0 & -x0 & 0 \\
            0 & \text{focal_y} & -y0 & 0 \\
            0 & 0 & 0 & 1 \\
            0 & 0 & 1 & 0
        \end{bmatrix} \\

        = \begin{bmatrix}
            2*\text{focal_x}/(r - l) & 0 & -2x0/(r - l) - tx & 0 \\
            0 & 2*\text{focal_y}/(t - b) & -2y0/(t - b) - ty & 0 \\
            0 & 0 & V & U \\
            0 & 0 & -1 & 0
        \end{bmatrix}

    where:

        - **focal_x**, **focal_y**, **x0**, **y0**: are the intrinsic parameters of the camera
          The focal length, together with the image plane width / height,
          determines the field of view (fov). This is the effective lens zoom of the scene.
          The principal point offsets: x0, y0 allow another DoF to translate the origin of the image plane.
          By default, kaolin assumes the NDC origin is at the canvas center (see projection_matrix())
        - **n**, **f**: are the near and far clipping planes,
          which define the min / max depth of the view frustum.
          The near and far planes are also used to normalize the depth values to
          normalized device coordinates (see :func:`ndc_matrix()` documentation).
        - **r**, **l**, **t**, **b**: are the right, left, top and bottom borders of the
          view frustum, and are defined by the perspective
          fov (derived from the focal length) and image plane dimensions.
        - **tx**, **ty**, **tz**: are defined as:

          :math:`tx = -(r + l) / (r - l)`

          :math:`ty = -(t + b) / (t - b)`

          :math:`tz = -(f + n) / (f - n)`
        - **U**, **V**: are elements which define the NDC range, see :func:`ndc_matrix()` for
          an elaboration on how these are defined.
        - **A**, **B**: can be reverse engineered from U, V and are uniquely defined by them
          (and in fact serve a similar function).

    This matrix sometimes appear in the literature in a slightly simplified form, for example,
    if the principal point offsets x0 = 0, y0 = 0 and the
    NDC coords are defined in the range :math:`[-1, 1]`:

    .. math::

        \begin{bmatrix}
            2*\text{focal_x}/(r-l) & 0 & -tx & 0 \\
            0 & 2*\text{focal_y}/(t - b) & -ty & 0 \\
            0 & 0 & V & U \\
            0 & 0 & -1 & 0
        \end{bmatrix}

    The resulting vector multiplied by this matrix is in homogeneous clip space,
    and requires division by the 4th
    coordinate (w) to obtain the final NDC coordinates.

    Since the choice of NDC space is application dependent,
    kaolin maintains the separation of Perspective matrix,
    which depends only on the choice of intrinsic parameters from the Depth Scale and Ortho matrices,
    (which are squashed together to define the view frustum and NDC range).

    .. seealso::

        :func:`perspective_matrix()` and :func:`ndc_matrix()` functions.

    This class is batched and may hold information from multiple cameras.
    Parameters are stored as a single tensor of shape :math:`(\text{num_cameras}, 4)`.

    The matrix returned by this class supports differentiable torch operations,
    which in turn may update the intrinsic parameters of the camera.
    """
    # Default near / far values are best used for small / medium scale scenes.
    # Use cases with bigger scale should use other explicit values.
    DEFAULT_NEAR = 1e-2
    DEFAULT_FAR = 1e2

    def __init__(self, width: int, height: int, params: torch.Tensor,
                 near: float = DEFAULT_NEAR, far: float = DEFAULT_FAR):
        super().__init__(width, height, params, near, far)

    @classmethod
    def param_types(cls) -> Type[IntrinsicsParamsDefEnum]:
        """
        Returns:
            (IntrinsicsParamsDefEnum):

                an enum describing each of the intrinsic parameters managed by the pinhole camera.
                This enum also defines the order in which values are kept within the params buffer.
        """
        return PinholeParamsDefEnum

    @property
    def lens_type(self) -> str:
        return 'pinhole'

    @classmethod
    def from_focal(cls,
                   width: int, height: int,
                   focal_x: float, focal_y: Optional[float] = None,
                   x0: Optional[float] = None, y0: Optional[float] = None,
                   near: float = DEFAULT_NEAR, far: float = DEFAULT_FAR,
                   num_cameras: int = 1,
                   device: Union[torch.device, str] = None,
                   dtype: torch.dtype = default_dtype) -> PinholeIntrinsics:
        """Constructs a new instance of PinholeIntrinsics from focal length

        Args:
             width (int): width of the camera resolution
             height (int): height of the camera resolution
             focal_x (float): focal on x-axis
             focal_y (optional, float): focal on y-axis. Default: same that focal_x
             x0 (optional, float): horizontal offset from origin of the image plane (by default the center). Default: 0.
             y0 (optional, float): vertical offset origin of the image place (by default the center). Default: 0.
             near (optional, float):
                 near clipping plane, define the min depth of the view frustrum
                 or to normalize the depth values. Default: 1e-2
             far (optional, float):
                 far clipping plane, define the max depth of teh view frustrum
                 or to normalize the depth values. Default: 1e2
             num_cameras (optional, int): the numbers of camera in this object. Default: 1
             device (optional, str): the device on which parameters will be allocated. Default: cpu
             dtype (optional, str): the dtype on which parameters will be alloacted. Default: torch.float

        Returns:
            (PinholeInstrinsics): the constructed pinhole camera intrinsics
        """
        if x0 is None:
            x0 = 0.0
        if y0 is None:
            y0 = 0.0
        focal_y = focal_y if focal_y else focal_x
        params = cls._allocate_params(x0, y0, focal_x, focal_y, num_cameras=num_cameras, device=device, dtype=dtype)
        return PinholeIntrinsics(width, height, params, near, far)

    @classmethod
    def from_fov(cls, width: int, height: int,
                 fov: float, fov_direction: CameraFOV = CameraFOV.VERTICAL,
                 x0: Optional[float] = 0., y0: Optional[float] = 0.,
                 near: float = DEFAULT_NEAR, far: float = DEFAULT_FAR,
                 num_cameras: int = 1,
                 device: Union[torch.device, str] = None,
                 dtype: torch.dtype = default_dtype) -> PinholeIntrinsics:
        """
        Constructs a new instance of PinholeIntrinsics from field of view

        Args:
             width (int): width of the camera resolution
             height (int): height of the camera resolution
             fov (float): the field of view, in radians
             fov_direction (optional, CameraFOV): the direction of the field-of-view
             x0 (optional, float): horizontal offset from origin of the image plane (by default the center). Default: 0.
             y0 (optional, float): vertical offset origin of the image place (by default the center). Default: 0.
             near (optional, float):
                 near clipping plane, define the min depth of the view frustrum
                 or to normalize the depth values. Default: 1e-2
             far (optional, float):
                 far clipping plane, define the max depth of teh view frustrum
                 or to normalize the depth values. Default: 1e2
             num_cameras (optional, int): the numbers of camera in this object. Default: 1
             device (optional, str): the device on which parameters will be allocated. Default: cpu
             dtype (optional, str): the dtype on which parameters will be alloacted. Default: torch.float

        Returns:
            (PinholeInstrinsics): the constructed pinhole camera intrinsics
        """
        assert fov_direction in (CameraFOV.HORIZONTAL, CameraFOV.VERTICAL),\
            "fov direction must be vertical or horizontal"
        tanHalfAngle = np.tan(fov / 2.0)
        aspectScale = width / 2.0 if fov_direction is CameraFOV.HORIZONTAL else height / 2.0
        focal = aspectScale / tanHalfAngle
        params = cls._allocate_params(x0, y0, focal, focal, num_cameras=num_cameras, device=device, dtype=dtype)
        return PinholeIntrinsics(width, height, params, near, far)

    def perspective_matrix(self) -> torch.Tensor:
        r"""Constructs a matrix which performs perspective projection from camera space to homogeneous clip space.

        The perspective matrix embeds the pinhole camera intrinsic parameters,
        which together with the near / far clipping planes specifies how the view-frustum should be transformed
        into a cuboid-shaped space. The projection does not affect visibility of objects,
        but rather specifies how the 3D world should be down-projected to a 2D image.

        This matrix does not perform clipping and is not concerned with NDC coordinates, but merely
        describes the perspective transformation itself.
        This leaves this matrix free from any api specific conventions of the NDC space.

        When coupled with :func:`ndc_matrix()`, the combination of these two matrices produces a complete perspective
        transformation from camera space to NDC space, which by default is aligned to traditional OpenGL standards.
        See also :func:`projection_matrix()`, which produces a squashed matrix of these two operations together.

        The logic essentially builds an torch autodiff compatible equivalent of the following tensor:

        .. math::

            \text{perspective_matrix} = \begin{bmatrix}
                \text{focal_x} & 0. & -x0 & 0. \\
                0. & \text{focal_y} & -y0 & 0. \\
                0. & 0. & 0. & 1. \\
                0. & 0. & 1. & 0.
            \end{bmatrix}

        which is a modified form of the intrinsic camera matrix:

        .. math::

            \begin{bmatrix}
                \text{focal_x} & 0. & x0 \\
                0. & \text{focal_y} & y0 \\
                0. & 0. & 1.
            \end{bmatrix}

        Returns:
            (torch.Tensor): The perspective matrix, of shape :math:`(\text{num_cameras}, 4, 4)`
        """
        zero = torch.zeros_like(self.focal_x)
        one = torch.ones_like(self.focal_x)
        rows = [
            torch.stack([self.focal_x, zero,           -self.x0,    zero],       dim=-1),
            torch.stack([zero,         self.focal_y,   -self.y0,    zero],       dim=-1),
            torch.stack([zero,         zero,            zero,       one],        dim=-1),
            torch.stack([zero,         zero,            one,        zero],       dim=-1)
        ]
        persp_mat = torch.stack(rows, dim=1)
        return persp_mat

    def ndc_matrix(self, left, right, bottom, top, near, far) -> torch.Tensor:
        r"""Constructs a matrix which performs the required transformation to project the scene onto the view frustum.
        (that is: it normalizes a cuboid-shaped view-frustum to clip coordinates, which are
        SCALED normalized device coordinates).

        When used in conjunction with a :func:`perspective_matrix()`, a transformation from camera view space to
        clip space can be obtained.
        
        .. seealso::

            projection_matrix() which combines both operations.

        .. note::
        
            This matrix actually converts coordinates to clip space, and requires an extra division by the w
            coordinates to obtain the NDC coordinates. However, it is named **ndc_matrix** as the elements are chosen
            carefully according to the definitions of the NDC space.

        Vectors transformed by this matrix will reside in the kaolin clip space,
        which is left handed (depth increases in the direction that goes inwards the screen)::

            Y      Z
            ^    /
            |  /
            |---------> X

        The final NDC coordinates can be obtained by dividing each vector by its w coordinate (perspective division).

        !! NDC matrices depends on the choice of NDC space, and should therefore be chosen accordingly !!
        The ndc matrix is a composition of 2 matrices which define the view frustum:

        .. math::

            ndc &= Ortho \times Depth Scale \\
            &= \begin{bmatrix}
                2. / (r - l) & 0. & 0. & tx \\
                0. & 2. / (t - b) & 0. & ty \\
                0. & 0. & -2. / (\text{far} - \text{near}) & tz \\
                0. & 0. & 0. & 1.
            \end{bmatrix}
            \times \begin{bmatrix}
                1. & 0. & 0. & 0. \\
                0. & 1. & 0. & 0. \\
                0. & 0. & B  & A \\
                0. & 0. & 0. & 1.
            \end{bmatrix} \\
            &= \begin{bmatrix}
                2. / (r - l) & 0. & 0. & -tx \\
                0. & 2. / (t - b) & 0. & -ty \\
                0. & 0. & U & V \\
                0. & 0. & 0. & -1.
            \end{bmatrix}
                
        - **n**, **f**: are the near and far clipping planes,
          which define the min / max depth of the view frustum.
          The near and far planes are also used to normalize the depth values to
          normalized device coordinates.
        - **r**, **l**, **t**, **b**: are the right, left, top and bottom borders of the
          view frustum, and are defined by the perspective
          fov (derived from the focal length) and image plane dimensions.
        - **tx**, **ty**, **tz**: are defined as:

          :math:`tx = -(r + l) / (r - l)`

          :math:`ty = -(t + b) / (t - b)`

          :math:`tz = -(f + n) / (f - n)`
        - **U**, **V**: are elements which define the NDC range.
        - **A**, **B**: can be reverse engineered from U, V and are uniquely defined by them
          (and in fact serve a similar function).

        Input values are determined by the screen dimensions and intrinsic coordinate conventions,
        for example:

            1) :math:`(\text{left}=0, \text{right}=\text{width}, \text{bottom}=\text{height}, \text{top}=0)`
               for origin at top-left of the screen, y axis pointing downwards.
            2) :math:`(\text{left}=-\dfrac{\text{width}}{2}, \text{right}=\dfrac{\text{width}}{2},
               \text{bottom}=-\dfrac{\text{height}}{2}, \text{top}=\dfrac{\text{height}}{2})`
               for origin at center of the screen, and y axis pointing upwards.

        Args:
            left (float): location of the left face of the view-frustum.
            right (float): location of the right face of the view-frustum.
            bottom (float): location of the bottom face of the view-frustum.
            top (float): location of the top face of the view-frustum.
            near (float):
                location of the near face of the view-frustum.
                Should always be larger than zero and smaller than the far clipping plane.
                If used in conjunction with a perspective matrix,
                the near clipping plane should be identical for both.
            far (float):
                location of the near face of the view-frustum.
                Should always be larger than the near clipping plane.
                If used in conjunction with a perspective matrix,
                the far clipping plane should be identical for both.

        Returns:
            (torch.Tensor): the ndc matrix, of shape :math:`(1, 4, 4)`.
        """
        tx = -(right + left) / (right - left)
        ty = -(top + bottom) / (top - bottom)
        # tz = -(far + near) / (far - near)    # Not used explicitly here, but makes easier to follow derivations

        # Some examples of U,V choices to control the depth of the NDC space obtained:
        # ------------------------------------------------------------------------------------------------------
        # | NDC in [-1, 1]  |   U = -2.0 * near * far / (far - near)    | i.e. OpenGL NDC space
        # |                 |   V = -(far + near) / (far - near)        |
        # |                 |                                           |
        # ------------------------------------------------------------------------------------------------------
        # | NDC in [1, 0]   |   U = (near * far) / (far - near)         | Reverse depth for better fp precision
        # |                 |   V = near / (far - near)                 |
        # |                 |                                           |
        # ------------------------------------------------------------------------------------------------------
        # | NDC in [0, 1]   |   U = (near * far) / (near - far)         |
        # |                 |   V = far / (far - near)                  |
        # |                 |                                           |
        # ------------------------------------------------------------------------------------------------------
        # Why? Vectors coming from camera space are first multiplied by the perspective matrix:
        # (they're assumed to be homogeneous, where w = 1)
        #              [focal_x,  0.0,        -x0,     0.0]  @  [ x ]   =   [ ... ]
        #              [0.0,      focal_y,    -y0,     0.0]     [ y ]       [ ... ]
        #              [0.0,      0.0,        0.0,     1.0]     [ z ]       [  1  ]
        #              [0.0,      0.0,        1.0,     0.0]     [ 1 ]       [  z  ]
        #
        # and next we convert them to clip space by using the matrix calculated below:
        #              [2.0 / (r - l),  0.0,             0.0,      -tx  ]   @   [ ... ]   =  [    ..    ]
        #              [0.0,            2.0 / (t - b),   0.0,      -ty  ]       [ ... ]      [    ..    ]
        #              [0.0,            0.0,             U,        V    ]       [  1  ]      [  U + Vz  ]
        #              [0.0,            0.0,             0.0,      -1.0 ]       [  z  ]      [    -z    ]
        #
        # the last step to move from clip space to ndc space involves perspective division: we divide by w.
        #              [    ..    ]    / (-z)       [    ..    ]
        #              [    ..    ]   =========>    [    ..    ]
        #              [  U + Vz  ]   persp. div    [ -U/z - V ]
        #              [    -z    ]                 [     1    ]
        #
        # And we obtain:
        #              z_ndc = -U / z - V
        #
        # We want to map specific values of z, the near and far planes, to specific NDC values
        # (for example, such that near --> -1,  far --> 1 ).
        #
        # kaolin assumes a left handed NDC space (depth goes inwards the screen), so we substitute
        # z = -near and z = -far in the equation above, paired with the requested z_ndc values.
        # A simple linear system of 2 equations is obtained, that once solved, yields U and V.
        #              -1 = -U / (-n) - V
        #               1 = -U / (-f) - V
        if self.ndc_min == -1 and self.ndc_max == 1:
            U = -2.0 * near * far / (far - near)
            V = -(far + near) / (far - near)
        elif self.ndc_min == 0 and self.ndc_max == 1:
            U = (near * far) / (near - far)
            V = far / (far - near)
        elif self.ndc_min == 1 and self.ndc_max == 0:
            U = (near * far) / (far - near)
            V = near / (far - near)
        else:
            raise NotImplementedError('Perspective Projection does not support NDC range of '
                                      f'[{self.ndc_min}, {self.ndc_max}]')

        # The matrix is non differentiable, as NDC coordinates are a fixed standard set by the graphics api
        ndc_mat = self.params.new_tensor([
            [2.0 / (right - left),  0.0,                   0.0,            -tx ],
            [0.0,                   2.0 / (top - bottom),  0.0,            -ty ],
            [0.0,                   0.0,                   U,               V  ],
            [0.0,                   0.0,                   0.0,            -1.0]
        ], dtype=self.dtype)

        # Add batch dim, to allow broadcasting
        return ndc_mat.unsqueeze(0)

    def projection_matrix(self) -> torch.Tensor:
        r"""Creates an OpenGL compatible perspective projection matrix to clip coordinates.
        This is the default perspective projection matrix used by kaolin: it assumes the NDC origin is at the
        center of the canvas (hence x0, y0 offsets are measured relative to the center).
        
        Return:
            (torch.Tensor): the projection matrix, of shape :math:`(\text{num_cameras}, 4, 4)`
        """
        # Obtain perspective projection matrix to non-NDC coordinates
        # The axis-skew is assumed to be negligible (m01 of the matrix is zero)
        persp_matrix = self.perspective_matrix()

        # Compute view frustum components, for conversion to clip / NDC coordinates.
        # By default, kaolin follows OpenGL conventions of NDC in [-1, 1],
        # where the center of the canvas is denoted as (0, 0)
        # The following lines ensure the projection matrix is compatible with OpenGL.
        # Practitioners using a different graphics api may modify this matrix.
        top = self.height / 2
        bottom = -top
        right = self.width / 2
        left = -right
        ndc = self.ndc_matrix(left, right, bottom, top, self.near, self.far)

        # Squash matrices together to form complete perspective projection matrix which maps to NDC coordinates
        proj = ndc @ persp_matrix
        return proj

    def project(self, vectors: torch.Tensor) -> torch.Tensor:
        r"""
        Applies perspective projection to obtain Clip Coordinates
        (this function does not perform perspective division the actual Normalized Device Coordinates).

        Assumptions:

        * Camera is looking down the negative "z" axis
          (that is: camera forward axis points outwards from screen, OpenGL compatible).
        * Practitioners are advised to keep near-far gap as narrow as possible,
          to avoid inherent depth precision errors.

        Args:
            vectors (torch.Tensor):
                the vectors to be transformed,
                can homogeneous of shape :math:`(\text{num_vectors}, 4)`
                or :math:`(\text{num_cameras}, \text{num_vectors}, 4)`
                or non-homogeneous of shape :math:`(\text{num_vectors}, 3)`
                or :math:`(\text{num_cameras}, \text{num_vectors}, 3)`

        Returns:
            (torch.Tensor): the transformed vectors, of same shape as ``vectors`` but, with homogeneous coordinates,
                where the last dim is 4
        """
        proj = self.projection_matrix()

        # Expand input vectors to 4D homogeneous coordinates if needed
        homogeneous_vecs = up_to_homogeneous(vectors)

        num_cameras = len(self)  # C - number of cameras
        batch_size = vectors.shape[-2]  # B - number of vectors

        v = homogeneous_vecs.expand(num_cameras, batch_size, 4)[..., None]  # Expand as (C, B, 4, 1)
        proj = proj[:, None].expand(num_cameras, batch_size, 4, 4)  # Expand as (C, B, 4, 4)

        transformed_v = proj @ v
        transformed_v = transformed_v.squeeze(-1)  # Reshape:  (C, B, 4)

        return transformed_v  # Return shape:  (C, B, 4)

    def transform(self, vectors: torch.Tensor) -> torch.Tensor:
        r"""
        Applies perspective projection to obtain Normalized Device Coordinates
        (this function also performs perspective division).

        Assumptions:

        * Camera is looking down the negative z axis (that is: Z axis points outwards from screen, OpenGL compatible).
        * Practitioners are advised to keep near-far gap as narrow as possible,
          to avoid inherent depth precision errors.

        Args:
            vectors (torch.Tensor):
                the vectors to be transformed,
                can homogeneous of shape :math:`(\text{num_vectors}, 4)`
                or :math:`(\text{num_cameras}, \text{num_vectors}, 4)`
                or non-homogeneous of shape :math:`(\text{num_vectors}, 3)`
                or :math:`(\text{num_cameras}, \text{num_vectors}, 3)`

        Returns:
            (torch.Tensor): the transformed vectors, of same shape as ``vectors`` but with non-homogeneous coords,
            e.g. the last dim 3
        """
        transformed_v = self.project(vectors)   # Project with homogeneous coords to shape (C, B, 4)
        normalized_v = down_from_homogeneous(transformed_v)  # Perspective divide to shape:  (C, B, 3)
        return normalized_v  # Return shape:  (C, B, 3)

    def normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        r"""Normalizes depth values to the NDC space defined by the view frustum.

        Args:
            depth (torch.Tensor):
                the depths to be normalized, of shape :math:`(\text{num_depths},)`
                or :math:`(\text{num_cameras}, \text{num_depths})`

        Returns:
            (torch.Tensor):
                The normalized depth values to the ndc range defined by the projection matrix,
                of shape :math:`(\text{num_cameras}, \text{num_depths})`
        """
        if depth.ndim < 2:
            depth = depth.expand(len(self), *depth.shape)
        proj = self.projection_matrix()
        a = -proj[:, 2, 2]
        b = -proj[:, 2, 3]
        depth = torch.clamp(depth, min=min(self.near, self.far), max=max(self.near, self.far))
        # Here we allow depth to be 0, as it will result in 'inf' values which torch will soon clamp.
        # If b is 0 as well, it most likely means the choice of near / far planes and ndc coordinates is invalid.
        ndc_depth = a - b / depth                   # from near: ndc_min to far: ndc_nax
        ndc_min = min(self.ndc_min, self.ndc_max)
        ndc_max = max(self.ndc_min, self.ndc_max)
        normalized_depth = (ndc_depth - ndc_min) / (ndc_max - ndc_min)  # from near: 0 to far: 1
        normalized_depth = torch.clamp(normalized_depth, min=0.0, max=1.0)
        return normalized_depth

    @CameraIntrinsics.width.setter
    def width(self, value: int) -> None:
        """ Updates the width of the image plane.
        The fov will remain invariant, and the focal length may change instead.
        """
        # Keep the fov invariant and change focal length instead
        fov = self.fov_x
        self._shared_fields['width'] = value
        self.fov_x = fov

    @CameraIntrinsics.height.setter
    def height(self, value: int) -> None:
        """ Updates the hieght of the image plane.
        The fov will remain invariant, and the focal length may change instead.
        """
        # Keep the fov invariant and change focal length instead
        fov = self.fov_y
        self._shared_fields['height'] = value
        self.fov_y = fov

    @property
    def x0(self) -> torch.FloatTensor:
        """The horizontal offset from the NDC origin in image space
        By default, kaolin defines the NDC origin at the canvas center.
        """
        return self.params[:, PinholeParamsDefEnum.x0]

    @x0.setter
    def x0(self, val: Union[float, torch.Tensor]) -> None:
        self._set_param(val, PinholeParamsDefEnum.x0)

    @property
    def y0(self) -> torch.FloatTensor:
        """The vertical offset from the NDC origin in image space
        By default, kaolin defines the NDC origin at the canvas center.
        """
        return self.params[:, PinholeParamsDefEnum.y0]

    @y0.setter
    def y0(self, val: Union[float, torch.Tensor]) -> None:
        self._set_param(val, PinholeParamsDefEnum.y0)

    @property
    def cx(self) -> torch.FloatTensor:
        """The principal point X coordinate.
        Note: By default, the principal point is canvas center (kaolin defines the NDC origin at the canvas center).
        """
        # Assumes the NDC x origin is at the center of the canvas
        return self.width / 2.0 + self.params[:, PinholeParamsDefEnum.x0]

    @property
    def cy(self) -> torch.FloatTensor:
        """The principal point Y coordinate.
        Note: By default, the principal point is canvas center (kaolin defines the NDC origin at the canvas center).
        """
        # Assumes the NDC y origin is at the center of the canvas
        return self.height / 2.0 + self.params[:, PinholeParamsDefEnum.y0]

    @property
    def focal_x(self) -> torch.FloatTensor:
        return self.params[:, PinholeParamsDefEnum.focal_x]

    @focal_x.setter
    def focal_x(self, val: Union[float, torch.Tensor]) -> None:
        self._set_param(val, PinholeParamsDefEnum.focal_x)

    @property
    def focal_y(self) -> torch.FloatTensor:
        return self.params[:, PinholeParamsDefEnum.focal_y]

    @focal_y.setter
    def focal_y(self, val: Union[float, torch.Tensor]) -> None:
        self._set_param(val, PinholeParamsDefEnum.focal_y)

    def tan_half_fov(self, camera_fov_direction: CameraFOV = CameraFOV.VERTICAL) -> torch.FloatTensor:
        r"""tan(fov/2) in radians

        Args:
            camera_fov_direction (optional, CameraFOV):
                the leading direction of the fov. Default: vertical

        Returns:
            (torch.Tensor): tan(fov/2) in radians, of size :math:`(\text{num_cameras},)`
        """
        if camera_fov_direction is CameraFOV.HORIZONTAL:
            tanHalfAngle = self.focal_x.new_tensor([self.width / 2.0]) / self.focal_x
        elif camera_fov_direction is CameraFOV.VERTICAL:
            tanHalfAngle = self.focal_y.new_tensor([self.height / 2.0]) / self.focal_y
        else:
            raise ValueError(f'Unsupported CameraFOV direction enum given to tan_half_fov: {camera_fov_direction}')
        return tanHalfAngle

    def fov(self, camera_fov_direction: CameraFOV = CameraFOV.VERTICAL, in_degrees=True) -> torch.FloatTensor:
        r"""The field-of-view

        Args:
            camera_fov_direction (CameraFOV):
                the leading direction of the fov. Default: vertical
            in_degrees (bool): if True return result in degrees, else in radians. Default: True

        Returns:
            (torch.Tensor): the field-of-view, of shape :math:`(\text{num_cameras},)`
        """
        if camera_fov_direction is CameraFOV.HORIZONTAL:
            x, y = self.focal_x, self.width / 2.0
        elif camera_fov_direction is CameraFOV.VERTICAL:
            x, y = self.focal_y, self.height / 2.0
        y = x.new_tensor(y)
        fov = 2 * torch.atan2(y, x)
        if in_degrees:
            fov = fov * 180 / np.pi
        return fov

    @property
    def fov_x(self):
        """The field-of-view on horizontal leading direction"""
        return self.fov(CameraFOV.HORIZONTAL, in_degrees=True)

    @fov_x.setter
    def fov_x(self, angle_degs: Union[float, torch.Tensor]) -> None:
        if isinstance(angle_degs, torch.Tensor):
            angle_degs = angle_degs.to(device=self.focal_x.device, dtype=self.focal_x.dtype)
        else:
            angle_degs = self.focal_x.new_tensor(angle_degs)
        fov = angle_degs / 180 * np.pi
        tanHalfAngle = torch.tan(fov / 2.0)
        aspectScale = self.width / 2.0
        self.focal_x = aspectScale / tanHalfAngle

    @property
    def fov_y(self):
        """The field-of-view on vertical leading direction"""
        return self.fov(CameraFOV.VERTICAL, in_degrees=True)

    @fov_y.setter
    def fov_y(self, angle_degs: Union[float, torch.Tensor]) -> None:
        if isinstance(angle_degs, torch.Tensor):
            angle_degs = angle_degs.to(device=self.focal_y.device, dtype=self.focal_y.dtype)
        else:
            angle_degs = self.focal_y.new_tensor(angle_degs)
        fov = angle_degs / 180 * np.pi
        tanHalfAngle = torch.tan(fov / 2.0)
        aspectScale = self.height / 2.0
        self.focal_y = aspectScale / tanHalfAngle

    def zoom(self, amount):
        r"""Applies a zoom on the camera by adjusting the lens.

        Args:
            amount (torch.Tensor or float):
                Amount of adjustment, measured in degrees.
                Mind the conventions -
                To zoom in, give a positive amount (decrease fov by amount -> increase focal length)
                To zoom out, give a negative amount (increase fov by amount -> decrease focal length)
        """
        fov_ratio = self.fov_x / self.fov_y
        self.fov_y -= amount
        self.fov_x = self.fov_y * fov_ratio  # Make sure the view is not distorted
