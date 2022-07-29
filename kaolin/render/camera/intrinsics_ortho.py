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
from typing import Type, Union
import torch
from .intrinsics import CameraIntrinsics, IntrinsicsParamsDefEnum,\
    up_to_homogeneous, down_from_homogeneous, default_dtype

__all__ = [
    "OrthographicIntrinsics"
]

class OrthoParamsDefEnum(IntrinsicsParamsDefEnum):
    """Orthographic projections do not use real intrinsics.
    However since for this type of projection all objects appear at the same
    distance to the camera, a scale factor is included with the intrinsics, to allow
    for "zoom" adjustments.
    """
    fov_distance = 0   # Zoom factor, to adjust the scale of the view. Measured in distance units.


class OrthographicIntrinsics(CameraIntrinsics):
    """Holds the intrinsics parameters of a theoretical orthographic camera:
    how it should project from camera space to normalized screen / clip space.
    This is the most simplistic form of a camera projection model which does not distort objects at the distance.

    See documentation of CameraIntrinsics for numerous ways of how to use this class.

    The orthographic camera explicitly exposes the orthographic transformation matrix.
    This may typically be useful for rasterization based rendering pipelines (i.e: OpenGL).

    In general, intrinsic classes keep a batched tensor of parameters.
    However, for orthographic projections there are no parameters to keep, and therefore the params tensor is empty.

    The matrix returned by this class supports differentiable torch operations.
    """
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

                an enum describing each of the intrinsic parameters managed by the orthographic camera.
                This enum also defines the order in which values are kept within the params buffer.
        """
        return OrthoParamsDefEnum

    @property
    def lens_type(self) -> str:
        return 'ortho'

    @classmethod
    def from_frustum(cls, width: int, height: int, fov_distance: float = 1.0,
                     near: float = DEFAULT_NEAR, far: float = DEFAULT_FAR,
                     num_cameras: int = 1,
                     device: Union[torch.device, str] = None,
                     dtype: torch.dtype = default_dtype) -> OrthographicIntrinsics:
        """Constructs a new instance of OrthographicIntrinsics from view frustum dimensions

        fov_distance artificially defines the "zoom scale" of the view.

        Args:
             width (int): width of the camera resolution
             height (int): height of the camera resolution
             fov_distance (optiona, float): the field of view distance. Default: 1.0
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
            (OrthographicIntrinsics): the constructed orthographic camera intrinsics
        """
        params = cls._allocate_params(fov_distance, num_cameras=num_cameras, device=device, dtype=dtype)
        return OrthographicIntrinsics(width, height, params, near, far)

    def orthographic_matrix(self, left, right, bottom, top, near, far) -> torch.Tensor:
        r"""Constructs a matrix which normalizes a cuboid-shaped view-frustum to normalized device coordinates (NDC).
        Orthographic projections do not consider perspectives.
        This method is similar in behaviour to the now deprecated OpenGL function :func:`glOrtho()`.

        Input values are determined by the screen dimensions and intrinsic coordinate conventions, for example:

            1) :math:`(\text{left}=0, \text{right}=\text{width}, \text{bottom}=\text{height}, \text{top}=0)`
               for origin at top-left of the screen, y axis pointing downwards
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
            (torch.Tensor): the orthographic matrix, of shape :math:`(1, 4, 4)`.
        """
        zero = torch.zeros_like(self.fov_distance)
        one = torch.ones_like(self.fov_distance)
        tx = torch.full_like(self.fov_distance, fill_value=-(right + left) / (right - left))
        ty = torch.full_like(self.fov_distance, fill_value=-(top + bottom) / (top - bottom))
        tz = torch.full_like(self.fov_distance, fill_value=-(far + near) / (far - near))
        W = right - left
        H = top - bottom
        D = torch.full_like(self.fov_distance, fill_value=far - near)
        fov = self.fov_distance

        # The ortho matrix is defined as:
        #     [2.0 / (fov * W),  0.0,              0.0,        tx],
        #     [0.0,              2.0 / (fov * H),  0.0,        ty],
        #     [0.0,              0.0,              -2.0 / D,   tz],
        #     [0.0,              0.0,              0,          1.0]
        rows = [
            torch.stack([2.0 / (fov * W),  zero,             zero,       tx],       dim=-1),
            torch.stack([zero,             2.0 / (fov * H),  zero,       ty],       dim=-1),
            torch.stack([zero,             zero,             -2.0 / D,   tz],       dim=-1),
            torch.stack([zero,             zero,             zero,       one],     dim=-1)
        ]
        ortho_mat = torch.stack(rows, dim=1)
        return ortho_mat

    def projection_matrix(self) -> torch.Tensor:
        r"""Creates an OpenGL compatible orthographic projection matrix to NDC coordinates.

        Return:
            (torch.Tensor): the projection matrix, of shape :math:`(\text{num_cameras}, 4, 4)`
        """
        # Compute view frustum components, for conversion to NDC coordinates.
        # kaolin follows OpenGL conventions of NDC in [-1, 1], where the center of the canvas is denoted as (0, 0)
        # The following lines ensure the projection matrix is compatible with OpenGL.
        # Practitioners using a different graphics api may modify this matrix.
        # top = self.height / 2
        # bottom = -top
        # right = self.width / 2
        # left = -right
        top = 1.0
        bottom = -top
        right = 1.0 * self.width / self.height
        left = -right
        ortho = self.orthographic_matrix(left, right, bottom, top, self.near, self.far)
        return ortho

    def transform(self, vectors: torch.Tensor) -> torch.Tensor:
        r"""Apply perspective projection to NDC coordinates.

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
            (torch.Tensor): the transformed vectors, of same shape than ``vectors`` but last dim 3
        """
        proj = self.projection_matrix()

        # Expand input vectors to 4D homogeneous coordinates if needed
        homogeneous_vecs = up_to_homogeneous(vectors)

        num_cameras = len(self)         # C - number of cameras
        batch_size = vectors.shape[-2]  # B - number of vectors

        v = homogeneous_vecs.expand(num_cameras, batch_size, 4)[..., None]  # Expand as (C, B, 4, 1)
        proj = proj[:, None].expand(num_cameras, batch_size, 4, 4)          # Expand as (C, B, 4, 4)

        transformed_v = proj @ v
        transformed_v = transformed_v.squeeze(-1)  # Reshape:  (C, B, 4)
        normalized_v = down_from_homogeneous(transformed_v)

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

    @property
    def fov_distance(self) -> torch.FloatTensor:
        return self.params[:, OrthoParamsDefEnum.fov_distance]

    @fov_distance.setter
    def fov_distance(self, val: Union[float, torch.Tensor]) -> None:
        self._set_param(val, OrthoParamsDefEnum.fov_distance)

    def zoom(self, amount):
        self.fov_distance += amount
        self.fov_distance = torch.max(self.fov_distance, self.fov_distance.new_tensor(1e-5))    # Don't go below eps
