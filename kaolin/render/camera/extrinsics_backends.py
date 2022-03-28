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
from typing import Type, Union
from enum import IntEnum
import warnings
import torch


_REGISTERED_BACKENDS = dict()       # available extrinsics representation backends

def register_backend(backend_class: Type[ExtrinsicsRep]):
    """Registers a representation backend class with a unique name.
    CameraExtrinsics can switch between registered representations dynamically (see switch_backend()).
    """
    _REGISTERED_BACKENDS[backend_class.backend_name()] = backend_class
    return backend_class


class ExtrinsicsParamsDefEnum(IntEnum):
    R = 0   # Orientation of the camera axes (directional axes of camera forward, camera right, camera up)
    t = 1   # Position of camera center


class ExtrinsicsRep(ABC):
    """
    An abstract class for representing CameraExtrinsics representation backends.
    This class keeps the separation between the parameter representation space and the associated rigid
    transformation (usually represented as a view matrix) separate.

    Different representations are tuned to varied use cases: speed, differentiability w.r.t rigid transformations space,
    and so forth.
    """

    def __init__(self, params: torch.Tensor,
                 dtype: torch.dtype = None,
                 device: Union[torch.device, str] = None,
                 requires_grad: bool = False):
        self.params = params
        if device is not None:
            self.params = self.params.to(device=device, dtype=dtype)
        elif dtype is not None:
            self.params = self.params.to(dtype=dtype)

        # If the params tensor already has a gradient computation graph
        if self.params.grad_fn is not None:
            if not requires_grad:   # if the requires_grad arg is true, do nothing, we're set
                # Otherwise, if explicitly requested not to have grads,
                # then detach computation graph to create a separate tensor
                self.params = self.params.detach()
                self.params.requires_grad = requires_grad
        else:
            # No computation graph was generated so requires_grad can be safely set
            self.params.requires_grad = requires_grad

    @classmethod
    def from_mat(cls, mat: torch.Tensor, dtype: torch.dtype = None, device: Union[torch.device, str] = None,
                 requires_grad: bool = False):
        """ Constructs backend from given (C, 4, 4) view matrix. """
        params = cls.convert_from_mat(mat)
        return cls(params, dtype, device, requires_grad)

    def update(self, mat: torch.Tensor):
        """ Updates the underlying representation by mapping the 4x4 view matrix to representation space """
        self.params = self.convert_from_mat(mat)

    @abstractmethod
    def convert_to_mat(self) -> torch.Tensor:
        """ Converts the underlying representation to view-matrix form of shape (C, 4, 4) """
        pass

    @classmethod
    @abstractmethod
    def convert_from_mat(cls, mat: torch.Tensor) -> torch.Tensor:
        """ Converts a view-matrix to the underlying representation form of shape (C, K) where K is the number
            of representation parameters.
        """
        pass

    @classmethod
    @abstractmethod
    def param_idx(cls, param: ExtrinsicsParamsDefEnum):
        """ Returns the indices of elements in the 'self.params' field of instances of this class, which are belong
            under the ExtrinsicsParamsDefEnum param argument.
            i.e: For ExtrinsicsParamsDefEnum.R, a 4x4 matrix representation will return the 9 indices of
            the camera axes R component.
        """
        pass

    def __getitem__(self, item):
        """ :return torch.Tensor of shape (M, 4, 4) representing a single camera's extrinsics from this batched object
        """
        params = self.params[item]
        if params.ndim < self.params.ndim:
            params = params.unsqueeze(0)
        entry = type(self)(params, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        return entry

    def to(self, *args, **kwargs):
        """ Cast to a different device / dtype """
        converted_params = self.params.to(*args, **kwargs)
        if self.params.device == converted_params.device and self.params.dtype == converted_params.dtype:
            return self
        else:
            return type(self)(converted_params)

    @property
    def device(self) -> torch.device:
        """ :return the torch device of parameters tensor """
        return self.params.device

    @property
    def dtype(self) -> torch.dtype:
        return self.params.dtype

    @property
    def requires_grad(self) -> bool:
        return self.params.requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self.params.requires_grad = val

    def __len__(self) -> int:
        return self.params.shape[0]

    @classmethod
    @abstractmethod
    def backend_name(cls) -> str:
        pass


@register_backend
class _MatrixSE3Rep(ExtrinsicsRep):
    """ 4x4 matrix form of rigid transformations from SE(3), the special Euclidean group.
        Uses the identity mapping from representation space to transformation space,
        and thus simple and quick for non-differentiable camera operations.
        However, without additional constraints, the over-parameterized nature of this representation
        makes it unsuitable for optimization (e.g: transformations are not guaranteed to remain in SE(3)
        during backpropagation).
    """
    def __init__(self, params: torch.Tensor,
                 dtype: torch.dtype = None,
                 device: Union[torch.device, str] = None,
                 requires_grad: bool = False):
        super().__init__(params, dtype, device, requires_grad)
        if requires_grad:
            self.prompt_differentiability_warning()

    def convert_to_mat(self):
        return self.params.clone().reshape(-1, 4, 4)

    @classmethod
    def convert_from_mat(cls, mat: torch.Tensor):
        return mat.reshape(-1, 16)

    @classmethod
    def param_idx(cls, param: ExtrinsicsParamsDefEnum):
        """ Returns the indices of elements in the 'self.params' field of instances of this class, which are belong
            under the ExtrinsicsParamsDefEnum param argument.
            i.e: For ExtrinsicsParamsDefEnum.R, a 4x4 matrix representation will return the 9 indices of
            the camera axes R component.
        """
        if param == ExtrinsicsParamsDefEnum.R:
            return [0, 1, 2, 4, 5, 6, 8, 9, 10]
        elif param == ExtrinsicsParamsDefEnum.t:
            return [3, 7, 11]

    @property
    def requires_grad(self):
        return super(_MatrixSE3Rep, self).requires_grad

    @requires_grad.setter
    def requires_grad(self, val):
        if val:
            self.prompt_differentiability_warning()
        super(_MatrixSE3Rep, self.__class__).requires_grad.fset(self, val)

    @staticmethod
    def prompt_differentiability_warning():
        warnings.warn('matrix_se3 representations can converge to non-rigid transformations due to '
                      'differentiable operations. Either explicitly enforce constraints on the learned '
                      'matrix parameters, or switch to a different backend representation.')

    @classmethod
    def backend_name(cls) -> str:
        return "matrix_se3"


@register_backend
class _Matrix6DofRotationRep(ExtrinsicsRep):
    """ A representation space which supports differentiability in the space of rigid transformations.
    That is, the view-matrix is guaranteed to represent a valid rigid transformation.
    Under the hood, this representation keeps 6 DoF for rotation, and 3 additional ones for translation.
    For conversion to view-matrix form, a single Gram–Schmidt step is required.
    See: On the Continuity of Rotation Representations in Neural Networks, Zhou et al. 2019
    """
    def __init__(self, params: torch.Tensor, dtype: torch.dtype = None, device: Union[torch.device, str] = None,
                 requires_grad: bool = False):
        super().__init__(params, dtype, device, requires_grad)

    def convert_to_mat(self) -> torch.Tensor:
        batch_size = self.params.shape[0]
        a1 = self.params[:, 0:3]
        a2 = self.params[:, 3:6]
        translation = self.params[:, 6:9]

        b1 = torch.nn.functional.normalize(a1, dim=1)
        b1_dot_a2 = torch.bmm(b1.view(-1, 1, 3), a2.view(-1, 3, 1)).view(batch_size, 1)    # Batched dot product
        b2 = torch.nn.functional.normalize(a2 - b1_dot_a2 * b1, dim=1)
        b3 = torch.cross(b1, b2)

        rotation = torch.stack([b1, b2, b3], dim=1)  # Stack row-wise
        extrinsics_mat = torch.cat([rotation, translation.unsqueeze(-1)], dim=2)  # Stack column-wise
        homogeneous_row = translation.new_tensor([[0.0, 0.0, 0.0, 1.0]]).unsqueeze(0).expand(batch_size, 1, 4)
        mat = torch.cat([extrinsics_mat, homogeneous_row], dim=1)
        return mat

    @classmethod
    def convert_from_mat(cls, mat: torch.Tensor):
        batch_dim = mat.shape[0]

        # Select first 2 column vectors of rotation sub-matrix
        # [r1,   r2,   r3,     Right
        #  u1,   u2,   u3,     Up
        #  -b1,  -b2,  -b3]    Forward
        rotation = mat[:, :2, :3]

        # Translation vector
        translation = mat[:, :3, -1:]

        # Representation is 6DoF for rotation + 3DoF for translation: (r1, r2, r3, u1, u2, u3, tx, ty, tz)
        params = torch.cat((rotation.reshape(batch_dim, -1), translation.reshape(batch_dim, -1)), dim=1)
        return params

    @classmethod
    def param_idx(cls, param: ExtrinsicsParamsDefEnum):
        """ Returns the indices of elements in the 'self.params' field of instances of this class, which are belong
            under the ExtrinsicsParamsDefEnum param argument.
            i.e: For ExtrinsicsParamsDefEnum.R, a 4x4 matrix representation will return the 9 indices of
            the camera axes R component.
        """
        if param == ExtrinsicsParamsDefEnum.R:
            return list(range(0, 6))
        elif param == ExtrinsicsParamsDefEnum.t:
            return list(range(6, 9))

    @classmethod
    def backend_name(cls) -> str:
        return "matrix_6dof_rotation"
