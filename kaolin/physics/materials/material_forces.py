# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import torch

from kaolin.physics.materials import neohookean_elastic_material
from kaolin.physics.materials import linear_elastic_material
from kaolin.physics.materials import muscle_material
import kaolin.physics.materials.utils as material_utils
import kaolin.physics.utils as physics_utils

__all__ = [
    'NeohookeanMaterial',
    'MuscleMaterial',
]


class NeohookeanMaterial(physics_utils.ForceWrapper):
    r"""Wrapper for kaolin.physics.material.neohookean_elastic_material's energy, gradient hessian functions. Initialize with youngs modulus (stiffness value measure in Pascals) and poisson ratios (a measure of compressibility, unitless). 
    For more background information, refer to `Ted Kim's Siggraph Course Notes\
    <https://www.tkim.graphics/DYNAMIC_DEFORMABLES/>`_
    """

    def __init__(self, yms, prs):
        r"""Initializer

        Args:
            yms (torch.Tensor): Tensor of youngs modulus per-primitive, of shape :math:`(\text{batch_dim}, 1)`
            prs (_type_): Tensor of poisson ratio per primitive, of shape :math:`(\text{batch_dim}, 1)`
        """
        self.mus, self.lams = material_utils.to_lame(yms, prs)

    def _energy(self, defo_grad):
        r"""Energy wrapper

        Args:
            defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

        Returns:
            torch.Tensor: Tensor of per-element energies, of shape :math:`(\text{batch_dim}, 1)`
        """
        return neohookean_elastic_material.unbatched_neohookean_energy(self.mus, self.lams, defo_grad)

    def _gradient(self, defo_grad):
        r"""Gradient wrapper

        Args:
            defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

        Returns:
            torch.Tensor: Tensor of per-element gradients, of shape :math:`(\text{batch_dim}, 9)`
        """
        return neohookean_elastic_material.unbatched_neohookean_gradient(self.mus, self.lams, defo_grad)

    def _hessian(self, defo_grad):
        r"""Hessian wrapper

        Args:
            defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

        Returns:
            torch.Tensor: Tensor of per-element hessians, of shape :math:`(\text{batch_dim}, 9,9)`
        """
        return neohookean_elastic_material.unbatched_neohookean_hessian(self.mus, self.lams, defo_grad)


class MuscleMaterial(physics_utils.ForceWrapper):
    r"""Wrapper for kaolin.physics.material.muscle_material's energy, gradient hessian functions. Initialize with fiber vectors of shape :math:`(\text{n}, 3)` for n integration points.
    """

    def __init__(self, fiber_vecs):
        r"""Wrapper around muscle energy, gradients, hessian functions. Stores fibers and activation.

        Args:
            fiber_vecs (torch.Tensor): Matrix of per-primitive fiber directions, of shape :math:`(\text{batch_dim}, 3)`
        """
        self.fiber_vecs = fiber_vecs
        self.activation = 0
        self.fiber_mat_blocks = muscle_material.precompute_fiber_matrix(self.fiber_vecs)

    def set_activation(self, a):
        r"""Sets muscle activation.

        Args:
            a (float): Activation amount
        """
        self.activation = a

    def _energy(self, defo_grad):
        r"""Energy wrapper

        Args:
            defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

        Returns:
            torch.Tensor: Tensor of per-element energies, of shape :math:`(\text{batch_dim}, 1)`
        """
        return muscle_material.unbatched_muscle_energy(self.activation, self.fiber_mat_blocks, defo_grad)

    def _gradient(self, defo_grad):
        r"""Gradient wrapper

        Args:
            defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

        Returns:
            torch.Tensor: Tensor of per-element gradients, of shape :math:`(\text{batch_dim}, 9)`
        """
        return muscle_material.unbatched_muscle_gradient(self.activation, self.fiber_mat_blocks, defo_grad)

    def _hessian(self, defo_grad):
        r"""Hessian wrapper

        Args:
            defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

        Returns:
            torch.Tensor: Tensor of per-element hessians, of shape :math:`(\text{batch_dim}, 9,9)`
        """
        return muscle_material.unbatched_muscle_hessian(self.activation, self.fiber_mat_blocks, defo_grad)
