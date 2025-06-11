# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import warp as wp
import warp.sparse as wps
import torch
from .material_utils import kron3
from kaolin.physics.utils.warp_utilities import mat99, vec9

__all__ = [
    'wp_neohookean_energy',
    'unbatched_neohookean_energy',
    'unbatched_neohookean_gradient',
    'unbatched_neohookean_hessian',
    'NeohookeanElasticMaterial'
]


@wp.func
def neohookean_elastic_energy(mu: wp.float32, lam: wp.float32, F: wp.mat33, vol: wp.float32) -> wp.float32:
    r"""Implements a version of neohookean energy. Calculate energy per-integration primitive. For more background information, refer to `Ted Kim's Siggraph Course Notes\
    <https://www.tkim.graphics/DYNAMIC_DEFORMABLES/>`_

    Args:
        mu (wp.float32): Lame coefficient mu
        lam (wp.float32): Lame coefficient lambda
        F (wp.mat33): Deformation gradient
        vol (wp.float32): Volume

    Returns:
        wp.float32: Neohookean elastic energy
    """
    C1 = mu / 2.0
    D1 = lam / 2.0
    F_transpose = wp.transpose(F)
    I1 = wp.trace(F_transpose @ F)  # TODO: I1 = wp.ddot(F, F)
    J = wp.determinant(F)
    W = C1 * (I1 - 3.0) + D1 * (J - 1.0) * (J - 1.0) - mu * (J - 1.0)
    return W*vol


@wp.func
def neohookean_elastic_gradient(mu: wp.float32, lam: wp.float32, F: wp.mat33, vol: wp.float32) -> wp.mat33:
    r"""Implements a version of neohookean gradient. Calculate gradient per-integration primitive. For more background information, refer to `Ted Kim's Siggraph Course Notes\
    <https://www.tkim.graphics/DYNAMIC_DEFORMABLES/>`_

    Args:
        mu (wp.float32): Lame coefficient mu
        lam (wp.float32): Lame coefficient lambda
        F (wp.mat33): Deformation gradient
        vol (wp.float32): Volume

    Returns:
        wp.mat33: Neohookean elastic gradient w.r.t. F, :math:`\frac{dE}{dF}`
    """

    C1 = mu / 2.0
    D1 = lam / 2.0
    F_transpose = wp.transpose(F)

    r"""Calculate the first invariant I1 = trace(C) = trace(F^TF)"""
    I1 = wp.trace(F_transpose @ F)

    # Calculate batched determinant of tensor and reshape to (batch_dim, 1)
    J = wp.determinant(F)

    # Energy = (mu/2)I1 - (mu/2)*3
    #  + D1*(J-1)^2
    #  - mu*J + mu
    # Energy = C1 * (I1 - 3.0) + D1 * (J - 1.0) * (J - 1.0) - mu * (J - 1.0)

    r"""Calculate the gradients of Energy w.r.t F"""
    # d (mu/2)*I1 / dF ==> (mu/2) d tr(F^TF)/dF = (mu/2)*2*F
    g1 = mu * F

    # d (lam/2)*(J-1)^2 / dF ==> (lam/2) d (J-1)^2/dF = (lam/2) * 2*(J-1) * dJ/dF ==> (lam/2) * 2*(J-1) * J*F^-T
    g2 = lam * (J - 1.0) * J * wp.transpose(wp.inverse(F))

    # d -mu*J / dF ==> -mu dJ/dF = -mu J F^-T
    g3 = -mu * J * wp.transpose(wp.inverse(F))

    return vol*(g1 + g2 + g3)


@wp.func
def neohookean_elastic_hessian(mu: wp.float32, lam: wp.float32, F: wp.mat33, vol: wp.float32) -> mat99:
    r"""Implements a version of neohookean hessian. Calculate hessian per-integration primitive. For more background information, refer to `Ted Kim's Siggraph Course Notes\
    <https://www.tkim.graphics/DYNAMIC_DEFORMABLES/>`_

    Args:
        mu (wp.float32): Lame coefficient mu
        lam (wp.float32): Lame coefficient lambda
        F (wp.mat33): Deformation gradient
        vol (wp.float32): Volume

    Returns:
        wp.mat99: Neohookean elastic gradient w.r.t. F, :math:`\frac{d^2E}{dF^2}`
    """

    id_mat = wp.identity(9, dtype=wp.float32)

    J = wp.determinant(F)
    Finv = wp.inverse(F)
    FinvT = wp.transpose(wp.inverse(F))
    gamma = J * (lam * (2.0 * J + (-1.0)) + (-mu))
    dgamma = gamma - lam * J * J

    # Reshape FinvT from 3x3 to 9x1 vector
    Finv_flat = wp.vector(Finv[0, 0], Finv[0, 1], Finv[0, 2],
                          Finv[1, 0], Finv[1, 1], Finv[1, 2],
                          Finv[2, 0], Finv[2, 1], Finv[2, 2])

    FinvT_flat = wp.vector(FinvT[0, 0], FinvT[0, 1], FinvT[0, 2],
                           FinvT[1, 0], FinvT[1, 1], FinvT[1, 2],
                           FinvT[2, 0], FinvT[2, 1], FinvT[2, 2])

    gamma = J*(lam*(2.0*J + (-1.0)) + (-mu))
    dgamma = gamma - lam*J*J
    H1 = mu*id_mat
    H2 = gamma*wp.outer(FinvT_flat, FinvT_flat)
    H3 = -dgamma*wp.outer(FinvT_flat, FinvT_flat)
    H = H1+H2+H3
    return vol*H

    # TODO: The following has math bugs

    # # From matrixcookbook
    # dinvFdF = -kron3(FinvT, FinvT)
    # dJdF_x_FinvT = J*wp.outer(FinvT_flat, FinvT_flat)

    # # d mu*F/ dF => mu*I (x) I => flattened => mu * I9
    # H1 = mu * id_mat

    # # d [lam * (J - 1.0) * J * F^-T] / dF
    # # =[lam * d(J - 1.0)/dF * J * F^-T]
    # #  + [lam * (J - 1.0) * dJ/dF * F^-T]
    # #  + [lam * (J - 1.0) * dJ/dF * F^-T]
    # #  + [lam * (J - 1.0) * J * d(F^-T)/dF]
    # H2_1 = lam*J*dJdF_x_FinvT
    # H2_2 = lam*(J-1.0)*dJdF_x_FinvT
    # H2_3 = lam * (J - 1.0) * J * dinvFdF
    # H2 = H2_1 + H2_2 + H2_3

    # # d -mu J F^-1 / dF => -mu dJ/dF * F^-T - mu J d(F^-T)/dF
    # H3 = -mu*dJdF_x_FinvT - mu*J*dinvFdF

    # H = H1 + H2 + H3
    # return vol*H


@wp.kernel
def neohookean_energy_kernel(mus: wp.array(dtype=wp.float32),
                             lams: wp.array(dtype=wp.float32),
                             Fs: wp.array(dtype=wp.mat33),
                             vol: wp.array(dtype=wp.float32),
                             coeff: wp.float32,
                             wp_e: wp.array(dtype=wp.float32)):
    pt_idx = wp.tid()

    mu_ = mus[pt_idx]
    lam_ = lams[pt_idx]

    F_ = Fs[pt_idx]

    # E = nh_energy(F_, lame)
    E = coeff * neohookean_elastic_energy(mu_, lam_, F_, vol[pt_idx])
    wp.atomic_add(wp_e, 0, E)


@wp.kernel
def neohookean_gradient_kernel(mus: wp.array(dtype=wp.float32),
                               lams: wp.array(dtype=wp.float32),
                               Fs: wp.array(dtype=wp.mat33),
                               vol: wp.array(dtype=wp.float32),
                               coeff: wp.float32,
                               dEdF: wp.array(dtype=wp.mat33)):
    pt_idx = wp.tid()

    mu_ = mus[pt_idx]
    lam_ = lams[pt_idx]

    F_ = Fs[pt_idx]

    # grad = -wp.ddot(tau, snh_stress(F_, lame(s)))
    grad = coeff * neohookean_elastic_gradient(mu_, lam_, F_, vol[pt_idx])
    dEdF[pt_idx] += grad


@wp.kernel
def neohookean_hessian_kernel(mus: wp.array(dtype=wp.float32),
                              lams: wp.array(dtype=wp.float32),
                              Fs: wp.array(dtype=wp.mat33),
                              vol: wp.array(dtype=wp.float32),
                              coeff: wp.float32,
                              d2EdF2: wp.array(dtype=mat99)):
    pt_idx = wp.tid()

    mu_ = mus[pt_idx]
    lam_ = lams[pt_idx]

    F_ = Fs[pt_idx]

    hess = coeff * neohookean_elastic_hessian(mu_, lam_, F_, vol[pt_idx])

    d2EdF2[pt_idx] = hess


class NeohookeanElasticMaterial:
    r""" Neohookean elastic material class.
    """

    def __init__(self,
                 mu,
                 lam,
                 integration_pt_volume):
        r""" Initializes a NeohookeanElasticMaterial object.
        Args:
            mu (wp.array(dtype=wp.float32)): Lame coefficient mu
            lam (wp.array(dtype=wp.float32)): Lame coefficient lambda
            integration_pt_volume (wp.array(dtype=wp.float32)): Volume distributed across each point
        """
        self.mu = mu
        self.lam = lam
        self.integration_pt_volume = integration_pt_volume

        # pre-allocated gradients and hessians
        self.gradients = wp.zeros(
            integration_pt_volume.shape[0], dtype=wp.types.vector(9, dtype=wp.float32), device=integration_pt_volume.device)
        self.hessians_blocks = wp.zeros(
            integration_pt_volume.shape[0], dtype=mat99, device=integration_pt_volume.device)

    def __str__(self):
        return "NeohookeanElasticMaterial"

    def energy(self, defo_grads, coeff=1.0, wp_energy=None):
        r""" Returns the neohookean elastic energy at each integration primitive.
        Args:
            defo_grads (wp.array(dtype=wp.mat33)): Deformation gradient
            coeff (float): Coefficient
        Returns:
            wp.array(dtype=wp.float32): Neohookean elastic energy size [1]
        """
        if wp_energy is None:
            wp_energy = wp.zeros(1, dtype=wp.float32, device=defo_grads.device)

        # Launch kernel
        wp.launch(
            kernel=neohookean_energy_kernel,
            dim=defo_grads.shape[0],
            inputs=[self.mu, self.lam, defo_grads,
                    self.integration_pt_volume, coeff],
            outputs=[wp_energy],
            adjoint=False
        )
        return wp_energy

    def gradient(self, defo_grads, coeff=1.0, gradients=None):
        r""" Returns the neohookean elastic gradient at each integration primitive.
        Args:
            defo_grads (wp.array(dtype=wp.mat33)): Deformation gradient
            coeff (float): Coefficient
        Returns:
            wp.array(dtype=wp.mat33): Neohookean elastic gradient
        """
        # Launch kernel
        if gradients is None:
            gradients = wp.zeros_like(defo_grads)
        wp.launch(
            kernel=neohookean_gradient_kernel,
            dim=defo_grads.shape[0],
            inputs=[self.mu, self.lam, defo_grads,
                    self.integration_pt_volume, coeff],
            outputs=[gradients],
            adjoint=False
        )
        return gradients

    def hessian(self, defo_grads, coeff=1.0):
        r""" Returns the neohookean elastic hessian at each integration primitive.
        Args:
            defo_grads (wp.array(dtype=wp.mat33)): Deformation gradient
            coeff (float): Coefficient
        Returns:
            wps.bsr_matrix: Neohookean elastic hessian as a 9x9 block-wise sparse matrix.
        """
        n = defo_grads.shape[0]

        # construct sparse hessians and hessian blocks
        self.hessians_blocks.zero_()

        wp.launch(
            kernel=neohookean_hessian_kernel,
            dim=defo_grads.shape[0],
            inputs=[self.mu, self.lam, defo_grads,
                    self.integration_pt_volume, coeff],
            outputs=[self.hessians_blocks],
            adjoint=False
        )

        return self.hessians_blocks


@wp.func
def wp_neohookean_energy(mu: float, lam: float, F: wp.mat33) -> float:
    r"""Implements a version of neohookean energy. Calculate energy per-integration primitive. For more background information, refer to `Ted Kim's Siggraph Course Notes\
    <https://www.tkim.graphics/DYNAMIC_DEFORMABLES/>`_

    Args:
        mu (torch.Tensor): Batched lame parameter mu, of shape :math:`(\text{batch_dims}, 1)`
        lam (torch.Tensor): Batched lame parameter lambda, of shape, :math:`(\text{batch_dims}, 1)` 
        defo_grad (torch.Tensor): Batched deformation gradients (denoted in literature as F) of 3 or more dimensions, :math:`(\text{batch_dims}, 3, 3)`

    Returns:
        torch.Tensor: :math:`(\text{batch_dims}, 1)` vector of per defo-grad energy values
    """
    C1 = mu / 2.0
    D1 = lam / 2.0
    F_transpose = wp.transpose(F)
    I1 = wp.trace(F_transpose @ F)
    J = wp.determinant(F)
    W = C1 * (I1 - 3.0) + D1 * (J - 1.0) * (J - 1.0) - mu * (J - 1.0)
    return W


def neohookean_energy(mu, lam, defo_grad):
    r"""Implements a version of neohookean energy. Calculate energy per-integration primitive. For more background information, refer to `Ted Kim's Siggraph Course Notes\
    <https://www.tkim.graphics/DYNAMIC_DEFORMABLES/>`_

    Args:
        mu (torch.Tensor): Batched lame parameter mu, of shape :math:`(\text{batch_dims}, 1)`
        lam (torch.Tensor): Batched lame parameter lambda, of shape, :math:`(\text{batch_dims}, 1)` 
        defo_grad (torch.Tensor): Batched deformation gradients (denoted in literature as F) of 3 or more dimensions, :math:`(\text{batch_dims}, 3, 3)`

    Returns:
        torch.Tensor: :math:`(\text{batch_dims}, 1)` vector of per defo-grad energy values
    """
    # Shape (batch_dims, 1)
    C1 = mu / 2
    # Shape (batch_dims, 1)
    D1 = lam / 2

    dimensions = defo_grad.shape
    batched_dims = dimensions[:-2]
    batched_trace = torch.vmap(torch.trace)

    r"""Calculate the first invariant I1 = trace(C) = trace(F^TF)"""

    # Last 2 dimensions of defo_grad (3,3) are the matrix dimensions.
    # All prev dimensions are batch dimensions
    FtF = torch.matmul(torch.transpose(defo_grad, -2, -1), defo_grad)

    # Flatten to (-1, 3, 3)
    cauchy_green_strains = FtF.reshape(batched_dims.numel(), 3, 3)
    # Calculate batched traces of tensor and reshape to (batch_dim, 1)
    I1 = batched_trace(cauchy_green_strains).reshape(
        batched_dims).unsqueeze(-1)

    # Calculate batched determinant of tensor and reshape to (batch_dim, 1)
    J = torch.det(defo_grad).unsqueeze(-1)
    W1 = C1 * (I1 - 3)
    W2 = D1 * (J - 1) * (J - 1)
    W3 = - mu * (J - 1.0)
    W = W1 + W2 + W3
    return W


def neohookean_gradient(mu, lam, defo_grad):
    """Implements a batched version of the jacobian of neohookean elastic energy. Calculates gradients per-integration primitive. For more background information, refer to `Jernej Barbic's Siggraph Course Notes\
    <https://viterbi-web.usc.edu/~jbarbic/femdefo/sifakis-courseNotes-TheoryAndDiscretization.pdf>`_ section 3.2.

    Args:
        mu (torch.Tensor): Batched lame parameter mu, of shape :math:`(\text{batch_dim}, 1)`
        lam (torch.Tensor): Batched lame parameter lambda, of shape :math:`(\text{batch_dim}, 1)`
        defo_grad (torch.Tensor): Batched deformation gradients (denoted in literature as F) of any dimension where the last 2 dimensions are 3 x 3, of shape :math:`(\text{batch_dim}, 3, 3)`

    Returns:
        torch.Tensor: Vector of per-primitive jacobians of neohookean elastic energy w.r.t defo_grad values, of shape :math:`(\text{batch_dim}, 9)`
    """
    # Shape (batch_dims, 1)
    C1 = mu / 2
    # Shape (batch_dims, 1)
    D1 = lam / 2

    dimensions = defo_grad.shape
    batched_dims = dimensions[:-2]
    batched_trace = torch.vmap(torch.trace)

    r"""Calculate the first invariant I1 = trace(C) = trace(F^TF)"""
    # Last 2 dimensions of defo_grad (3,3) are the matrix dimensions.
    FtF = torch.matmul(torch.transpose(defo_grad, -2, -1), defo_grad)
    cauchy_green_strains = FtF.reshape(batched_dims.numel(), 3, 3)
    # Calculate batched traces of tensor and reshape to (batch_dim, 1)
    I1 = batched_trace(cauchy_green_strains).reshape(
        batched_dims).unsqueeze(-1)

    # Calculate batched determinant of tensor and reshape to (batch_dim, 1)
    J = torch.det(defo_grad).unsqueeze(-1)
    Energy = C1 * (I1 - 3) + D1 * (J - 1) * (J - 1) - mu * (J - 1.0)

    # Energy = (mu/2)I1 - (mu/2)*3
    #  + D1*(J-1)^2
    #  - mu*J + mu

    r"""Calculate the gradients of Energy w.r.t F"""
    # d (mu/2)*I1 / dF ==> (mu/2) d tr(F^TF)/dF = (mu/2)*2*F
    g1 = mu.unsqueeze(-1) * defo_grad

    # d (lam/2)*(J-1)^2 / dF ==> (lam/2) d (J-1)^2/dF = (lam/2) * 2*(J-1) * dJ/dF ==> (lam/2) * 2*(J-1) * J*F^-1
    g2 = lam.unsqueeze(-1) * (J.unsqueeze(-1) - 1) * \
        J.unsqueeze(-1) * torch.linalg.inv(defo_grad)

    # d -mu*J / dF ==> -mu dJ/dF = -mu J F^-1
    g3 = -mu.unsqueeze(-1) * J.unsqueeze(-1) * torch.linalg.inv(defo_grad)

    ggg = g1 + g2 + g3
    return ggg


def unbatched_neohookean_energy(mu, lam, defo_grad):
    r"""Implements an version of neohookean energy. Calculate energy per-integration primitive.

    Args:
        mu (torch.Tensor): Tensor of lame parameter mu, of shape :math:`(\text{batch_dim}, 1)`
        lam (torch.Tensor): Tensor of lame parameter lambda, of shape :math:`(\text{batch_dim}, 1)`
        defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

    Returns:
        torch.Tensor: Vector of per-primitive energy values, shape :math:`(\text{batch_dim}, 1)`
    """
    F = defo_grad.reshape(-1, 9)
    F_0 = F[:, 0]
    F_1 = F[:, 1]
    F_2 = F[:, 2]
    F_3 = F[:, 3]
    F_4 = F[:, 4]
    F_5 = F[:, 5]
    F_6 = F[:, 6]
    F_7 = F[:, 7]
    F_8 = F[:, 8]

    J = (F_0 * (F_4 * F_8 - F_5 * F_7)
         - F_1 * (F_3 * F_8 - F_5 * F_6)
         + F_2 * (F_3 * F_7 - F_4 * F_6)).unsqueeze(dim=1)

    IC = (F_0 * F_0 + F_1 * F_1 + F_2 * F_2 +
          F_3 * F_3 + F_4 * F_4 + F_5 * F_5 +
          F_6 * F_6 + F_7 * F_7 + F_8 * F_8).unsqueeze(dim=1)

    return 0.5 * mu * (IC - 3.0) + 0.5 * lam * (J - 1.) * (J - 1.) - mu * (J - 1.0)


def unbatched_neohookean_gradient(mu, lam, defo_grad):
    r"""Implements a version of neohookean gradients w.r.t deformation gradients. Calculate gradient per-integration primitive.

    Args:
        mu (torch.Tensor): Tensor of lame parameter mu, of shape :math:`(\text{batch_dim}, 1)`
        lam (torch.Tensor): Tensor of lame parameter lambda, of shape :math:`(\text{batch_dim}, 1)`
        defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

    Returns:
        torch.Tensor: Tensor of per-primitive neohookean gradient w.r.t defo_grad, flattened, shape :math:`(\text{batch_dim}, 9)`
    """
    F = defo_grad.reshape(-1, 9)
    F_0 = F[:, 0]
    F_1 = F[:, 1]
    F_2 = F[:, 2]
    F_3 = F[:, 3]
    F_4 = F[:, 4]
    F_5 = F[:, 5]
    F_6 = F[:, 6]
    F_7 = F[:, 7]
    F_8 = F[:, 8]
    # compute batch determinant of all F's
    J = (F_0 * (F_4 * F_8 - F_5 * F_7)
         - F_1 * (F_3 * F_8 - F_5 * F_6)
         + F_2 * (F_3 * F_7 - F_4 * F_6)).unsqueeze(dim=1)

    # a = 1.0 + mu/lam
    FinvT = torch.inverse(defo_grad.reshape(-1, 3, 3)).transpose(1, 2)

    return (mu[:, :, None] * defo_grad.reshape(-1, 3, 3) + (J * (lam * (J + (-1.0)) + (-mu))).unsqueeze(dim=2) * FinvT)


def unbatched_neohookean_hessian(mu, lam, defo_grad):
    r"""Implements a version of neohookean hessian w.r.t deformation gradients. Calculate per-integration primitive.

    Args:
        mu (torch.Tensor): Tensor of lame parameter mu, of shape :math:`(\text{batch_dim}, 1)`
        lam (torch.Tensor): Tensor of lame parameter lambda, of shape :math:`(\text{batch_dim}, 1)`
        defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

    Returns:
        torch.Tensor: Tensor of per-primitive neohookean hessian w.r.t defo_grad, flattened, shape :math:`(\text{batch_dim}, 9,9)`
    """
    F = defo_grad.reshape(-1, 9)
    F_0 = F[:, 0]
    F_1 = F[:, 1]
    F_2 = F[:, 2]
    F_3 = F[:, 3]
    F_4 = F[:, 4]
    F_5 = F[:, 5]
    F_6 = F[:, 6]
    F_7 = F[:, 7]
    F_8 = F[:, 8]
    id_mat = torch.eye(9, device=mu.device)

    # can save more time by not recomputing this stuff
    J = (F_0 * (F_4 * F_8 - F_5 * F_7)
         - F_1 * (F_3 * F_8 - F_5 * F_6)
         + F_2 * (F_3 * F_7 - F_4 * F_6)).unsqueeze(dim=1)

    # a = 1.0 + mu/lam
    FinvT = torch.inverse(defo_grad.reshape(-1, 3, 3)).transpose(1, 2)
    gamma = J * (lam * (2.0 * J + (-1.0)) + (-mu))
    dgamma = gamma - lam * J * J

    FFinv = torch.bmm(FinvT.reshape(-1, 9, 1), FinvT.reshape(-1, 1, 9))
    H1 = mu[:, :, None] * id_mat
    H2 = gamma[:, :, None] * FFinv
    H3 = -dgamma[:, :, None] * \
        FFinv.reshape(-1, 3, 3, 3, 3).transpose(2, 4).reshape(-1, 9, 9)

    return H3
