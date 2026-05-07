# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Union, Optional

import torch
import torch.nn as nn
import logging


from kaolin.ops.pointcloud import farthest_point_sampling
from kaolin.physics.materials.material_utils import to_lame
from scipy.spatial import cKDTree

from .network import SkinningModule

logger = logging.getLogger(__name__)

__all__ = [
    'SimplicitsRKPM',
]

class SimplicitsRKPM(SkinningModule):
    r"""Simplicits skinning weights using Reproducing Kernel Particle Method (RKPM).

    Computes skinning weights for Simplicits physics simulation using RKPM basis
    functions. The skinning weights are derived from the eigenvectors of a
    generalized eigenvalue problem involving the mass and elastic hessian matrices
    assembled from RKPM kernel evaluations.


    For more information, see the
    `Freeform <https://research.nvidia.com/labs/sil/projects/freeform/>`_
    project page.

    """

    def __init__(
        self,
        num_handles: int,
        num_nodes: int,
        radius_scale: float = 1.0,
        radius_init_kNN: int = 2,
        radius_min: Union[float, str, None] = "3x",
        num_points: Optional[int] = None,
        dtype: torch.dtype = torch.float64,
        bb_min=None,
        bb_max=None,
    ):
        r""" Constructor for RKPM based skinning module.
        
        Args:
            num_handles (int): Number of deformation handles (non-zero eigenvectors) to use.
            num_nodes (int): Number of RKPM kernel nodes.
            radius_scale (float, optional): Scaling factor applied to node radii
                computed from nearest-neighbor distances. Defaults to 1.0.
            radius_init_kNN (int, optional): Number of nearest neighbors used to
                determine initial node radius. Defaults to 2.
            radius_min (Union[float, str, None], optional): Minimum node radius.
                Can be a float value, or a string of the form ``"Nx"`` (e.g.
                ``"3x"``) to set the minimum as a multiple of the mean
                nearest-neighbor distance among input points. Defaults to ``"3x"``.
            num_points (int, optional): Number of input points to use as samples when forming the mass and stiffness matrices for the RKPM
                    generalized eigenproblem. This only affects **offline** mode construction; the returned :class:`SimplicitsObject` still
                    stores the full ``pts`` geometry, and skinning can be evaluated at any point. 
                    Using a lot of points may lead to high memory usage during basis construction. 
                    Defaults to all input points being used.
            dtype (torch.dtype, optional): Floating-point precision used for RKPM
                kernel evaluations and eigenanalysis. Defaults to ``torch.float64``
                for numerical stability of the generalized eigenproblem.
        """
        super().__init__(bb_min=bb_min, bb_max=bb_max)
        self.num_points = num_points
        self.num_handles = num_handles - 1 # subtract 1 for the constant handle
        self.num_nodes = num_nodes

        self.radius_scale = radius_scale
        self.radius_init_kNN = radius_init_kNN
        self.radius_min = radius_min

        self.rkpm = RKPM(num_nodes)

        self.dtype = dtype
        self.rkpm.to(dtype=self.dtype)

        # eigenvectors
        evecs = nn.Parameter(torch.zeros(self.num_nodes, self.num_handles, dtype=self.dtype))
        self.register_parameter("evecs", evecs)
        self.evecs.requires_grad = False


    def init(self, pts, yms, prs, rhos, appx_vol):
        r"""Initializes the RKPM nodes and eigenvectors from input point cloud data.

        Selects RKPM kernel nodes via Farthest Point Sampling, computes node
        radii from nearest-neighbor distances, and performs a generalized
        eigenanalysis on the mass and stiffness matrices to determine the
        deformation modes.

        Args:
            pts (torch.Tensor): Input points of shape :math:`(N, 3)` in the same coordinate
                frame as :meth:`~kaolin.physics.simplicits.network.SkinningModule.compute_skinning_weights`
                expects (typically world units). They are mapped with ``bb_min`` / ``bb_max``
                before FPS and eigenanalysis so the basis matches evaluation coordinates.
            yms (torch.Tensor): Young's moduli of shape :math:`(N,)`.
            prs (torch.Tensor): Poisson's ratios of shape :math:`(N,)`.
            rhos (torch.Tensor): Densities of shape :math:`(N,)`.
            appx_vol (float): Approximate volume of the object.
        """
        # currently assume all integration samples have equal volume weights = appx_vol / num_points, weights cancelled out

        # Same normalization as SkinningModule.compute_skinning_weights so RKPM nodes,
        # radii, and eigenvectors live in the normalized frame used at evaluation time.
        pts = self._offset_scale(pts)

        # Use Farthest Point Sampling to determine nodes
        device = pts.device
        if pts.shape[0] < self.num_nodes:
            logger.warning(f"num_nodes ({self.num_nodes}) is less than the number of points ({pts.shape[0]}). Using all points as nodes.")
       
            self.num_nodes = pts.shape[0]
            node_indices = torch.arange(0, pts.shape[0], device=device)
            evecs = nn.Parameter(torch.zeros(self.num_nodes, self.num_handles, dtype=self.dtype, device=self.evecs.device))
            self.register_parameter("evecs", evecs)
            self.evecs.requires_grad = False
        else:
            node_indices = farthest_point_sampling(pts[None], self.num_nodes).squeeze(0)
        
        nodes = pts[node_indices]
        nodes_np = nodes.cpu().numpy()
        nodes_kdtree = cKDTree(nodes_np)

        pts_np = pts.cpu().numpy()
        pts_kdtree = cKDTree(pts_np)

        # Compute node radii
        dists, _ = nodes_kdtree.query(nodes.cpu().numpy(), k=self.radius_init_kNN + 1, workers=-1)
        node_radius = torch.tensor(dists[:, -1] * self.radius_scale, device=nodes.device, dtype=nodes.dtype)

        if isinstance(self.radius_min, float):
            node_radius = node_radius.clamp(min=self.radius_min)
        elif isinstance(self.radius_min, str):
            assert self.radius_min[-1] == "x", "radius_min must end with 'x'"
            min_dist_factor = float(self.radius_min[:-1])
            pts_dists, _ = pts_kdtree.query(pts_np, k=2, workers=-1)
            radius_min = pts_dists[:, -1].mean() * min_dist_factor
            node_radius = node_radius.clamp(min=radius_min)
        else:
            raise ValueError("Unknown radius_min")
        
        self.rkpm.set_kernels(nodes, node_radius)

        # Farthest Point Sampling to determine integration points
        if self.num_points is None:
            sample_indices = torch.arange(pts.shape[0], device=pts.device)
        else:
            sample_indices = farthest_point_sampling(pts[None], self.num_points).squeeze(0)
        x = pts[sample_indices]
        yms_x = yms[sample_indices]
        prs_x = prs[sample_indices]

        x = x.to(dtype=self.dtype)
        yms_x = yms_x.to(dtype=self.dtype)
        prs_x = prs_x.to(dtype=self.dtype)

        # Perform eigenanalysis
        M = self.get_mass_matrix(x)
        H = self.get_hessian_matrix(x, yms_x, prs_x)

        # add one for the zero eigenvalue
        evals, evecs = torch.lobpcg(A=H, B=M, k=(self.num_handles + 1), largest=False, X=None)
        self.evecs.data.copy_(evecs[:, 1:])

    def get_mass_matrix(self, x):
        r"""Computes the RKPM mass matrix.

        The mass matrix is :math:`M = \Phi^T \Phi`, where :math:`\Phi` is the
        matrix of RKPM kernel evaluations at the sample points.

        Args:
            x (torch.Tensor): Sample points of shape :math:`(n, 3)`.

        Returns:
            torch.Tensor: Mass matrix of shape :math:`(N, N)`, where :math:`N`
            is the number of RKPM nodes.
        """
        phi_x = self.rkpm.phi(x)
        M = phi_x.T @ phi_x
        return M
    
    def get_hessian_matrix(self, x, yms, prs, reparameterize_lame=True):
        r"""Computes the RKPM stiffness (Hessian) matrix.

        The stiffness matrix is assembled from spatial gradients of the RKPM
        basis functions, scaled by per-point elastic material coefficients
        derived from Young's modulus and Poisson's ratio.

        Args:
            x (torch.Tensor): Sample points of shape :math:`(n, 3)`.
            yms (torch.Tensor): Young's moduli at sample points of shape :math:`(n,)`.
            prs (torch.Tensor): Poisson's ratios at sample points of shape :math:`(n,)`.
            reparameterize_lame (bool, optional): If True, scales by
                :math:`\lambda + 4\mu` (for Neo-Hookean energy whose lame coefficients are reparameterized).
                If False, scales by :math:`\lambda + 3\mu`. Defaults to True.

        Returns:
            torch.Tensor: Stiffness matrix of shape :math:`(N, N)`, where
            :math:`N` is the number of RKPM nodes.
        """
        grad_phi_x = self.rkpm.grad_phi(x)  # (n, N, D=3)
        n, N, D = grad_phi_x.shape
        J = grad_phi_x.permute(0, 2, 1).reshape(n * D, N)
        # assume the stable neohookean energy
        mus, lams = to_lame(yms, prs)
        if reparameterize_lame:
            # scaling factor (\lambda + 4\mu)
            per_point_coeff = lams + 4 * mus
        else:
            per_point_coeff = lams + 3 * mus
        per_dim_coeff = torch.kron(per_point_coeff.flatten(), torch.ones(D, device=x.device, dtype=x.dtype))
        H = J.T @ (per_dim_coeff[:, None] * J)
        return H

    def forward(self, x):
        r"""Evaluates RKPM skinning weights at query points.

        Args:
            x (torch.Tensor): Query points of shape :math:`(n, 3)`.

        Returns:
            torch.Tensor: Skinning weights of shape :math:`(n, C)`, where
            :math:`C` is the number of handles.
        """
        out_dtype = x.dtype
        x = x.to(dtype=self.dtype)
        return self.rkpm(x, self.evecs).to(dtype=out_dtype)

    def grad(self, x):
        r"""Computes spatial gradients of RKPM skinning weights at query points.

        Args:
            x (torch.Tensor): Query points of shape :math:`(n, 3)`.

        Returns:
            torch.Tensor: Skinning weight gradients of shape :math:`(n, C, 3)`,
            where :math:`C` is the number of handles.
        """
        out_dtype = x.dtype
        x = x.to(dtype=self.dtype)
        grad_phi = self.rkpm.grad_phi(x)  # (n, N, D)
        grad = torch.einsum("nNd,Nc->ncd", grad_phi, self.evecs)  # (n, D, C)
        return grad.to(dtype=out_dtype)


class RKPM(nn.Module):
    r"""Reproducing Kernel Particle Method (RKPM) function module.

    Implements first-order RKPM functions with consistency correction,
    allowing gaussiankernel-based interpolation over scattered point data. The corrected
    kernel :math:`\phi_I(x)` satisfies polynomial completeness up to the
    specified polynomial degree. Currently only degree 1 is supported. 

    Args:
        num_nodes (int): Number of kernel nodes.
        polynomial_degree (int, optional): Degree of polynomial basis used for
            consistency correction. Currently only degree 1 is supported.
            Defaults to 1.
    """

    def __init__(
        self,
        num_nodes: int,
        polynomial_degree: int = 1,
    ):
        super(RKPM, self).__init__()

        self.num_nodes = num_nodes

        self.num_dims = 3
        self.polynomial_degree = polynomial_degree
        
        self.initialized = False
        self.register_parameter("nodes", torch.nn.Parameter(torch.zeros(self.num_nodes, self.num_dims)))
        self.nodes.requires_grad = False
        self.register_parameter("radius", torch.nn.Parameter(torch.ones(self.num_nodes)))
        self.radius.requires_grad = False

    def set_kernels(self, nodes, radius):
        r"""Sets the node positions and radii for the RKPM kernels.

        Args:
            nodes (torch.Tensor): Node positions of shape :math:`(N, 3)`.
            radius (torch.Tensor): Per-node kernel radii of shape :math:`(N,)`.
        """
        if self.nodes.shape != nodes.shape:
            self.register_parameter("nodes", torch.nn.Parameter(nodes.to(dtype=self.nodes.dtype, device=self.nodes.device)))
            self.nodes.requires_grad = False
            self.register_parameter("radius", torch.nn.Parameter(radius.to(dtype=self.radius.dtype, device=self.radius.device)))
            self.radius.requires_grad = False
            self.num_nodes = nodes.shape[0]
        else:
            self.nodes.data.copy_(nodes)
            self.radius.data.copy_(radius)
        self.initialized = True

    def func_r(self, r):
        r"""Evaluates the uncorrected radial basis kernel as a function of distance.

        Args:
            r (torch.Tensor): Distances from query points to nodes of shape
                :math:`(n, N)`.

        Returns:
            torch.Tensor: Kernel values of shape :math:`(n, N)`.
        """
        # uncorrected RBF kernel, as a function of radius
        return torch.exp(-(r / self.radius) ** 2)
    
    def func_x(self, x):
        r"""Evaluates the uncorrected radial basis kernel at input locations.

        Args:
            x (torch.Tensor): Query points of shape :math:`(n, 3)`.

        Returns:
            torch.Tensor: Kernel values of shape :math:`(n, N)`.
        """
        # uncorrected RBF kernel, as a function of input location
        r = torch.linalg.norm(x[:, None, :] - self.nodes[None, :, :], dim=-1)
        return self.func_r(r)
    
    def dfunc_dx(self, x):
        r"""Computes the spatial gradient of the uncorrected radial basis kernel.

        Args:
            x (torch.Tensor): Query points of shape :math:`(n, 3)`.

        Returns:
            torch.Tensor: Kernel gradients of shape :math:`(n, N, 3)`.
        """
        # derivative of uncorrected RBF kernel, as a function of input location
        displacement = x[:, None, :] - self.nodes[None, :, :]
        func_x = self.func_x(x)
        return func_x[..., None] * (-2 / self.radius[None, :, None] ** 2) * displacement

    def polynomial(self, x):
        r"""Evaluates the polynomial basis at input locations.

        For degree 1, returns :math:`[1, x, y, z]` for each point.

        Args:
            x (torch.Tensor): Input points of shape :math:`(n, 3)`.

        Returns:
            torch.Tensor: Polynomial basis values of shape :math:`(n, P)`,
            where :math:`P` is the number of polynomial terms.
        """
        if self.polynomial_degree == 1:
            return torch.cat([torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype), x], dim=-1)
        else:
            raise ValueError("Unknown polynomial degree")

    @property
    def P(self):
        r"""Number of polynomial basis terms.

        For degree 1 in 3D, returns 4 (one constant term plus three linear terms).

        Returns:
            int: Number of polynomial terms.
        """
        # number of polynomial terms
        if self.polynomial_degree == 1:
            # [1, x, y, z] for the first order
            return (1 + self.num_dims)
        else:
            raise ValueError("Unknown polynomial degree")
    
    def grad_polynomial(self, x):
        r"""Computes spatial gradients of the polynomial basis.

        For degree 1, the gradient of :math:`[1, x, y, z]` with respect to
        position is :math:`[0, I_{3 \times 3}]`.

        Args:
            x (torch.Tensor): Input points of shape :math:`(n, 3)`.

        Returns:
            torch.Tensor: Polynomial basis gradients of shape :math:`(n, P, 3)`.
        """
        if self.polynomial_degree == 1:
            # Px = [1, x], so dPx/dx = [0, I]
            dPx_dx = torch.zeros(x.shape[0], self.P, self.num_dims, device=x.device, dtype=x.dtype)
            dPx_dx[:, 1:, :] = torch.eye(self.num_dims, device=x.device, dtype=x.dtype)[None, :, :]
        else:
            raise ValueError("Unknown polynomial degree")
        return dPx_dx

    def phi(self, x):
        r"""Evaluates the corrected RKPM basis functions at query points.

        The corrected basis satisfies polynomial completeness, ensuring that
        the interpolation can exactly reproduce polynomials up to the specified
        degree.

        Args:
            x (torch.Tensor): Query points of shape :math:`(n, 3)`.

        Returns:
            torch.Tensor: Corrected kernel weights of shape :math:`(n, N)`.
        """
        # corrected RKPM kernel function, weights for each node value
        func_x = self.func_x(x)
        Pn = self.polynomial(self.nodes)  # (N, P)
        Pn_PnT = torch.einsum("Ni,Nj->Nij", Pn, Pn)  # (N, P, P)
        Mx = torch.einsum("nN,Nij->nij", func_x, Pn_PnT)  # (n, P, P)
        Px = self.polynomial(x)  # (n, P)
        Cx = torch.linalg.solve(Mx, Px)  # (n, P)
        phi_x = (Cx @ Pn.T) * func_x  # (n, P) @ (P, N) -> (n, N)
        return phi_x

    def grad_phi(self, x):
        r"""Computes spatial gradients of the corrected RKPM basis functions.

        Args:
            x (torch.Tensor): Query points of shape :math:`(n, 3)`.

        Returns:
            torch.Tensor: Gradients of corrected kernel weights of shape
            :math:`(n, N, 3)`.
        """
        dfunc_dx = self.dfunc_dx(x)  # (n, N, D)
        func_x = self.func_x(x)  # (n, N)
        
        Pn = self.polynomial(self.nodes)  # (N, P)
        Pn_PnT = torch.einsum("Ni,Nj->Nij", Pn, Pn)  # (N, P, P)
        Mx = torch.einsum("nN,Nij->nij", func_x, Pn_PnT)  # (n, P, P)
        
        Px = self.polynomial(x)  # (n, P)
        Cx = torch.linalg.solve(Mx, Px)  # (n, P)
        
        # phi = (Cx @ Pn.T) * func_x
        # dW/dx = d((Cx @ Pn.T) * func_x)/dx
        #       = (Cx @ Pn.T) * dfunc_dx + (dCx/dx @ Pn.T) * func_x
        
        # First term: (Cx @ Pn.T) * dfunc_dx
        CxPnT = Cx @ Pn.T  # (n, N)
        term1 = CxPnT[..., None] * dfunc_dx  # (n, N, D)

        # Second term: (dCx/dx @ Pn.T) * func_x
        dPx_dx = self.grad_polynomial(x)  # (n, P, D)

        # Compute dMx/dx: shape (n, P, P, D)
        dMx_dx = torch.einsum("nNd,Nij->nijd", dfunc_dx, Pn_PnT)  # (n, P, P, D)

        # dCx/dx = Mx^{-1} @ (dPx/dx - dMx/dx @ Cx)
        # Shape: (n, P, D) = (n, P, P) @ ((n, P, D) - (n, P, D))
        dMx_Cx = torch.einsum("nijd,nj->nid", dMx_dx, Cx)  # (n, P, D)
        dCx_dx = torch.linalg.solve(Mx, dPx_dx - dMx_Cx)  # (n, P, D)

        term2 = torch.einsum("npd,Np->nNd", dCx_dx, Pn) * func_x[..., None]  # (n, N, D)
                    
        grad_phi_x = term1 + term2
        return grad_phi_x

    def forward(self, x, c):
        r"""Evaluates the RKPM interpolation of node values at query points.

        Args:
            x (torch.Tensor): Query points of shape :math:`(n, 3)`.
            c (torch.Tensor): Node values of shape :math:`(N, C)`.

        Returns:
            torch.Tensor: Interpolated values of shape :math:`(n, C)`.
        """
        if not self.initialized:
            raise ValueError("RKPM not initialized.")
        return self.phi(x) @ c
