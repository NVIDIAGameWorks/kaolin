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

import torch
import logging
import warp as wp
from warp.optim.linear import cg, preconditioner

from kaolin.physics.utils import _wp_bsr_to_torch_bsr

logger = logging.getLogger(__name__)

__all__ = ['newtons_method']


def _print_callback(i, err, tol):
    print(f"CG solver: at iteration {i} error = \t {err}  \t tol: {tol}")


def _full_to_red(wp_Pt, wp_x):
    if wp_Pt is None:
        return wp_x
    else:
        return wp.array(wp_Pt @ wp_x).flatten()


def _red_to_full(wp_P, wp_red_x):
    if wp_P is None:
        return wp_red_x
    else:
        return wp.array(wp_P @ wp_red_x).flatten()

@torch.no_grad()
def _apply_bounds(direction, bounds, t, bounds_qr_tfm=None, bounds_qr_tfm_inv=None):
    """Applies bounds to a search direction in optimization.

    This function takes a search direction and scales it element-wise by the minimum
    of the provided bounds and current line search step size t. This ensures the optimization step respects
    any constraints defined by the bounds.

    Args:
        direction (torch.Tensor): The search direction vector
        bounds (torch.Tensor): Per-element bounds on the step size
        t (float): Global line search step size scaling factor
        bounds_qr_tfm (torch.Tensor, optional): When ``bounds`` were computed against a
            Jacobian in a different DOF basis (e.g. the raw pre-QR basis while ``direction``
            lives in the post-QR basis), this matrix rotates ``direction`` into the basis
            where the clamp is meaningful. Must be paired with ``bounds_qr_tfm_inv``.
        bounds_qr_tfm_inv (torch.Tensor, optional): Inverse of ``bounds_qr_tfm``. Used
            to rotate the clamped step back into the basis ``direction`` came from.

    Returns:
        torch.Tensor: The bounded/scaled search direction
    """
    min_bounds = torch.min(bounds, torch.as_tensor(
        t, dtype=bounds.dtype, device=bounds.device))

    if bounds_qr_tfm is None or bounds_qr_tfm_inv is None:
        return direction * min_bounds

    # Background:
    # Updated direction lives in the post-QR (new) basis; bounds live in the raw (old) basis.
    # Details:
    # 1) Rotate direction into the old basis
    # 2) Do the element-wise clamp there
    # 3) Rotate the clamped step back to the new basis
    # Vectors transform linearly under change of basis; bounds (an axis-aligned box) would not, 
    # which is why we wrap the clamp instead of trying to transform bounds directly.

    direction_old = bounds_qr_tfm @ direction
    dz_old_step = direction_old * min_bounds
    dz_new_step = bounds_qr_tfm_inv @ dz_old_step
    return dz_new_step


@torch.no_grad()
def _line_search(func, x, wp_P, direction, gradient, initial_step_size, bounds,
                 alpha=1e-3, beta=0.6, max_steps=10,
                 bounds_qr_tfm=None, bounds_qr_tfm_inv=None):
    r"""Implements the simplest backtracking line search with the sufficient decrease Armijo condition

    Args:
        func (function): Optimization Energy/Loss function
        x (torch.Tensor): Optimization objective (the value you're optimizing)
        wp_P (wps.bsr_matrix or None): Projection matrix mapping the reduced ``x`` back
            to the full DOF space before calling ``func``. ``None`` means no projection.
        direction (torch.Tensor): Search direction
        gradient (torch.Tensor): Optimization gradient
        initial_step_size (float): Initial step size
        bounds (torch.Tensor): Per-element bounds on the step size; forwarded to ``_apply_bounds``.
        alpha (float, optional): LS parameter. Defaults to 1e-3.
        beta (float, optional): LS parameter. Defaults to 0.6.
        max_steps (int, optional): Max number of line search steps. Defaults to 10.
        bounds_qr_tfm (torch.Tensor, optional): Forwarded to ``_apply_bounds``; see that
            function for the basis-change semantics.
        bounds_qr_tfm_inv (torch.Tensor, optional): Forwarded to ``_apply_bounds``.

    Returns:
        torch.Tensor: The bounded search direction scaled by the chosen step size
        (i.e. the update to add to ``x``), not a scalar step size.
    """
    t = initial_step_size  # Initial step size
    wp_x = _red_to_full(wp_P, wp.from_torch(x))
    f = func(wp_x)

    can_break = False
    bounded_direction = _apply_bounds(
        direction, bounds, t, bounds_qr_tfm, bounds_qr_tfm_inv)

    for __ in range(max_steps):
        x_new = x + bounded_direction
        wp_x_new = _red_to_full(wp_P, wp.from_torch(x_new))
        f_new = func(wp_x_new)

        if f_new <= f + alpha * (gradient.permute(torch.arange(gradient.ndim - 1, -1, -1)) @ bounded_direction):
            if (can_break == True):
                return bounded_direction
            else:
                can_break = True
                t = t / beta  # Increase the step size
                bounded_direction = _apply_bounds(
                    direction, bounds, t, bounds_qr_tfm, bounds_qr_tfm_inv)
        else:
            t *= beta  # Reduce the step size
            bounded_direction = _apply_bounds(
                direction, bounds, t, bounds_qr_tfm, bounds_qr_tfm_inv)

    return bounded_direction  # Return the final step size if max_iters is reached

@torch.no_grad()
def newtons_method(x,
                   energy_fcn,
                   gradient_fcn,
                   hessian_fcn,
                   bounds_fcn=None,
                   preconditioner_fcn=None,
                   P=None,
                   Pt=None,
                   nm_max_iters=5,
                   cg_tol=1e-4,
                   cg_iters=100,
                   conv_tol=1e-4,
                   direct_solve=False,
                   bounds_qr_tfm=None,
                   bounds_qr_tfm_inv=None):
    r""" Newton's method optimizes for the updated dofs at the next time step. At each iteration, it computes the updated direction `dz`, 
    finds an appropriate step size, and updates the dofs. It continues to do this iteratively until the directional update is small which indicates the energy is minimized.

    Args:
        x (wp.array): Initial guess of size :math:`(\text{num_dofs},)`
        energy_fcn (function): Energy function. Takes in warp.array and returns a scalar.
        gradient_fcn (function): Gradient function. Takes in warp.array and returns a warp.array.
        hessian_fcn (function): Hessian function. Takes in warp.array and returns a sparse matrix (wps.bsr_matrix).
        bounds_fcn (function): Bounds function to compute the collision bounds of each DOF. Takes two warp.arrays and returns a warp.array.
        preconditioner_fcn (function): Preconditioner function. Takes in a sparse matrix and returns a preconditioner.
        P (wps.bsr_matrix, optional): Projection matrix. Defaults to None.
        Pt (wps.bsr_matrix, optional): Projection matrix transpose. Defaults to None.
        nm_max_iters (int, optional): Maximum number of iterations. Defaults to 5.
        cg_tol (float, optional): CG tolerance. Defaults to 1e-4.
        cg_iters (int, optional): CG iterations. Defaults to 100.
        conv_tol (float, optional): Convergence tolerance. Defaults to 1e-4.
        direct_solve (bool, optional): Whether to use a dense direct solver, or a sparse CG solver. Defaults to False.
        bounds_qr_tfm (torch.Tensor, optional): If apply_qr=True, this is the forward direction used in the line search's apply_bounds for collision bounds in the raw sparse-DOF basis. If apply_qr=False, this is None. Forwarded to ``_apply_bounds``; see that function for the basis-change semantics.
        bounds_qr_tfm_inv (torch.Tensor, optional): If apply_qr=True, this is the inverse direction used in the line search's apply_bounds for collision bounds in the raw sparse-DOF basis. If apply_qr=False, this is None. Forwarded to ``_apply_bounds``.

    Returns:
        wp.array: Optimized objective of size :math:`(\text{num_dofs},)`
    """
    # 1. Detect collisions, build collision jacobians
    # 2. Start newton iteration
    # 3. Compute update dz with all grads, hessians,
    # 4. Compute collision bounds
    # 5. Do line search
    
    # Get the kinematic dofs
    t_x_kinematic = wp.to_torch(x) - wp.to_torch(_red_to_full(P, _full_to_red(Pt, x))) # x - P @ Pt @ x

    converged = False
    for k in range(nm_max_iters):
        E_curr = energy_fcn(x)  # scalar
        G_curr = gradient_fcn(x).flatten()  # vector
        H_curr = hessian_fcn(x)  # sparse matrix

        # Project out the kinematic dofs
        if P is not None:
            red_H = Pt @ H_curr @ P
            red_g = Pt @ G_curr
        else:
            red_H = H_curr
            red_g = G_curr

        # Convert reduced dofs and gradient to torch
        t_red_x = wp.to_torch(_full_to_red(Pt, x))
        t_red_g = wp.to_torch(red_g)

        if direct_solve:
            A = _wp_bsr_to_torch_bsr(red_H).to_dense()
            b = wp.to_torch(red_g)
            t_red_dx = -torch.linalg.solve(A, b)
        else:
            # TODO: Change this to a block-wise preconditioner, so CG converges in 1 step when
            # The hessian is block-diagonal and there is no contact
            precond = preconditioner(
                red_H, "diag") if preconditioner_fcn is None else preconditioner_fcn(red_H)
            dx = wp.zeros_like(red_g)
            ret_vals = cg(
                A=red_H,
                b=red_g,
                x=dx,
                tol=cg_tol,
                maxiter=cg_iters,
                M=precond,
                callback=_print_callback,
            )
            t_red_dx = -wp.to_torch(dx)

        # Converges if the directional update is small
        if (torch.abs(t_red_dx.t() @ t_red_g) < conv_tol):
            converged = True
            break

        full_dx = _red_to_full(P, wp.from_torch(t_red_dx))
        wp_bounds = bounds_fcn(
            full_dx, x)
        if wp_bounds is None:
            t_bounds = torch.ones_like(t_red_x)
        else:
            t_bounds = wp.to_torch(_full_to_red(Pt, wp_bounds))

        last_alpha = 1.0
        bounded_update = _line_search(func=energy_fcn,
                                      x=t_red_x,
                                      wp_P=P,
                                      direction=t_red_dx,
                                      gradient=t_red_g,
                                      initial_step_size=last_alpha,
                                      bounds=t_bounds,
                                      bounds_qr_tfm=bounds_qr_tfm,
                                      bounds_qr_tfm_inv=bounds_qr_tfm_inv)

        t_red_x = t_red_x + bounded_update

        t_x = wp.to_torch(_red_to_full(P, wp.from_torch(t_red_x))) + t_x_kinematic 
        x = wp.from_torch(t_x)

    return x
