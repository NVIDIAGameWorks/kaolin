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

r"""Newton's method that can run inside a CUDA graph.

It keeps its working values on the GPU and needs CUDA 12.4 or later.
"""

import torch
import warp as wp

from kaolin.physics.utils import warp_utilities

__all__ = ['CapturableNewtonBuffers', 'newtons_method_capturable',
           'pin_kinematic_dofs', 'mask_in_place']


def pin_kinematic_dofs(H_dense, free_dof_mask):
    r"""Pins kinematic DOFs in a dense Hessian, in place.

    Args:
        H_dense (wp.array2d): Dense Hessian of shape :math:`(n, n)`. Modified in place.
        free_dof_mask (wp.array): Length-:math:`n` mask, 1.0 for free DOFs and 0.0 for
            kinematic ones.
    """
    wp.launch(_pin_kinematic_dofs_kernel, dim=H_dense.shape,
              inputs=[H_dense, free_dof_mask])


def mask_in_place(x, mask):
    r"""Element-wise ``x *= mask``, in place and allocation-free.

    Args:
        x (wp.array): Array to mask. Modified in place.
        mask (wp.array): Mask of the same length.
    """
    wp.launch(_mask_in_place_kernel, dim=x.shape, inputs=[x, mask])


@wp.kernel
def _update_if_cond_kernel(if_cond: wp.array(dtype=wp.int32),
                           compare_value: wp.array(dtype=wp.float32)):  # pragma: no cover
    r"""Turns the sign of a device scalar into an int32 predicate for ``wp.capture_if``.

    This is the bridge that replaces every host-side ``if <float comparison>`` in
    the solver.
    """
    if_cond[0] = wp.int32(compare_value[0] >= 0.0)


@wp.kernel
def _pin_kinematic_dofs_kernel(H: wp.array2d(dtype=wp.float32),
                               free_dof_mask: wp.array(dtype=wp.float32)):  # pragma: no cover
    r"""Pins kinematic DOFs in a dense Hessian: ``H[i,j] = H[i,j]m[i]m[j] + d_ij(1-m[i])``.

    Zeroing the kinematic rows *and* columns and putting 1 on their diagonal makes the
    full-size solve algebraically identical to solving the reduced free-DOF system and
    mapping back: ``[H_ff 0; 0 I][dz_f; dz_k] = [-g_f; 0]`` yields
    ``dz_f = -H_ff^-1 g_f`` with ``dz_k`` exactly zero. Zeroing only the diagonal would
    leave the off-diagonal coupling blocks and give a different answer.
    """
    i, j = wp.tid()
    mi = free_dof_mask[i]
    H[i, j] = H[i, j] * mi * free_dof_mask[j]
    if i == j:
        H[i, j] = H[i, j] + (1.0 - mi)


@wp.kernel
def _mask_in_place_kernel(x: wp.array(dtype=wp.float32),
                          mask: wp.array(dtype=wp.float32)):  # pragma: no cover
    tid = wp.tid()
    x[tid] = x[tid] * mask[tid]


@wp.kernel
def _array_min_scalar_kernel(x: wp.array(dtype=wp.float32),
                             t: wp.array(dtype=wp.float32),
                             ti: int,
                             y: wp.array(dtype=wp.float32)):  # pragma: no cover
    r"""``y[i] = min(x[i], t[ti])`` where the scalar step size lives on device.

    Taking ``t`` as an array rather than a Python float is what lets the line
    search vary its step size inside a captured graph.
    """
    tid = wp.tid()
    y[tid] = wp.min(x[tid], t[ti])


def _check_dofs(arr, buffers, what):
    r"""Checks that ``arr`` is a contiguous float32 DOF vector matching ``buffers``.

    Args:
        arr: Value to check.
        buffers (CapturableNewtonBuffers): Buffers the solver was given.
        what (str): Name of the argument or callback that produced ``arr``, so the
            error says which one to go and fix.
    """
    if not isinstance(arr, wp.array):
        raise TypeError(f"{what} must be a wp.array, got {type(arr)}.")
    if arr.dtype != wp.float32:
        raise TypeError(f"{what} must have dtype wp.float32, got {arr.dtype}.")
    if arr.ndim != 1 or arr.shape[0] != buffers.num_dofs:
        raise ValueError(
            f"{what} has shape {tuple(arr.shape)}, expected ({buffers.num_dofs},) to match "
            f"the buffers. CapturableNewtonBuffers was built for {buffers.num_dofs} degrees "
            "of freedom; build it with the same count you solve for.")


def _apply_bounds_capturable(direction, bounds, t, bounded_direction):
    r"""Capturable form of :func:`kaolin.physics.common.optimization._apply_bounds`.

    Computes ``direction * min(bounds, t)`` into a preallocated output, reading the
    step size ``t`` from a device array instead of a Python float.
    """
    wp.launch(_array_min_scalar_kernel, dim=direction.shape,
              inputs=[bounds, t, 0], outputs=[bounded_direction])
    bounded_direction *= direction
    return bounded_direction


class CapturableNewtonBuffers:
    r"""Preallocated scratch for :func:`newtons_method_capturable`.

    Create once and reuse. Do not replace these arrays after recording a graph.

    Args:
        num_dofs (int): Number of values to solve for.
        device (optional): Warp device. Defaults to the current Warp device.
    """

    def __init__(self, num_dofs, device=None):
        device = wp.get_device(device)
        self.num_dofs = num_dofs
        self.device = device

        # Descent direction and its torch alias (a view, not a copy).
        self.dz = wp.zeros(num_dofs, dtype=wp.float32, device=device)
        self.dz_th = wp.to_torch(self.dz)

        self.bounded_direction = wp.zeros(num_dofs, dtype=wp.float32, device=device)
        self.default_bounds = wp.ones(num_dofs, dtype=wp.float32, device=device)

        # Newton loop control.
        self.nm_while_cond = wp.zeros(1, dtype=wp.int32, device=device)
        self.nm_if_cond = wp.zeros(1, dtype=wp.int32, device=device)
        self.nm_branch_value = wp.zeros(1, dtype=wp.float32, device=device)
        self.nm_step_count = wp.zeros(1, dtype=wp.int32, device=device)

        # Line search control.
        self.ls_t = wp.zeros(1, dtype=wp.float32, device=device)
        self.ls_f = wp.zeros(1, dtype=wp.float32, device=device)
        self.ls_branch_value = wp.zeros(1, dtype=wp.float32, device=device)
        self.ls_while_cond = wp.zeros(1, dtype=wp.int32, device=device)
        self.ls_if_cond = wp.zeros(1, dtype=wp.int32, device=device)
        self.ls_can_break = wp.zeros(1, dtype=wp.int32, device=device)
        self.ls_x_new = wp.zeros(num_dofs, dtype=wp.float32, device=device)

        # Linear solve scratch.
        #
        # NOT torch.linalg.solve_ex: even with out=, it allocates its LU and pivot
        # tensors internally (2 allocations per call, verified via TorchDispatchMode).
        # Inside a captured graph those addresses get baked in and then handed back to
        # torch's ordinary cache, owned by nothing -- a later empty_cache() or allocator
        # pressure frees them and the next replay writes into unmapped memory
        # (reproduced as CUDA error 700, illegal memory access).
        #
        # lu_factor_ex(out=) + lu_solve(out=) are both allocation-free and numerically
        # equivalent to torch.linalg.solve.
        torch_device = wp.device_to_torch(device)
        eye = torch.eye(num_dofs, device=torch_device, dtype=torch.float32)
        rhs = torch.zeros(num_dofs, 1, device=torch_device, dtype=torch.float32)
        # Pre-warm so cuSOLVER's workspace exists before any capture begins, and to
        # obtain correctly-shaped/typed LU, pivot and info buffers.
        self.lu_th, self.piv_th, self.solve_info_th = torch.linalg.lu_factor_ex(eye)
        torch.linalg.lu_solve(self.lu_th, self.piv_th, rhs)
        # 2-D views for lu_solve, which requires matrix operands. Views, not copies.
        self.dz_2d_th = self.dz_th.unsqueeze(1)
        del eye, rhs

def _line_search_capturable(energy_fcn, x, direction, gradient, bounds, buffers,
                            initial_step_size=1.0, alpha=1e-3, beta=0.6,
                            max_steps=10):
    r"""Choose a safe Newton step without leaving the GPU.

    After the first accepted step, the search tries one larger step. It stops after
    two accepted steps in a row.

    Args:
        energy_fcn (callable): Takes a ``wp.array`` of DOFs, returns a one-element
            ``wp.array`` holding the energy (NOT a Python float).
        x (wp.array): Current reduced DOFs. Read only.
        direction (wp.array): Search direction.
        gradient (wp.array): Energy gradient at ``x``.
        bounds (wp.array): Per-DOF upper bound on the step size.
        buffers (CapturableNewtonBuffers): Preallocated scratch.
        initial_step_size (float, optional): Starting step size. Defaults to 1.0.
        alpha (float, optional): Armijo parameter. Defaults to 1e-3.
        beta (float, optional): Backtracking factor. Defaults to 0.6.
        max_steps (int, optional): Maximum line search iterations. Defaults to 10.

    Returns:
        wp.array: ``buffers.bounded_direction``, the update to add to ``x``.
    """
    buffers.ls_t.fill_(initial_step_size)
    f0 = energy_fcn(x)
    # Traced once, so free on replay. Without it a float return surfaces as "Copy
    # source and destination must be arrays" from inside wp.copy, several frames away
    # from the callback that caused it.
    if not isinstance(f0, wp.array) or f0.dtype != wp.float32 or f0.shape != (1,):
        raise TypeError(
            "energy_fcn must return a one-element wp.array of float32 -- the energy "
            "stays on the device so the Armijo test never reads it back. Got "
            f"{type(f0).__name__}"
            f"{f'(dtype={f0.dtype}, shape={tuple(f0.shape)})' if isinstance(f0, wp.array) else ''}"
            ". A slice of a longer array is fine, e.g. `return self._energy[2:3]`.")
    wp.copy(dest=buffers.ls_f, src=f0)

    _apply_bounds_capturable(direction, bounds, buffers.ls_t, buffers.bounded_direction)

    buffers.ls_while_cond.fill_(max_steps)
    buffers.ls_if_cond.fill_(0)
    buffers.ls_can_break.fill_(0)
    buffers.ls_branch_value.fill_(0.0)

    def while_body():
        def set_break():
            # Set to 1 so the unconditional decrement at the end of the body
            # drives it to 0, terminating the loop.
            buffers.ls_while_cond.fill_(1)

        def increase_t():
            buffers.ls_can_break += 1
            buffers.ls_t /= beta
            _apply_bounds_capturable(direction, bounds, buffers.ls_t, buffers.bounded_direction)

        def on_armijo_satisfied():
            wp.capture_if(buffers.ls_can_break, on_true=set_break, on_false=increase_t)

        def on_armijo_violated():
            buffers.ls_t *= beta
            _apply_bounds_capturable(direction, bounds, buffers.ls_t, buffers.bounded_direction)

        wp.copy(dest=buffers.ls_x_new, src=x)
        buffers.ls_x_new += buffers.bounded_direction
        f_new = energy_fcn(buffers.ls_x_new)

        # residual = f + alpha * (g . d) - f_new;  >= 0 means Armijo is satisfied.
        warp_utilities.array_inner_capturable(
            gradient, buffers.bounded_direction, buffers.ls_branch_value)
        buffers.ls_branch_value *= alpha
        buffers.ls_branch_value += buffers.ls_f
        buffers.ls_branch_value -= f_new
        wp.launch(_update_if_cond_kernel, dim=1,
                  inputs=[buffers.ls_if_cond, buffers.ls_branch_value])

        wp.capture_if(buffers.ls_if_cond,
                      on_true=on_armijo_satisfied,
                      on_false=on_armijo_violated)
        buffers.ls_while_cond -= 1

    wp.capture_while(buffers.ls_while_cond, while_body=while_body)
    return buffers.bounded_direction


def newtons_method_capturable(x, energy_fcn, gradient_fcn, hessian_fcn, buffers,
                              bounds_fcn=None, nm_max_iters=5, conv_tol=1e-4,
                              max_ls_steps=10, ls_alpha=1e-3, ls_beta=0.6):
    r"""Solve a dense Newton system inside a CUDA graph.

    ``x`` is updated in place. Pin fixed values before calling this function with
    :func:`pin_kinematic_dofs` and :func:`mask_in_place`.

    Callbacks run while the graph is recorded. They must use preallocated GPU arrays,
    must not read values back to Python, and must return the same buffers on every run.
    Values read from Python are fixed until the graph is recorded again.

    ``energy_fcn`` returns one float32 value. ``gradient_fcn`` returns a float32 vector
    with one value per degree of freedom. ``hessian_fcn`` returns a dense float32 matrix
    of the same size. Energy and Hessian callbacks must not change the gradient buffer.
    ``bounds_fcn`` returns per-value limits or ``None``.

    Check ``buffers.solve_info_th`` after replay if a failed solve must be reported. The
    line search may try a larger step after its first accepted step.

    Args:
        x (wp.array): DOFs of shape :math:`(\text{num_dofs},)`. Updated in place.
        energy_fcn (callable): DOFs -> one-element ``wp.array`` energy.
        gradient_fcn (callable): DOFs -> ``wp.array`` gradient of shape :math:`(\text{num_dofs},)`.
        hessian_fcn (callable): DOFs -> dense ``wp.array2d`` Hessian of shape
            :math:`(\text{num_dofs}, \text{num_dofs})`.
        buffers (CapturableNewtonBuffers): Preallocated scratch sized to ``num_dofs``.
        bounds_fcn (callable, optional): ``(dz, x)`` -> per-DOF step bounds, or ``None``
            for unbounded (bounds of 1.0). Defaults to None.
        nm_max_iters (int, optional): Maximum Newton iterations. Defaults to 5.
        conv_tol (float, optional): Convergence tolerance on :math:`|g^T dz|`. Defaults to 1e-4.
        max_ls_steps (int, optional): Maximum line search steps. Defaults to 10.
        ls_alpha (float, optional): Armijo parameter. Defaults to 1e-3.
        ls_beta (float, optional): Backtracking factor. Defaults to 0.6.

    Returns:
        wp.array: ``x``, updated in place.
    """
    # The torch solve below must be issued on the same stream Warp is recording on.
    #
    # Torch's default current stream is the legacy default stream, which CUDA forbids
    # capturing on, so a graph recorded there silently omits the linear solve. But
    # "anything except the default stream" is not sufficient: on any *other* stream the
    # torch work still goes somewhere Warp is not watching, and is dropped just as
    # silently. Warp resolves the active capture from its own device stream, so that is
    # the stream to match.
    #
    # Compared by raw handle, which is an integer comparison on the host and therefore
    # safe to run while a capture is in progress.
    if buffers.device.is_cuda:
        warp_stream = wp.stream_to_torch(buffers.device)
        if torch.cuda.current_stream().cuda_stream != warp_stream.cuda_stream:
            raise RuntimeError(
                "newtons_method_capturable must run on Warp's stream for "
                f"{buffers.device}, but torch's current stream is "
                f"{torch.cuda.current_stream()}. Any other stream -- including the "
                "default one -- means the linear solve is issued where Warp is not "
                "recording, and is dropped from the graph with no error. Wrap the call "
                "in `with torch.cuda.stream(wp.stream_to_torch(device)):`, or invoke it "
                "through kaolin.physics.utils.warp_utilities.replay_or_capture, "
                "which establishes this for you.")

    _check_dofs(x, buffers, "x")

    buffers.nm_while_cond.fill_(nm_max_iters)
    buffers.nm_if_cond.fill_(0)
    buffers.nm_branch_value.fill_(0.0)
    buffers.nm_step_count.fill_(0)

    def newton_while_body():
        G_curr = gradient_fcn(x)
        H_curr = hessian_fcn(x)

        # This body is traced once, so these checks cost nothing on replay. Without
        # them the failures below surface far from their cause: a wrong Hessian shape
        # becomes a torch resize warning (an allocation, which a capture forbids), and
        # an energy function returning a float becomes "Copy source and destination
        # must be arrays" from inside the line search.
        _check_dofs(G_curr, buffers, "gradient_fcn")
        if G_curr.ptr == x.ptr:
            raise ValueError(
                "gradient_fcn returned x itself. Its result is read throughout the "
                "line search, which advances x, so the two must be separate buffers.")
        if not isinstance(H_curr, wp.array) or H_curr.ndim != 2:
            raise TypeError(
                f"hessian_fcn must return a dense 2-D wp.array, got {type(H_curr)}"
                f"{'' if not isinstance(H_curr, wp.array) else f' with ndim={H_curr.ndim}'}"
                ". Sparse matrices are not supported here: their products reallocate "
                "and can change topology between iterations, neither of which can be "
                "captured.")
        if tuple(H_curr.shape) != tuple(buffers.lu_th.shape):
            raise ValueError(
                f"hessian_fcn returned shape {tuple(H_curr.shape)}, but the buffers are "
                f"sized {tuple(buffers.lu_th.shape)}. A mismatch makes lu_factor_ex resize "
                "its outputs, which allocates, which a capture forbids.")

        buffers.dz.zero_()
        # Factor + solve with fully preallocated outputs. The _ex variant returns an
        # info code rather than raising, since raising would need a host readback.
        torch.linalg.lu_factor_ex(
            wp.to_torch(H_curr), out=(buffers.lu_th, buffers.piv_th, buffers.solve_info_th))
        torch.linalg.lu_solve(
            buffers.lu_th, buffers.piv_th, wp.to_torch(G_curr).unsqueeze(1),
            out=buffers.dz_2d_th)
        buffers.dz *= -1.0

        # Converged if |g . dz| < conv_tol. Evaluated on device; the sign of
        # (conv_tol - |g.dz|) becomes the branch predicate.
        warp_utilities.array_inner_capturable(
            G_curr, buffers.dz, buffers.nm_branch_value, take_abs=True)
        buffers.nm_branch_value *= -1.0
        buffers.nm_branch_value += conv_tol
        wp.launch(_update_if_cond_kernel, dim=1,
                  inputs=[buffers.nm_if_cond, buffers.nm_branch_value])

        def exit_while():
            buffers.nm_while_cond.fill_(0)

        def run_line_search():
            # Required: `x += ...` below would otherwise bind x as a local to this
            # function and raise UnboundLocalError.
            nonlocal x
            if bounds_fcn is None:
                wp_bounds = buffers.default_bounds
            else:
                wp_bounds = bounds_fcn(buffers.dz, x)
                if wp_bounds is None:
                    buffers.default_bounds.fill_(1.0)
                    wp_bounds = buffers.default_bounds

            _line_search_capturable(
                energy_fcn=energy_fcn, x=x, direction=buffers.dz,
                gradient=G_curr, bounds=wp_bounds, buffers=buffers,
                initial_step_size=1.0, alpha=ls_alpha, beta=ls_beta,
                max_steps=max_ls_steps)

            x += buffers.bounded_direction
            buffers.nm_while_cond -= 1

        wp.capture_if(buffers.nm_if_cond, on_true=exit_while, on_false=run_line_search)
        buffers.nm_step_count += 1

    wp.capture_while(buffers.nm_while_cond, while_body=newton_while_body)
    return x
