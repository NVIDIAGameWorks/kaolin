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

r"""CUDA-graph-capturable Newton's method with backtracking line search.

The host-side implementation in :mod:`kaolin.physics.common.optimization` cannot be
captured into a CUDA graph: it breaks out of the Newton loop on a GPU scalar
comparison, runs the Armijo test on Python floats, and reallocates buffers every
iteration. Each of those forces a device-to-host sync, and a sync is illegal
inside a capture.

This module removes all three. Every quantity that would otherwise be a Python
``float`` or ``bool`` -- the step size ``t``, the reference energy ``f``, the
``can_break`` latch, the Armijo residual, the convergence residual, and both loop
counters -- lives in a one-element Warp device array. Control flow is expressed
with ``wp.capture_while`` / ``wp.capture_if``, which lower to CUDA conditional
graph nodes, so the loops stay data-dependent without the host ever observing a
value. "Break" becomes ``cond.fill_(0)``.

Requires CUDA 12.4+ for conditional graph node support. Without it Warp falls
back to a host readback per iteration, which still produces correct results but
forfeits the speedup.
"""

import torch
import warp as wp

__all__ = ['CapturableNewtonBuffers', 'newtons_method_capturable',
           'apply_kinematic_bc', 'mask_in_place']


def apply_kinematic_bc(H_dense, free_mask):
    r"""Pins kinematic DOFs in a dense Hessian, in place.

    Args:
        H_dense (wp.array2d): Dense Hessian of shape :math:`(n, n)`. Modified in place.
        free_mask (wp.array): Length-:math:`n` mask, 1.0 for free DOFs and 0.0 for
            kinematic ones.
    """
    wp.launch(_apply_kinematic_bc_kernel, dim=H_dense.shape,
              inputs=[H_dense, free_mask])


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
def _array_inner_kernel(a: wp.array(dtype=wp.float32),
                        b: wp.array(dtype=wp.float32),
                        out: wp.array(dtype=wp.float32)):  # pragma: no cover
    tid = wp.tid()
    wp.atomic_add(out, 0, a[tid] * b[tid])


@wp.kernel
def _array_abs_kernel(a: wp.array(dtype=wp.float32),
                      out: wp.array(dtype=wp.float32)):  # pragma: no cover
    tid = wp.tid()
    out[tid] = wp.abs(a[tid])


@wp.kernel
def _assert_zero_kernel(a: wp.array(dtype=wp.int32), i: int):  # pragma: no cover
    assert a[i] == 0


def _assert_zero(a, i=0):
    r"""Device-side assertion that ``a[i] == 0``, safe to launch inside a capture.

    Warp only emits kernel asserts in debug mode, so this is a no-op in release builds
    and must be paired with a host-side check for anything that must not pass silently.
    """
    wp.launch(_assert_zero_kernel, dim=1, inputs=[a, i])


@wp.kernel
def _apply_kinematic_bc_kernel(H: wp.array2d(dtype=wp.float32),
                               free_mask: wp.array(dtype=wp.float32)):  # pragma: no cover
    r"""Pins kinematic DOFs in a dense Hessian: ``H[i,j] = H[i,j]m[i]m[j] + d_ij(1-m[i])``.

    Zeroing the kinematic rows *and* columns and putting 1 on their diagonal makes the
    full-size solve algebraically identical to solving the reduced free-DOF system and
    mapping back: ``[H_ff 0; 0 I][dz_f; dz_k] = [-g_f; 0]`` yields
    ``dz_f = -H_ff^-1 g_f`` with ``dz_k`` exactly zero. Zeroing only the diagonal would
    leave the off-diagonal coupling blocks and give a different answer.
    """
    i, j = wp.tid()
    mi = free_mask[i]
    H[i, j] = H[i, j] * mi * free_mask[j]
    if i == j:
        H[i, j] = H[i, j] + (1.0 - mi)


@wp.kernel
def _mask_in_place_kernel(x: wp.array(dtype=wp.float32),
                          mask: wp.array(dtype=wp.float32)):  # pragma: no cover
    tid = wp.tid()
    x[tid] = x[tid] * mask[tid]


@wp.kernel
def _array_min_scalar_kernel(x: wp.array(dtype=wp.float32),
                             a: wp.array(dtype=wp.float32),
                             ai: int,
                             y: wp.array(dtype=wp.float32)):  # pragma: no cover
    r"""``y[i] = min(x[i], a[ai])`` where the scalar bound lives on device.

    Taking ``a`` as an array rather than a Python float is what lets the line
    search vary its step size inside a captured graph.
    """
    tid = wp.tid()
    y[tid] = wp.min(x[tid], a[ai])


def _launch_array_inner(a, b, out, alpha=0.0, take_abs=False):
    r"""Device-side inner product accumulating into ``out[0]``.

    Args:
        a (wp.array): Left operand.
        b (wp.array): Right operand.
        out (wp.array): One-element output; reset or scaled by ``alpha`` before accumulating.
        alpha (float, optional): Prescale applied to ``out``. 0.0 means overwrite. Defaults to 0.0.
        take_abs (bool, optional): Take the absolute value of the result. Defaults to False.
    """
    if alpha == 0.0:
        # NOT `out *= 0.0`: 0.0 * inf and 0.0 * NaN are both NaN, so a single non-finite
        # value would stick permanently. The downstream predicate is
        # `int32(value >= 0.0)`, which reads 0 for NaN -- so convergence would report
        # "not converged" and Armijo "violated" forever, and Newton would silently burn
        # every iteration on every subsequent step.
        out.zero_()
    else:
        out *= alpha
    wp.launch(_array_inner_kernel, dim=a.shape, inputs=[a, b], outputs=[out])
    if take_abs:
        wp.launch(_array_abs_kernel, dim=out.shape, inputs=[out], outputs=[out])


def _check_dofs(arr, buf, what):
    r"""Checks that ``arr`` is a contiguous float32 DOF vector matching ``buf``.

    Args:
        arr: Value to check.
        buf (CapturableNewtonBuffers): Buffers the solver was given.
        what (str): Name of the argument or callback that produced ``arr``, so the
            error says which one to go and fix.
    """
    if not isinstance(arr, wp.array):
        raise TypeError(f"{what} must be a wp.array, got {type(arr)}.")
    if arr.dtype != wp.float32:
        raise TypeError(f"{what} must have dtype wp.float32, got {arr.dtype}.")
    if arr.ndim != 1 or arr.shape[0] != buf.num_dofs:
        raise ValueError(
            f"{what} has shape {tuple(arr.shape)}, expected ({buf.num_dofs},) to match "
            f"the buffers. CapturableNewtonBuffers was built for {buf.num_dofs} degrees "
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

    Allocated once and reused for every step. Graph capture records raw device
    pointers, so these buffers must never be reassigned -- only written in place.

    Args:
        num_dofs (int): Number of (reduced) degrees of freedom.
        device (optional): Warp device, as anything :func:`warp.get_device` accepts.
            Defaults to Warp's current device, so this class is constructible on a
            machine without a GPU rather than failing with a raw CUDA error.
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
        self.nm_compare_value = wp.zeros(1, dtype=wp.float32, device=device)
        self.nm_step_count = wp.zeros(1, dtype=wp.int32, device=device)

        # Line search control.
        self.ls_t = wp.zeros(1, dtype=wp.float32, device=device)
        self.ls_f = wp.zeros(1, dtype=wp.float32, device=device)
        self.ls_compare_value = wp.zeros(1, dtype=wp.float32, device=device)
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

        # Device-side view of the info code, for an in-graph assert. .view(1) on the
        # 0-dim info tensor shares storage; reshape().contiguous() could silently copy
        # and would then never observe what lu_factor_ex writes.
        self.solve_info = wp.from_torch(self.solve_info_th.view(1))
        assert self.solve_info.ptr == self.solve_info_th.data_ptr(), \
            "solve_info must alias solve_info_th"


def _line_search_capturable(energy_fcn, x, direction, gradient, bounds, buf,
                            initial_step_size=1.0, alpha=1e-3, beta=0.6,
                            max_steps=10):
    r"""Backtracking line search with the Armijo sufficient-decrease condition, capturable.

    Semantically identical to :func:`kaolin.physics.common.optimization._line_search`,
    including its two-phase behaviour: on the first success the step is *grown*
    (``t /= beta``) and the ``can_break`` latch is set, so the search only returns
    once a second consecutive success occurs.

    Args:
        energy_fcn (callable): Takes a ``wp.array`` of DOFs, returns a one-element
            ``wp.array`` holding the energy (NOT a Python float).
        x (wp.array): Current reduced DOFs. Read only.
        direction (wp.array): Search direction.
        gradient (wp.array): Energy gradient at ``x``.
        bounds (wp.array): Per-DOF upper bound on the step size.
        buf (CapturableNewtonBuffers): Preallocated scratch.
        initial_step_size (float, optional): Starting step size. Defaults to 1.0.
        alpha (float, optional): Armijo parameter. Defaults to 1e-3.
        beta (float, optional): Backtracking factor. Defaults to 0.6.
        max_steps (int, optional): Maximum line search iterations. Defaults to 10.

    Returns:
        wp.array: ``buf.bounded_direction``, the update to add to ``x``.
    """
    buf.ls_t.fill_(initial_step_size)
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
    wp.copy(dest=buf.ls_f, src=f0)

    _apply_bounds_capturable(direction, bounds, buf.ls_t, buf.bounded_direction)

    buf.ls_while_cond.fill_(max_steps)
    buf.ls_if_cond.fill_(0)
    buf.ls_can_break.fill_(0)
    buf.ls_compare_value.fill_(0.0)

    def while_body():
        def set_break():
            # Set to 1 so the unconditional decrement at the end of the body
            # drives it to 0, terminating the loop.
            buf.ls_while_cond.fill_(1)

        def increase_t():
            buf.ls_can_break += 1
            buf.ls_t /= beta
            _apply_bounds_capturable(direction, bounds, buf.ls_t, buf.bounded_direction)

        def on_armijo_satisfied():
            wp.capture_if(buf.ls_can_break, on_true=set_break, on_false=increase_t)

        def on_armijo_violated():
            buf.ls_t *= beta
            _apply_bounds_capturable(direction, bounds, buf.ls_t, buf.bounded_direction)

        wp.copy(dest=buf.ls_x_new, src=x)
        buf.ls_x_new += buf.bounded_direction
        f_new = energy_fcn(buf.ls_x_new)

        # residual = f + alpha * (g . d) - f_new;  >= 0 means Armijo is satisfied.
        _launch_array_inner(gradient, buf.bounded_direction, buf.ls_compare_value)
        buf.ls_compare_value *= alpha
        buf.ls_compare_value += buf.ls_f
        buf.ls_compare_value -= f_new
        wp.launch(_update_if_cond_kernel, dim=1,
                  inputs=[buf.ls_if_cond, buf.ls_compare_value])

        wp.capture_if(buf.ls_if_cond,
                      on_true=on_armijo_satisfied,
                      on_false=on_armijo_violated)
        buf.ls_while_cond -= 1

    wp.capture_while(buf.ls_while_cond, while_body=while_body)
    return buf.bounded_direction


def newtons_method_capturable(x, energy_fcn, gradient_fcn, hessian_fcn, buf,
                              bounds_fcn=None, nm_max_iters=5, conv_tol=1e-4,
                              max_ls_steps=10, ls_alpha=1e-3, ls_beta=0.6):
    r"""CUDA-graph-capturable Newton's method.

    Drop-in replacement for :func:`kaolin.physics.common.optimization.newtons_method`.
    Updates ``x`` in place rather than rebinding it, so the caller's buffer address
    stays valid across graph replays.

    Takes no ``P``/``Pt``: kinematic DOFs are not projected out but **pinned in place**,
    by zeroing their Hessian rows and columns (unit diagonal) and their gradient entries
    before the solve. See :func:`apply_kinematic_bc`. That is algebraically identical to
    the reduced solve, because ``create_projection_matrix`` returns a pure row-selection
    matrix, and it is what keeps every shape fixed for capture. The caller is responsible
    for applying the mask; this function just solves whatever system it is handed.

    The Hessian is dense and preallocated: sparse BSR products reallocate and can
    change topology between iterations, which is not capturable.

    **The callback contract**

    The four callbacks are invoked *once*, while the graph is being recorded, from
    inside ``wp.capture_while`` / ``wp.capture_if`` bodies. Everything below follows
    from that, and none of it is enforced by Python -- most violations produce a wrong
    answer rather than an error, so they are listed here rather than left to be
    discovered.

    Rules that apply to all four:

    * **Do not allocate on the device.** Allocation inside a conditional graph node is
      illegal. Preallocate every buffer the callback touches, once, before the first
      call.
    * **Do not wait on the GPU.** No ``.numpy()``, ``.item()``, ``int()`` of a device
      value, or anything else that reads a result back to the host.
    * **Do not defer first-time setup into the callback.** Creating a cuBLAS handle
      during a capture poisons the context, and Warp compiling a kernel mid-capture is
      not allowed. Run one throwaway call of anything lazy beforehand -- this class
      factors an identity in its own constructor for exactly that reason.
    * **Return the same addresses every time.** The graph records raw pointers. A fresh
      Python object wrapping a stable buffer is fine (``return self._energy[2:3]``);
      a freshly allocated array is not.
    * **Every Python value a callback reads is frozen when the graph is recorded**,
      including ``self.some_scalar`` and any ``if`` on a host value. To change one, the
      graph has to be recorded again.
    * A closure that assigns to a name from an enclosing scope needs ``nonlocal``, or
      Python treats it as local and raises ``UnboundLocalError`` during tracing.
    * **Any device array handed to a callback from outside must keep its address across
      steps.** Where that array comes from a library that can either allocate or fill in
      place, use the in-place form -- e.g. allocate a contacts buffer once and pass it
      to every collision query, rather than letting each query return a new one.

    Per callback:

    * ``energy_fcn(x)`` returns a **one-element** ``wp.array`` of float32, not a Python
      float -- the energy stays on the device so the Armijo test never reads it back. It
      is called with two *different* arrays (``x`` itself, and the trial point), so it
      must read its argument rather than assume a fixed input buffer.
    * ``gradient_fcn(x)`` returns a contiguous float32 array of shape
      :math:`(\text{num_dofs},)`. **Neither ``hessian_fcn`` nor ``energy_fcn`` may write
      into it.** Its value is held across the whole line search, so clobbering it makes
      the solver report convergence and return ``x`` unchanged -- silently, and only on
      some problems, which is why it is stated here and not left to a test.
    * ``hessian_fcn(x)`` returns a dense 2-D ``wp.array`` whose shape matches
      ``buf.lu_th`` exactly. A mismatch makes the factorization resize its outputs,
      which allocates.
    * ``bounds_fcn(dz, x)`` returns per-DOF bounds or ``None``. Returning ``None`` must
      be a fixed decision, not one that depends on device state -- it is evaluated once,
      when the graph is recorded.

    **Reading the outcome.** A singular Hessian does not raise: the factorization writes
    a non-zero code into ``buf.solve_info_th`` and the solve produces NaNs. The in-graph
    assertion only fires in Warp debug builds, so a caller that needs to know must read
    ``buf.solve_info_th`` on the host after the replay and roll back its own state.

    **The line search grows before it stops.** On the *first* step that satisfies Armijo
    it does not return -- it increases the step (``t /= ls_beta``) and tries again,
    returning only on a second consecutive success. This is deliberate and predates the
    capturable version; it means the accepted step can exceed ``initial_step_size``
    unless ``bounds_fcn`` caps it.

    Args:
        x (wp.array): DOFs of shape :math:`(\text{num_dofs},)`. Updated in place.
        energy_fcn (callable): DOFs -> one-element ``wp.array`` energy.
        gradient_fcn (callable): DOFs -> ``wp.array`` gradient of shape :math:`(\text{num_dofs},)`.
        hessian_fcn (callable): DOFs -> dense ``wp.array2d`` Hessian of shape
            :math:`(\text{num_dofs}, \text{num_dofs})`.
        buf (CapturableNewtonBuffers): Preallocated scratch sized to ``num_dofs``.
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
    if buf.device.is_cuda:
        warp_stream = wp.stream_to_torch(buf.device)
        if torch.cuda.current_stream().cuda_stream != warp_stream.cuda_stream:
            raise RuntimeError(
                "newtons_method_capturable must run on Warp's stream for "
                f"{buf.device}, but torch's current stream is "
                f"{torch.cuda.current_stream()}. Any other stream -- including the "
                "default one -- means the linear solve is issued where Warp is not "
                "recording, and is dropped from the graph with no error. Wrap the call "
                "in `with torch.cuda.stream(wp.stream_to_torch(device)):`, or invoke it "
                "through kaolin.physics.utils.warp_utilities.capture_and_run_torch, "
                "which establishes this for you.")

    _check_dofs(x, buf, "x")

    buf.nm_while_cond.fill_(nm_max_iters)
    buf.nm_if_cond.fill_(0)
    buf.nm_compare_value.fill_(0.0)
    buf.nm_step_count.fill_(0)

    def newton_while_body():
        G_curr = gradient_fcn(x)
        H_curr = hessian_fcn(x)

        # This body is traced once, so these checks cost nothing on replay. Without
        # them the failures below surface far from their cause: a wrong Hessian shape
        # becomes a torch resize warning (an allocation, which a capture forbids), and
        # an energy function returning a float becomes "Copy source and destination
        # must be arrays" from inside the line search.
        _check_dofs(G_curr, buf, "gradient_fcn")
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
        if tuple(H_curr.shape) != tuple(buf.lu_th.shape):
            raise ValueError(
                f"hessian_fcn returned shape {tuple(H_curr.shape)}, but the buffers are "
                f"sized {tuple(buf.lu_th.shape)}. A mismatch makes lu_factor_ex resize "
                "its outputs, which allocates, which a capture forbids.")

        buf.dz.zero_()
        # Factor + solve with fully preallocated outputs. The _ex variant returns an
        # info code rather than raising, since raising would need a host readback.
        torch.linalg.lu_factor_ex(
            wp.to_torch(H_curr), out=(buf.lu_th, buf.piv_th, buf.solve_info_th))
        torch.linalg.lu_solve(
            buf.lu_th, buf.piv_th, wp.to_torch(G_curr).unsqueeze(1),
            out=buf.dz_2d_th)
        # Device-side check on the factorization info code. Free, but Warp only emits
        # kernel asserts when wp.config.mode == "debug", so the scene also reads
        # buf.solve_info_th on the host after capture_launch (see check_solve_info).
        _assert_zero(buf.solve_info, 0)
        buf.dz *= -1.0

        # Converged if |g . dz| < conv_tol. Evaluated on device; the sign of
        # (conv_tol - |g.dz|) becomes the branch predicate.
        _launch_array_inner(G_curr, buf.dz, buf.nm_compare_value, take_abs=True)
        buf.nm_compare_value *= -1.0
        buf.nm_compare_value += conv_tol
        wp.launch(_update_if_cond_kernel, dim=1,
                  inputs=[buf.nm_if_cond, buf.nm_compare_value])

        def exit_while():
            buf.nm_while_cond.fill_(0)

        def run_line_search():
            # Required: `x += ...` below would otherwise bind x as a local to this
            # function and raise UnboundLocalError.
            nonlocal x
            if bounds_fcn is None:
                wp_bounds = buf.default_bounds
            else:
                wp_bounds = bounds_fcn(buf.dz, x)
                if wp_bounds is None:
                    buf.default_bounds.fill_(1.0)
                    wp_bounds = buf.default_bounds

            _line_search_capturable(
                energy_fcn=energy_fcn, x=x, direction=buf.dz,
                gradient=G_curr, bounds=wp_bounds, buf=buf,
                initial_step_size=1.0, alpha=ls_alpha, beta=ls_beta,
                max_steps=max_ls_steps)

            x += buf.bounded_direction
            buf.nm_while_cond -= 1

        wp.capture_if(buf.nm_if_cond, on_true=exit_while, on_false=run_line_search)
        buf.nm_step_count += 1

    wp.capture_while(buf.nm_while_cond, while_body=newton_while_body)
    return x
