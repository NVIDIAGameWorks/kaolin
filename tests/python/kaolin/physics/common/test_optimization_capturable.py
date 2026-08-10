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

r"""Tests for the capturable Newton solver, driven with no Simplicits scene.

Everything here builds its own callbacks against a plain optimization problem, which is
the point: the solver is usable outside the simulator, and these tests are what says so.

Note the choice of problems. A quadratic is the natural first test -- Newton reaches the
exact minimizer in one step -- but its line search accepts ``t = 1`` every iteration, so
it exercises none of the backtracking and cannot expose a callback that clobbers the
gradient buffer. Anything checking the line search or the aliasing rule uses the
non-quadratic below instead.
"""

import pytest
import torch
import warp as wp

from kaolin.physics.common import CapturableNewtonBuffers, newtons_method_capturable
from kaolin.physics.utils.warp_utilities import capture_and_run_torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="graph capture requires CUDA")


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class Quadratic:
    r"""f(x) = 0.5 x^T A x - b^T x, with a known minimizer A^-1 b.

    Written the way the contract requires: every buffer is allocated once here, the
    callbacks only write into them, and each returns the same array every call.
    """

    def __init__(self, n, device, seed=0):
        torch.manual_seed(seed)
        td = wp.device_to_torch(wp.get_device(device))
        m = torch.randn(n, n, device=td)
        self.A_th = (m @ m.T + n * torch.eye(n, device=td)).contiguous()
        self.b_th = torch.arange(1.0, n + 1, device=td).contiguous()
        self.solution = torch.linalg.solve(self.A_th, self.b_th)

        self.H = wp.from_torch(self.A_th)
        self._g_th = torch.zeros(n, device=td)
        self.g = wp.from_torch(self._g_th)
        self._e_th = torch.zeros(1, device=td)
        self.e = wp.from_torch(self._e_th)

    def energy(self, x):
        t = wp.to_torch(x)
        torch.matmul(self.A_th, t, out=self._g_th)
        self._e_th.copy_((0.5 * torch.dot(t, self._g_th) - torch.dot(self.b_th, t)).view(1))
        return self.e

    def gradient(self, x):
        t = wp.to_torch(x)
        torch.matmul(self.A_th, t, out=self._g_th)
        self._g_th.sub_(self.b_th)
        return self.g

    def hessian(self, x):
        return self.H


class SoftAbs:
    r"""f(x) = sum sqrt(1 + x_i^2): separable, strictly convex, and *not* quadratic.

    Newton overshoots from a distance here, so the line search genuinely backtracks --
    which is what makes this the right problem for the line-search and aliasing tests.
    The minimizer is the origin.
    """

    def __init__(self, n, device):
        td = wp.device_to_torch(wp.get_device(device))
        self._g_th = torch.zeros(n, device=td)
        self.g = wp.from_torch(self._g_th)
        self._e_th = torch.zeros(1, device=td)
        self.e = wp.from_torch(self._e_th)
        self._H_th = torch.zeros(n, n, device=td)
        self.H = wp.from_torch(self._H_th)

    def energy(self, x):
        t = wp.to_torch(x)
        self._e_th.copy_(torch.sqrt(1.0 + t * t).sum().view(1))
        return self.e

    def gradient(self, x):
        t = wp.to_torch(x)
        self._g_th.copy_(t / torch.sqrt(1.0 + t * t))
        return self.g

    def hessian(self, x):
        t = wp.to_torch(x)
        self._H_th.zero_()
        self._H_th.diagonal().copy_(torch.pow(1.0 + t * t, -1.5))
        return self.H


def _run(prob, x, buf, **kw):
    """Runs the solver on Warp's stream, which the solver requires on CUDA."""
    dev = wp.get_device(_device())
    if dev.is_cuda:
        with torch.cuda.stream(wp.stream_to_torch(dev)):
            return newtons_method_capturable(
                x, prob.energy, prob.gradient, prob.hessian, buf, **kw)
    return newtons_method_capturable(
        x, prob.energy, prob.gradient, prob.hessian, buf, **kw)


def test_quadratic_reaches_the_known_minimizer():
    """Newton on a quadratic: exact in one step, so any error is a real defect."""
    n, dev = 8, _device()
    prob = Quadratic(n, dev)
    buf = CapturableNewtonBuffers(n, device=dev)
    x = wp.zeros(n, dtype=wp.float32, device=dev)

    ptrs = (buf.dz.ptr, buf.bounded_direction.ptr, buf.lu_th.data_ptr(), x.ptr)
    _run(prob, x, buf, nm_max_iters=5)

    err = (wp.to_torch(x) - prob.solution).abs().max().item()
    assert err < 1e-4, f"did not reach A^-1 b: max error {err:.3e}"
    # Capture records addresses, so nothing may be reallocated along the way.
    assert (buf.dz.ptr, buf.bounded_direction.ptr, buf.lu_th.data_ptr(), x.ptr) == ptrs


def test_non_quadratic_converges():
    """Reaches the origin. conv_tol is what sets the floor, not the iteration count.

    With the default conv_tol=1e-4 this stops at ~7.7e-4 and stays there however many
    iterations it is given -- |g.dz| is already below tolerance. Tightening conv_tol is
    what buys accuracy, so the test states the tolerance it expects to be held to.
    """
    n, dev = 6, _device()
    prob = SoftAbs(n, dev)
    buf = CapturableNewtonBuffers(n, device=dev)
    x = wp.from_torch(torch.full((n,), 2.0, device=wp.device_to_torch(wp.get_device(dev))))

    _run(prob, x, buf, nm_max_iters=20, conv_tol=1e-8)

    assert wp.to_torch(x).abs().max().item() < 1e-8, "did not reach the origin"


def test_line_search_shortens_an_overshooting_step():
    """From x=2 the raw Newton step overshoots badly and must be cut back.

    Checked on the first iteration, not at the end: near the minimum the full step is
    fine and the search *grows* it, so the final ls_t says nothing about backtracking.
    """
    n, dev = 6, _device()
    prob = SoftAbs(n, dev)
    buf = CapturableNewtonBuffers(n, device=dev)
    x = wp.from_torch(torch.full((n,), 2.0, device=wp.device_to_torch(wp.get_device(dev))))

    _run(prob, x, buf, nm_max_iters=1)

    raw = wp.to_torch(buf.dz).abs().max().item()          # 10.0: x*(1+x^2) at x=2
    taken = wp.to_torch(buf.bounded_direction).abs().max().item()
    assert taken < raw * 0.5, f"step was not shortened: raw {raw:.3f}, taken {taken:.3f}"
    assert float(buf.ls_t.numpy()[0]) < 1.0


def test_line_search_grows_before_it_stops():
    """The first Armijo success grows the step rather than returning.

    Undocumented until now and easy to mistake for a bug, so it is pinned: from a point
    where the full Newton step is already acceptable, the accepted step size ends up
    above the initial 1.0 rather than at it.
    """
    n, dev = 4, _device()
    prob = SoftAbs(n, dev)
    buf = CapturableNewtonBuffers(n, device=dev)
    x = wp.from_torch(torch.full((n,), 0.05, device=wp.device_to_torch(wp.get_device(dev))))

    _run(prob, x, buf, nm_max_iters=1, max_ls_steps=10)

    assert float(buf.ls_t.numpy()[0]) > 1.0, (
        "expected the search to grow past the initial step size on a first success")


def test_gradient_buffer_must_survive_the_hessian_call():
    """A hessian_fcn that clobbers the gradient makes the solver stop, silently.

    This is the failure the docstring warns about: no exception, no NaN -- the solver
    reports convergence on iteration one and hands back its input unchanged.
    """
    n, dev = 6, _device()
    prob = SoftAbs(n, dev)
    buf = CapturableNewtonBuffers(n, device=dev)
    start = torch.full((n,), 2.0, device=wp.device_to_torch(wp.get_device(dev)))

    x_ok = wp.from_torch(start.clone())
    _run(prob, x_ok, buf, nm_max_iters=20, conv_tol=1e-8)
    assert wp.to_torch(x_ok).abs().max().item() < 1e-8

    class Clobbering(SoftAbs):
        def hessian(self, x):
            H = super().hessian(x)
            self._g_th.zero_()   # the buffer gradient() just returned
            return H

    bad = Clobbering(n, dev)
    buf2 = CapturableNewtonBuffers(n, device=dev)
    x_bad = wp.from_torch(start.clone())
    _run(bad, x_bad, buf2, nm_max_iters=20, conv_tol=1e-8)

    assert torch.allclose(wp.to_torch(x_bad), start), (
        "expected the clobbered run to stop immediately with x unchanged")


def test_bounds_none_and_bounds_fcn_returning_none_agree():
    n, dev = 6, _device()
    start = torch.full((n,), 2.0, device=wp.device_to_torch(wp.get_device(dev)))

    outs = []
    for bounds_fcn in (None, lambda dz, x: None):
        prob = SoftAbs(n, dev)
        buf = CapturableNewtonBuffers(n, device=dev)
        x = wp.from_torch(start.clone())
        _run(prob, x, buf, nm_max_iters=6, bounds_fcn=bounds_fcn)
        outs.append(wp.to_torch(x).clone())

    assert torch.equal(outs[0], outs[1])


def test_bounds_cap_the_step():
    """A constant bound of 0.5 must limit the update to half the raw direction."""
    n, dev = 6, _device()
    prob = SoftAbs(n, dev)
    buf = CapturableNewtonBuffers(n, device=dev)
    x = wp.from_torch(torch.full((n,), 2.0, device=wp.device_to_torch(wp.get_device(dev))))
    cap = wp.from_torch(torch.full((n,), 0.5, device=wp.device_to_torch(wp.get_device(dev))))

    _run(prob, x, buf, nm_max_iters=1, bounds_fcn=lambda dz, x_: cap)

    step = wp.to_torch(buf.bounded_direction).abs()
    raw = wp.to_torch(buf.dz).abs()
    assert torch.all(step <= raw * 0.5 + 1e-6)


class TestCallbackValidation:
    """Every malformed callback must say which one it is, not fail somewhere downstream."""

    def _fixture(self, n=4):
        dev = _device()
        return SoftAbs(n, dev), CapturableNewtonBuffers(n, device=dev), \
            wp.from_torch(torch.full((n,), 2.0, device=wp.device_to_torch(wp.get_device(dev))))

    def test_wrong_x_length(self):
        prob, buf, _ = self._fixture()
        bad = wp.zeros(9, dtype=wp.float32, device=_device())
        with pytest.raises(ValueError, match=r"^x has shape"):
            _run(prob, bad, buf)

    def test_gradient_returning_x_itself(self):
        prob, buf, x = self._fixture()
        prob.gradient = lambda z: z
        with pytest.raises(ValueError, match="gradient_fcn returned x itself"):
            _run(prob, x, buf)

    def test_energy_returning_a_float(self):
        prob, buf, x = self._fixture()
        prob.energy = lambda z: 1.0
        with pytest.raises(TypeError, match="energy_fcn must return a one-element"):
            _run(prob, x, buf)

    def test_hessian_wrong_shape(self):
        prob, buf, x = self._fixture()
        prob.hessian = lambda z: wp.zeros((9, 9), dtype=wp.float32, device=_device())
        with pytest.raises(ValueError, match="hessian_fcn returned shape"):
            _run(prob, x, buf)

    def test_hessian_not_a_dense_array(self):
        prob, buf, x = self._fixture()
        prob.hessian = lambda z: "not an array"
        with pytest.raises(TypeError, match="hessian_fcn must return a dense 2-D"):
            _run(prob, x, buf)


@cuda_only
def test_rejects_the_wrong_stream():
    """Torch work issued off Warp's stream is dropped from the graph with no error."""
    n = 4
    prob = SoftAbs(n, "cuda")
    buf = CapturableNewtonBuffers(n, device="cuda")
    x = wp.from_torch(torch.full((n,), 2.0, device="cuda"))

    with pytest.raises(RuntimeError, match="must run on Warp's stream"):
        newtons_method_capturable(x, prob.energy, prob.gradient, prob.hessian, buf)

    # Warp's own stream is accepted.
    with torch.cuda.stream(wp.stream_to_torch(wp.get_device("cuda"))):
        newtons_method_capturable(x, prob.energy, prob.gradient, prob.hessian, buf)


@cuda_only
def test_capture_and_replay_matches_and_allocates_nothing():
    """A replayed graph must give the uncaptured answer and allocate nothing.

    Replay is checked from a *different* starting point than the one recorded, so a graph
    that had baked in its answer instead of recomputing would be caught.
    """
    n = 6
    start_a = torch.full((n,), 2.0, device="cuda")
    start_b = torch.full((n,), -1.5, device="cuda")

    ref = []
    for start in (start_a, start_b):
        prob = SoftAbs(n, "cuda")
        buf = CapturableNewtonBuffers(n, device="cuda")
        x = wp.from_torch(start.clone())
        _run(prob, x, buf, nm_max_iters=8)
        ref.append(wp.to_torch(x).clone())

    prob = SoftAbs(n, "cuda")
    buf = CapturableNewtonBuffers(n, device="cuda")
    x = wp.from_torch(start_a.clone())
    graphs, pool = {}, None

    def step():
        newtons_method_capturable(x, prob.energy, prob.gradient, prob.hessian,
                                  buf, nm_max_iters=8)

    pool = capture_and_run_torch(step, "solve", graphs, device="cuda", pool=pool)
    assert torch.allclose(wp.to_torch(x), ref[0], atol=1e-5)

    # Replay from a different start: the graph must recompute, not reproduce.
    wp.to_torch(x).copy_(start_b)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    pool = capture_and_run_torch(step, "solve", graphs, device="cuda", pool=pool)
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()["allocation.all.allocated"]

    assert torch.allclose(wp.to_torch(x), ref[1], atol=1e-5)
    assert after == before, f"replay allocated {after - before} times"


@cuda_only
def test_uncaptured_run_works_through_the_same_entry_point():
    """capture_and_run_torch(captured=False) is the bisection path and must work.

    It used to call the function outside the stream redirection, so the solver's own
    stream check rejected it -- the documented workaround and the thing that raised were
    the same function.
    """
    n = 6
    prob = SoftAbs(n, "cuda")
    buf = CapturableNewtonBuffers(n, device="cuda")
    x = wp.from_torch(torch.full((n,), 2.0, device="cuda"))

    capture_and_run_torch(
        lambda: newtons_method_capturable(x, prob.energy, prob.gradient, prob.hessian,
                                          buf, nm_max_iters=20, conv_tol=1e-8),
        "solve", {}, captured=False, device="cuda")

    assert wp.to_torch(x).abs().max().item() < 1e-8
