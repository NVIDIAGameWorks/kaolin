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

r"""Tests for the CUDA-graph-capturable simulation path (``SimplicitsScene(capturable=True)``).

The headline claim is equivalence: a captured step must produce the same trajectory as
the host-side Newton solve. The rest of the tests cover the guards that turn
"silently different physics" into a clear error.
"""

import pytest
import torch
import warp as wp
import warp.sparse as wps

from kaolin.physics.utils.torch_utilities import hess_reduction

from kaolin.physics.simplicits import PhysicsPoints, SimplicitsObject, SimplicitsScene

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="capturable path requires CUDA")

# y translation given to object 0 by _make_scene; the animation test shifts relative to it.
_KIN_INIT_Y = 0.55


def _make_object(n_pts=600, num_handles=4, num_nodes=128, ym=1e6, seed=0):
    torch.manual_seed(seed)
    pts = torch.rand(n_pts, 3, device="cuda", dtype=torch.float32) - 0.5
    phys = PhysicsPoints(pts=pts, yms=ym, prs=0.45, rhos=500.0, appx_vol=1.0)
    return SimplicitsObject.create_with_rkpm(
        physics_points=phys, num_handles=num_handles,
        num_nodes=num_nodes, num_points=n_pts)


def _make_scene(sim_obj, capturable, num_objects=2, num_qp=96, max_ls_steps=10,
                is_kinematic=False, kinematic_ids=None, collisions=False, seed=0,
                **kwargs):
    r"""Build a test scene.

    Seeded, because add_object(num_qp=...) subsamples quadrature points at random.
    Two unseeded builds get different points and therefore different dynamics: measured
    at 1.0e-1 relative error between two runs of the *same* path over 15 steps, which is
    a thousand times the tolerance the comparisons below use. Any test that builds two
    scenes and compares them is otherwise comparing two different problems.

    Args:
        is_kinematic: shorthand for ``kinematic_ids=(0,)``.
        kinematic_ids: explicit set of kinematic object indices. Use this to cover
            configurations other than "object 0 only" -- multiple kinematic objects, a
            kinematic object that is not first, or an entirely kinematic scene.
        seed: fixes the quadrature subsampling. Two scenes built with the same seed have
            identical points; pass different seeds only if you want different geometry.
    """
    torch.manual_seed(seed)
    if kinematic_ids is None:
        kinematic_ids = (0,) if is_kinematic else ()
    kinematic_ids = set(kinematic_ids)

    scene = SimplicitsScene(device="cuda", timestep=0.03, max_newton_steps=4,
                            max_ls_steps=max_ls_steps, capturable=capturable, **kwargs)
    for i in range(num_objects):
        T = torch.eye(4, device="cuda", dtype=torch.float32)
        T[1, 3] = _KIN_INIT_Y + 1.2 * i
        scene.add_object(sim_obj, num_qp=num_qp, init_transform=T, apply_qr=False,
                         is_kinematic=(i in kinematic_ids))
    # +9.8, not -9.8: set_scene_gravity treats +y as down (its own default is
    # [0, 9.8, 0], documented as downward), because the energy is dot(g, x) * m and
    # motion follows -g. With -9.8 the objects drift upward away from the floor and
    # the floor term contributes nothing to the trajectory comparisons below.
    scene.set_scene_gravity(torch.tensor([0.0, 9.8, 0.0]))
    scene.set_scene_floor(floor_height=0.0, floor_axis=1,
                          floor_penalty=1e4, flip_floor=False)
    if collisions:
        scene.enable_collisions(collision_particle_radius=0.1,
                                collision_penalty=1000.0, max_contact_pairs=4096)
    return scene


def _trajectory(scene, n_steps):
    zs = []
    for _ in range(n_steps):
        scene.run_sim_step()
        zs.append(torch.as_tensor(scene.sim_z.numpy()).clone())
    return torch.stack(zs)


@cuda_only
@pytest.mark.parametrize("max_ls_steps", [10, 4])
def test_capturable_matches_host(max_ls_steps):
    r"""The captured step must reproduce the host Newton trajectory.

    Parametrized over max_ls_steps because newtons_method previously ignored it
    entirely (always using _line_search's default of 10), so the two paths diverged
    for any other value.
    """
    obj = _make_object()
    ref_scene = _make_scene(obj, False, max_ls_steps=max_ls_steps)
    cap_scene = _make_scene(obj, True, max_ls_steps=max_ls_steps)
    # The comparison only means anything if both scenes are the same scene. _make_scene
    # seeds the quadrature subsampling for this; without it two builds differ by ~1e-1,
    # a thousand times the tolerance below, and the test measures sampling noise.
    assert torch.equal(wp.to_torch(ref_scene.sim_pts), wp.to_torch(cap_scene.sim_pts))
    ref = _trajectory(ref_scene, 15)
    cap = _trajectory(cap_scene, 15)

    # Sanity: the scene must actually move, or matching is meaningless.
    assert ref.abs().max() > 1e-3

    rel = (ref - cap).abs().max() / ref.abs().max()
    assert rel < 1e-4, f"captured vs host relative error {rel:.3e}"


@cuda_only
def test_capturable_survives_empty_cache():
    r"""Regression: torch.cuda.empty_cache() must not corrupt a captured graph.

    Torch backends (cuSOLVER's LU workspace) allocate below the dispatcher and return
    the block to torch's ordinary cache when the call returns. Captured inside a graph,
    that address is baked in while owned by nothing, and empty_cache() frees it -- the
    next replay then wrote into unmapped memory (CUDA error 700). Fixed by capturing
    with a graph-private allocator pool.
    """
    scene = _make_scene(_make_object(), True)
    for _ in range(3):
        scene.run_sim_step()
    before = torch.as_tensor(scene.sim_z.numpy()).clone()

    torch.cuda.empty_cache()
    junk = [torch.randn(1024, 1024, device="cuda") for _ in range(4)]
    del junk
    torch.cuda.empty_cache()

    scene.run_sim_step()  # would raise CUDA 700 before the fix
    after = torch.as_tensor(scene.sim_z.numpy())
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after), "sim should have advanced"


@cuda_only
def test_kinematic_object_stays_pinned():
    r"""A kinematic object must not move under capture, while dynamic ones do.

    Kinematic DOFs are pinned by zeroing their Hessian rows/columns (unit diagonal) and
    their gradient entries, so the solve returns exactly zero for them. That is
    algebraically identical to the host path's sim_P/sim_Pt reduction, because
    create_projection_matrix builds P as a pure selection matrix.
    """
    scene = _make_scene(_make_object(), True, is_kinematic=True)
    kin_id, dyn_id = 0, 1
    y_kin0 = float(scene.get_object_deformed_pts(kin_id)[:, 1].mean())
    y_dyn0 = float(scene.get_object_deformed_pts(dyn_id)[:, 1].mean())

    for _ in range(20):
        scene.run_sim_step()

    y_kin1 = float(scene.get_object_deformed_pts(kin_id)[:, 1].mean())
    y_dyn1 = float(scene.get_object_deformed_pts(dyn_id)[:, 1].mean())

    assert abs(y_kin1 - y_kin0) < 1e-5, \
        f"kinematic object moved {abs(y_kin1 - y_kin0):.3e}"
    assert abs(y_dyn1 - y_dyn0) > 1e-2, \
        "dynamic object did not move, so pinning was not actually exercised"


@cuda_only
def test_pin_kinematic_dofs_zeroes_off_diagonal():
    r"""Direct unit test of the boundary-condition mask on a fully coupled matrix.

    This is the only test that currently distinguishes a correct
    ``pin_kinematic_dofs`` from one that zeroes just the diagonal. The scene-level
    kinematic tests cannot: with collisions off, ``_assemble_hessians_capturable``
    writes only the per-object diagonal blocks and ``BMB``/``reg*I`` are themselves
    block-diagonal, so ``H_kf`` is already zero and the off-diagonal term is a no-op.
    Inter-object contact is what makes it load-bearing, by producing genuine ``H_ij``
    blocks for ``i != j``.
    """
    from kaolin.physics.common.optimization_capturable import pin_kinematic_dofs

    n = 8
    kin = [1, 4, 5]
    free = [i for i in range(n) if i not in kin]

    torch.manual_seed(0)
    H_th = torch.randn(n, n, device="cuda", dtype=torch.float32) + 5.0 * torch.eye(
        n, device="cuda")
    H = wp.from_torch(H_th.contiguous())

    mask_th = torch.ones(n, device="cuda", dtype=torch.float32)
    mask_th[kin] = 0.0
    pin_kinematic_dofs(H, wp.from_torch(mask_th.contiguous()))

    out = wp.to_torch(H)
    assert torch.equal(out[kin][:, free], torch.zeros(len(kin), len(free), device="cuda")), \
        "kinematic ROWS not zeroed"
    assert torch.equal(out[free][:, kin], torch.zeros(len(free), len(kin), device="cuda")), \
        "kinematic COLUMNS not zeroed -- coupling survives, dz_k would not be zero"
    assert torch.equal(out[kin, kin], torch.ones(len(kin), device="cuda")), \
        "kinematic diagonal must be exactly 1"
    # The free-free block must be untouched.
    assert torch.equal(out[free][:, free], H_th[free][:, free])


@cuda_only
def test_kinematic_dofs_are_bitwise_zero_in_dz():
    r"""Pinned DOFs must not drift at all -- assert on the full DOF vector, not a scalar.

    The other pinning test collapses the state to mean y, which would miss a failure
    confined to the rotation/shear block or to x/z translation.
    """
    scene = _make_scene(_make_object(), True, is_kinematic=True)
    # sim_z.numpy() is on CPU, so the index tensor must be too.
    kin_dofs = wp.to_torch(scene.kin_obj_to_z_map[0]).long().cpu()
    z0 = torch.as_tensor(scene.sim_z.numpy()).clone()

    for _ in range(15):
        scene.run_sim_step()

    z1 = torch.as_tensor(scene.sim_z.numpy())
    assert torch.equal(z1[kin_dofs], z0[kin_dofs]), \
        f"kinematic DOFs drifted by up to {(z1[kin_dofs] - z0[kin_dofs]).abs().max():.3e}"
    free = torch.ones(z0.numel(), dtype=torch.bool)
    free[kin_dofs] = False
    assert (z1[free] - z0[free]).abs().max() > 1e-3, \
        "free DOFs did not move, so pinning was not actually exercised"


@cuda_only
@pytest.mark.xfail(strict=True, raises=RuntimeError, reason=(
    "Pre-existing host-path bug, not capture-related: an all-kinematic scene crashes at "
    "setup in warp_utilities._block_diagonalize with 'torch.cat(): expected a non-empty "
    "list of Tensors'. A kinematic object's dFdz is structurally all-zero, so when every "
    "object is kinematic every matrix has nnz==0 and nothing is collected to concatenate. "
    "Reproduces identically with capturable=False."))
def test_all_kinematic_scene():
    r"""Degenerate case: every DOF masked, so H is the identity and g is zero.

    Physically meaningless (nothing to solve) but it should fail cleanly rather than
    crash inside a matrix builder. Marked strict-xfail so this flips to a failure the
    moment the underlying bug is fixed.
    """
    scene = _make_scene(_make_object(), True, num_objects=2, kinematic_ids=(0, 1))
    scene.run_sim_step()


@cuda_only
@pytest.mark.parametrize("kinematic_ids", [(1,), (0, 1)],
                         ids=["not-first", "multiple"])
def test_kinematic_configurations(kinematic_ids):
    r"""Cover shapes other than 'object 0 only' -- a kinematic object that is not the
    leading contiguous DOF block, and more than one kinematic object."""
    scene = _make_scene(_make_object(), True, num_objects=3,
                        kinematic_ids=kinematic_ids)
    z0 = torch.as_tensor(scene.sim_z.numpy()).clone()
    for _ in range(10):
        scene.run_sim_step()
    z1 = torch.as_tensor(scene.sim_z.numpy())

    assert torch.isfinite(z1).all(), "non-finite DOFs"
    for obj_id in kinematic_ids:
        d = wp.to_torch(scene.kin_obj_to_z_map[obj_id]).long().cpu()
        assert torch.equal(z1[d], z0[d]), f"kinematic object {obj_id} moved"


@cuda_only
def test_capturable_matches_host_with_kinematic():
    r"""Equivalence must survive a kinematic object being present.

    Note the host path is internally inconsistent here: it evaluates gradient and
    Hessian at the full x including kinematic values, but _line_search evaluates energy
    at P @ x_red, with kinematic DOFs zeroed. For a separable energy that cancels in the
    Armijo test, since both f and f_new carry the same constant -- and every force the
    capturable path supports (gravity, floor, boundary, elastic, and the block-diagonal
    BMB kinetic term) is separable per object. This stops holding once inter-object
    collisions land, because contact couples kinematic and dynamic DOFs.

    A failure here means that argument is wrong; do not loosen the tolerance.
    """
    obj = _make_object()
    ref_scene = _make_scene(obj, False, is_kinematic=True)
    cap_scene = _make_scene(obj, True, is_kinematic=True)
    # The comparison only means anything if both scenes are the same scene. _make_scene
    # seeds the quadrature subsampling for this; without it two builds differ by ~1e-1,
    # a thousand times the tolerance below, and the test measures sampling noise.
    assert torch.equal(wp.to_torch(ref_scene.sim_pts), wp.to_torch(cap_scene.sim_pts))
    ref = _trajectory(ref_scene, 20)
    cap = _trajectory(cap_scene, 20)

    assert ref.abs().max() > 1e-3
    rel = (ref - cap).abs().max() / ref.abs().max()
    assert rel < 1e-4, f"captured vs host relative error {rel:.3e} with a kinematic object"


@cuda_only
def test_kinematic_object_can_be_animated_under_capture():
    r"""set_kinematic_object_transform must take effect without forcing a re-capture.

    It writes through a torch view of sim_z, so the store is in place and the captured
    graph -- which holds that pointer -- sees the new value on the next replay. This is
    the scripted-motion use case, and it only works because the setter no longer rebinds
    self.sim_z to a fresh wp.array afterwards.
    """
    scene = _make_scene(_make_object(), True, is_kinematic=True)
    for _ in range(3):
        scene.run_sim_step()
    graph_before = scene._sim_step_graph
    y0 = float(scene.get_object_deformed_pts(0)[:, 1].mean())

    # Assert on the *change* in y, not an absolute position. Transforms are relative to
    # the rest pose (standard_transform_to_relative), and the rest centroid is not exactly
    # zero -- torch.rand(n, 3) - 0.5 leaves a sampling offset of ~1e-3 at n=800 -- which
    # would otherwise show up as a spurious error of that size.
    SHIFT = 1.0
    T = torch.eye(4, device="cuda", dtype=torch.float32)
    T[1, 3] = _KIN_INIT_Y + SHIFT
    scene.set_kinematic_object_transform(0, T)

    for _ in range(5):
        scene.run_sim_step()

    y1 = float(scene.get_object_deformed_pts(0)[:, 1].mean())
    assert abs((y1 - y0) - SHIFT) < 1e-3, \
        f"kinematic object did not follow its scripted transform: moved {y1 - y0:.5f}, " \
        f"expected {SHIFT}"
    assert scene._sim_step_graph is graph_before, \
        "graph was re-captured despite only a kinematic transform changing"


@cuda_only
def test_collisions_with_qr_rejected():
    r"""QR plus collisions plus capture is the one combination that still cannot work.

    calculate_jacobian's QR branch rotates through a dense matmul and then
    _warp_csr_from_torch_dense -> torch.nonzero, which host-syncs, and the rotation
    destroys the Jacobian sparsity the chunked gather relies on. Raising beats silently
    producing a graph that bakes in one step's contact set.
    """
    obj = _make_object()
    scene = SimplicitsScene(device="cuda", timestep=0.03, max_newton_steps=4,
                            capturable=True)
    T = torch.eye(4, device="cuda", dtype=torch.float32)
    scene.add_object(obj, num_qp=96, init_transform=T, apply_qr=True)
    scene.set_scene_gravity(torch.tensor([0.0, 9.8, 0.0]))
    with pytest.raises(NotImplementedError, match="apply_qr"):
        scene.enable_collisions(collision_particle_radius=0.1,
                                collision_penalty=1000.0, max_contact_pairs=4096)


# The next two only exercise __init__ argument validation, which raises before the
# constructor touches CUDA -- so they run (and are worth running) on a GPU-less box.
def test_direct_solve_false_rejected():
    r"""A data-dependent CG iteration count is not capturable, so this must not be
    silently upgraded to a dense solve."""
    with pytest.raises(ValueError, match="direct_solve"):
        SimplicitsScene(device="cuda", capturable=True, direct_solve=False)


def test_use_cuda_graphs_conflict_rejected():
    with pytest.raises(ValueError, match="at most one"):
        SimplicitsScene(device="cuda", capturable=True, use_cuda_graphs=True)


@cuda_only
@pytest.mark.parametrize("attr,value", [
    ("timestep", 0.05),
    ("conv_tol", 1e-9),
    ("max_newton_steps", 9),
    ("max_ls_steps", 3),
    ("newton_hessian_regularizer", 1e-2),
])
def test_mutating_baked_scalar_raises(attr, value):
    r"""These are baked into the graph; a stale replay would silently use the old value."""
    scene = _make_scene(_make_object(), True)
    scene.run_sim_step()
    setattr(scene, attr, value)
    with pytest.raises(RuntimeError, match="baked into the graph"):
        scene.run_sim_step()


@cuda_only
def test_recapture_does_not_leak_pool_memory():
    r"""Repeated invalidate/re-capture cycles must not grow reserved memory.

    Capture-time torch allocations go into a private allocator pool so empty_cache()
    cannot free them out from under the graph. Minting a fresh pool per capture leaks:
    _cuda_endAllocateToPool only drops the stream filter, and nothing decrements
    PrivatePool::use_count, so the pool is never reclaimable. Measured at exactly
    2.00 MiB per re-capture before the fix, 0.00 after (one pool reused per scene).

    Reachable in practice because _invalidate_graphs() fires on every force setter --
    a gravity or floor slider re-captures each frame.
    """
    scene = _make_scene(_make_object(), True)
    scene.run_sim_step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_reserved()

    n = 6
    for i in range(n):
        # Changing a force invalidates the graph, forcing a re-capture next step.
        scene.set_scene_floor(floor_height=1e-6 * (i + 1), floor_axis=1,
                              floor_penalty=1e4, flip_floor=False)
        scene.run_sim_step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    growth_mib = (torch.cuda.memory_reserved() - baseline) / (1024.0 * 1024.0)
    # The leak was 2 MiB per re-capture; allow generous slack for allocator noise
    # while still failing decisively if per-recapture growth returns.
    assert growth_mib < 0.5 * n, (
        f"reserved memory grew {growth_mib:.2f} MiB over {n} re-captures "
        f"({growth_mib / n:.2f} MiB each) -- private pool is leaking")


@cuda_only
def test_force_setter_invalidates_graph():
    r"""Replacing a force struct must force a re-capture, not replay stale immediates."""
    scene = _make_scene(_make_object(), True)
    scene.run_sim_step()
    assert scene._sim_step_graph is not None, "expected a cached graph after the first step"

    scene.set_scene_floor(floor_height=0.5, floor_axis=1,
                          floor_penalty=1e4, flip_floor=False)
    assert scene._sim_step_graph is None, "force setter must clear the graph cache"

    scene.run_sim_step()
    assert torch.isfinite(torch.as_tensor(scene.sim_z.numpy())).all()


@cuda_only
def test_newton_loop_exits_early_on_convergence():
    r"""The captured loop must exit on convergence rather than run a fixed trip count.

    ``0 < iters <= max_newton_steps`` would also hold for a fixed-trip-count loop, so it
    proves nothing. Requiring a *strict* early exit is what distinguishes
    ``wp.capture_while`` driven by a device-side convergence flag from a plain loop:
    delete the ``exit_while``/``nm_if_cond`` branch and ``nm_step_count`` pins to
    ``max_newton_steps``, failing this.
    """
    scene = _make_scene(_make_object(ym=1e4), True, num_objects=1)
    scene.max_newton_steps  # noqa: B018  (documents what the bound is)
    for _ in range(4):
        scene.run_sim_step()
    iters = int(scene.newton_buffers.nm_step_count.numpy()[0])
    assert 0 < iters < scene.max_newton_steps, (
        f"expected early convergence exit, got {iters} of "
        f"{scene.max_newton_steps} iterations")


@cuda_only
def test_newton_iterations_scale_with_difficulty():
    r"""Iteration count must respond to the problem, confirming genuine data dependence."""
    def iters_for(ym, dt, conv_tol):
        # Seeded like _make_scene: add_object subsamples quadrature points at random,
        # and an iteration count should not move because the sampling did.
        torch.manual_seed(0)
        scene = SimplicitsScene(device="cuda", timestep=dt, max_newton_steps=8,
                                max_ls_steps=10, conv_tol=conv_tol, capturable=True)
        obj = _make_object(ym=ym)
        T = torch.eye(4, device="cuda", dtype=torch.float32)
        T[1, 3] = 0.55
        scene.add_object(obj, num_qp=96, init_transform=T, apply_qr=False)
        # +9.8 is downward here; see _make_scene. With -9.8 the object rose away from
        # the floor at y=0 and the floor term did nothing. Iteration counts are the same
        # either way (1 and 8, measured) since the difficulty comes from ym/dt/conv_tol,
        # but the scene should still describe what it claims to.
        scene.set_scene_gravity(torch.tensor([0.0, 9.8, 0.0]))
        scene.set_scene_floor(floor_height=0.0, floor_axis=1,
                              floor_penalty=1e4, flip_floor=False)
        for _ in range(4):
            scene.run_sim_step()
        return int(scene.newton_buffers.nm_step_count.numpy()[0])

    easy = iters_for(ym=1e4, dt=0.01, conv_tol=1e-3)
    hard = iters_for(ym=1e8, dt=0.20, conv_tol=1e-12)
    assert hard > easy, f"expected harder scene to need more iterations, got {hard} vs {easy}"


@cuda_only
def test_singular_hessian_raises_and_rolls_back():
    r"""check_solve_info must raise AND leave the scene state untouched.

    The reference path raises from inside newtons_method before assigning, so state is
    clean. The captured graph has already overwritten sim_z/sim_z_dot in place by the
    time the host reads the info code, so it must roll back explicitly -- otherwise a
    caller's halve-timestep-and-retry loop resumes from a NaN scene.
    """
    scene = _make_scene(_make_object(), True)
    scene.run_sim_step()
    good_z = torch.as_tensor(scene.sim_z.numpy()).clone()
    good_zdot = torch.as_tensor(scene.sim_z_dot.numpy()).clone()

    # Poison the Hessian. It must be a *device buffer the captured graph reads* --
    # patching Python-side has no effect once the graph is recorded, and _eval_H_dense_th
    # is recomputed inside the graph every iteration. _sim_BMB_plus_reg_dense_th is a
    # constant the graph adds in, so it survives. NaN rather than zero: zeroing only
    # removes mass + regularizer and the elastic block may still factor, whereas
    # cuSOLVER reliably reports nonzero info for NaN input.
    scene._sim_BMB_plus_reg_dense_th.fill_(float("nan"))

    with pytest.raises(torch.linalg.LinAlgError, match="LU factorization"):
        scene.run_sim_step()

    after_z = torch.as_tensor(scene.sim_z.numpy())
    after_zdot = torch.as_tensor(scene.sim_z_dot.numpy())
    assert torch.isfinite(after_z).all(), "sim_z left non-finite after a failed step"
    assert torch.isfinite(after_zdot).all(), "sim_z_dot left non-finite after a failed step"
    assert torch.equal(after_z, good_z), "sim_z was not rolled back"
    assert torch.equal(after_zdot, good_zdot), "sim_z_dot was not rolled back"


@cuda_only
def test_check_solve_info_false_skips_the_check():
    r"""With the check off the step must not raise, even on a singular Hessian."""
    scene = _make_scene(_make_object(), True, check_solve_info=False)
    scene.run_sim_step()
    scene._sim_BMB_plus_reg_dense_th.fill_(float("nan"))
    scene.run_sim_step()  # must not raise; result is garbage by design


def _contact_scene(capturable, n_obj=3, gap=0.9, radius=0.1, kinematic_ids=(),
                   max_contact_pairs=4096, **scene_kwargs):
    r"""Scene whose objects actually touch, unlike ``_make_scene``'s 1.2-spaced stack.

    Gravity is ``+9.8`` because ``set_scene_gravity`` treats +y as down, so the objects
    settle onto the floor and into each other instead of drifting apart.

    The radius/gap defaults are deliberately milder than they could be. Pushed harder
    (radius 0.15 at gap 0.75) the scene reaches roughly 1400 contacts but the *host*
    solve then hits a singular Hessian in about one run in four, which would make any
    test built on it flaky for reasons unrelated to capture.
    """
    obj = _make_object()
    scene = SimplicitsScene(device="cuda", timestep=0.03, max_newton_steps=4,
                            max_ls_steps=10, capturable=capturable, **scene_kwargs)
    for i in range(n_obj):
        T = torch.eye(4, device="cuda", dtype=torch.float32)
        T[1, 3] = 0.6 + gap * i
        scene.add_object(obj, num_qp=96, init_transform=T, apply_qr=False,
                         is_kinematic=(i in kinematic_ids))
    scene.set_scene_gravity(torch.tensor([0.0, 9.8, 0.0]))
    scene.set_scene_floor(floor_height=0.0, floor_axis=1, floor_penalty=1e4,
                          flip_floor=False)
    scene.enable_collisions(collision_particle_radius=radius,
                            collision_penalty=1000.0,
                            max_contact_pairs=max_contact_pairs)
    return scene


@cuda_only
def test_capturable_collision_assembly_matches_host():
    r"""Gradient, Hessian and step bounds must match the host on the *same* contacts.

    Trajectory comparison cannot establish this. Contact detection compacts with
    ``wp.atomic_add``, so two runs from an identical state produce the same contact set
    in a different storage order, and the energy reduction over that permuted order
    differs in the last bits -- which flips borderline Armijo decisions. Two host runs
    diverge from each other as fast as host and captured do.

    So this drives both assemblies from one Collision object with one frozen detection.
    Everything measured here is then attributable to assembly alone.
    """
    scene = _contact_scene(True)
    for _ in range(3):
        scene.run_sim_step()

    cs = scene.force_dict["collision"]["object"]
    coeff = scene.force_dict["collision"]["coeff"]
    num_contacts = cs.num_contacts
    assert num_contacts > 20, f"only {num_contacts} contacts; scene is not in contact"

    # The captured path skips this (it gathers from the dense basis instead), so build
    # the sparse Jacobian explicitly to give the host formulas something to read.
    cs.calculate_jacobian(cp_w=scene.sim_skinning_weights, cp_x0=scene.sim_pts,
                          cp_is_static=scene.qp_is_kinematic, qr_tfm=None)
    J_dense = cs.collision_J_dense
    wps.bsr_mv(A=scene.sim_B, x=scene.sim_z, y=scene._eval_dx)

    # ---- Hessian: J^T H J ----
    cs.hessian(scene._eval_dx, scene.sim_pts, coeff,
               hessian_blocks=scene._cap_collision_hess)
    h_blocks = wp.to_torch(scene._cap_collision_hess)[:num_contacts]
    want_H = hess_reduction(J_dense, h_blocks)
    got_H = torch.zeros_like(want_H)
    scene._cap_collision_reducer.reduce_capturable(
        scene._cap_B_dense, scene._cap_collision_hess, got_H)

    # One monolithic GEMM versus a sum of per-chunk GEMMs: same products, different
    # accumulation order, so the bound is float32 epsilon scaled by the term count.
    tol_H = torch.finfo(torch.float32).eps * (3 * num_contacts) * float(want_H.abs().max())
    err_H = (want_H - got_H).abs().max().item()
    assert err_H <= tol_H, f"collision Hessian differs by {err_H:.3e} (bound {tol_H:.3e})"

    # ---- Gradient: J^T dE/dx. Exact: one GEMV either way, no reassociation. ----
    cs.gradient(scene._eval_dx, scene.sim_pts, coeff,
                gradient=scene._cap_collision_dEdx)
    dEdx = wp.to_torch(scene._cap_collision_dEdx)[:num_contacts].reshape(-1)
    want_g = J_dense.transpose(0, 1) @ dEdx
    got_g = torch.zeros(scene._num_dofs, device=want_g.device, dtype=want_g.dtype)
    scene._cap_collision_reducer.accumulate_gradient_capturable(
        scene._cap_B_dense, scene._cap_collision_dEdx, got_g)
    assert torch.equal(want_g, got_g), "collision gradient is not bit-identical"

    # ---- Step bounds. Also exact: the same atomic_min over the same block set. ----
    # The clamp only engages for a step that closes more than 0.375 of the current gap,
    # so the direction is scaled up until it does. Asserting equality on an all-ones
    # bounds vector would pass against any implementation at all.
    torch.manual_seed(3)
    direction = torch.randn(scene._num_dofs, device="cuda")
    wps.bsr_mv(A=scene.sim_B, x=scene.sim_z, y=scene._cap_bounds_dx)

    want_b = got_b = None
    for scale in (0.05, 0.2, 1.0, 5.0, 25.0):
        dz = wp.from_torch((direction * scale).contiguous())
        wps.bsr_mv(A=scene.sim_B, x=dz, y=scene._cap_bounds_delta_dx)
        want_b = wp.to_torch(cs.get_bounds(cp_delta_dx=scene._cap_bounds_delta_dx,
                                           cp_dx=scene._cap_bounds_dx,
                                           cp_x0=scene.sim_pts)).clone()
        got_b = wp.to_torch(cs.get_bounds_capturable(
            cp_delta_dx=scene._cap_bounds_delta_dx, cp_dx=scene._cap_bounds_dx,
            b_dense=scene._cap_B_dense, dof_step_bounds=scene._cap_dof_step_bounds)).clone()
        if bool((want_b < 1.0).any()):
            break

    assert bool((want_b < 1.0).any()), \
        "no step size clamped any DOF; the bounds comparison would be vacuous"
    assert torch.equal(want_b, got_b), "capturable step bounds differ from the host's"


@cuda_only
def test_capturable_collisions_preserve_contact_invariants():
    r"""Long-run invariants, which is as much as a contact trajectory can be held to.

    A tolerance-based trajectory comparison against the host is not a meaningful test
    here and this docstring is the record of why: contact detection compacts slots with
    ``wp.atomic_add``, so the storage order varies run to run, the energy reduction over
    that order differs in the last bits, and stiff penalty contact amplifies it. Measured
    on this scene, two *host* runs six steps apart disagree by up to 9e-2 relative -- more
    than host-versus-captured does -- and the contact count swings between 40 and 96.

    What must hold regardless of ordering is checked instead. Assembly correctness is
    covered separately, and exactly, by
    :func:`test_capturable_collision_assembly_matches_host`.
    """
    scene = _contact_scene(True, n_obj=3)
    cs = scene.force_dict["collision"]["object"]
    seen_contacts = []

    for step in range(12):
        scene.run_sim_step()
        z = wp.to_torch(scene.sim_z)
        seen_contacts.append(cs.num_contacts)

        assert torch.isfinite(z).all(), f"step {step + 1}: non-finite DOFs"
        assert float(z.abs().max()) < 1e3, \
            f"step {step + 1}: |z| blew up to {float(z.abs().max()):.3e}"
        # Saturating capacity would silently drop contacts rather than error.
        assert cs.num_contacts < cs.max_contacting_pairs, \
            f"step {step + 1}: contact capacity saturated"

    assert max(seen_contacts) > 20, \
        f"scene never made meaningful contact (max {max(seen_contacts)})"

    # No interpenetration: every detected pair must stay outside the impenetrable
    # barrier. This is what the step bounds exist to guarantee, so it fails if
    # bounds_fcn is dropped from the captured Newton.
    num_contacts = cs.num_contacts
    if num_contacts > 0:
        pts = wp.to_torch(scene.sim_pts) + wp.to_torch(scene._eval_dx)
        ia = wp.to_torch(cs.collision_indices_a[:num_contacts]).long()
        ib = wp.to_torch(cs.collision_indices_b[:num_contacts]).long()
        live = (ia >= 0) & (ib >= 0)
        d = (pts[ia[live]] - pts[ib[live]]).norm(dim=1)
        barrier = cs.collision_radius * cs.collision_barrier_ratio
        assert float(d.min()) > 0.0, "contact points coincide exactly"
        assert float(d.min()) >= barrier * 0.5, (
            f"interpenetration: closest pair {float(d.min()):.4f} is well inside the "
            f"{barrier:.4f} barrier")


@cuda_only
def test_capturable_kinematic_object_pinned_while_in_contact():
    r"""A kinematic object in contact must not move, which needs the off-diagonal mask.

    This is the test the kinematic work could not have: with only floor and elastic
    forces the scene Hessian is block diagonal per object, so ``H_kf`` is already zero
    and ``pin_kinematic_dofs``'s off-diagonal zeroing is provably a no-op. Contact is
    what finally assembles genuine object-object blocks, so zeroing rows *and* columns
    becomes load-bearing: keep only the diagonal and the kinematic DOFs get dragged by
    whatever is resting on them.
    """
    scene = _contact_scene(True, n_obj=3, kinematic_ids=(1,))
    kin_dofs = wp.to_torch(scene.kin_obj_to_z_map[1]).cpu()
    z0 = wp.to_torch(scene.sim_z).cpu().clone()

    for _ in range(5):
        scene.run_sim_step()

    cs = scene.force_dict["collision"]["object"]
    assert cs.num_contacts > 20, "kinematic object is not actually in contact"

    z1 = wp.to_torch(scene.sim_z).cpu()
    moved = (z1[kin_dofs] - z0[kin_dofs]).abs().max().item()
    assert moved == 0.0, f"kinematic DOFs moved by {moved:.3e} while in contact"
    # And the rest of the scene must have moved, or "pinned" is trivially satisfied.
    free = torch.ones(z0.numel(), dtype=torch.bool)
    free[kin_dofs] = False
    assert (z1[free] - z0[free]).abs().max().item() > 1e-4


@cuda_only
def test_capturable_contact_step_does_no_host_sync():
    r"""A captured contact step must not block on the device.

    Contact used to cost three D2H per step: ``torch.unique(..., dim=0)`` and the
    following ``.cpu()`` / ``.numpy()`` in the ``object_pairs`` build (whose only consumer
    is the *host* Newton path's sparse block assembly), plus an eager readback of the
    contact count. None of the three fed anything the captured step used.

    ``check_solve_info=False`` because that check reads ``solve_info`` back deliberately
    and documents itself as costing a sync; it is opt-out, not part of the contact path.

    Caveat: ``set_sync_debug_mode`` instruments torch, so it catches the ``torch.unique``
    / ``.cpu()`` syncs directly but would not see a Warp-side ``.numpy()``. The count
    readback is covered instead by ``num_contacts`` being a property -- nothing on this
    path reads it.
    """
    scene = _contact_scene(True, n_obj=3, check_solve_info=False)
    for _ in range(3):
        scene.run_sim_step()

    collision = scene.force_dict["collision"]["object"]
    assert collision.num_contacts > 20, "scene is not in contact; this would prove nothing"

    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        scene.run_sim_step()
    except RuntimeError as e:  # pragma: no cover - only on regression
        raise AssertionError(
            f"captured contact step performed a synchronizing CUDA operation: {e}")
    finally:
        torch.cuda.set_sync_debug_mode("default")


@cuda_only
def test_object_pairs_skipped_when_capturable():
    r"""The host-only ``object_pairs`` list is not built for a captured scene.

    It exists to drive ``_assemble_hessians``'s sparse per-pair block assembly, which the
    capturable path replaces with a full-width reduction. Building it anyway was the
    source of two of the three per-step syncs.
    """
    host = _contact_scene(False, n_obj=3)
    host.run_sim_step()
    assert len(host.force_dict["collision"]["object"].object_pairs) > 0, \
        "host path must still build object_pairs -- _assemble_hessians reads it"

    cap = _contact_scene(True, n_obj=3)
    cap.run_sim_step()
    cs = cap.force_dict["collision"]["object"]
    assert cs.num_contacts > 20
    assert len(cs.object_pairs) == 0, \
        "capturable scene built object_pairs despite having no consumer for it"
