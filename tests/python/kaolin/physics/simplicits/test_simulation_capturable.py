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
                is_kinematic=False, kinematic_ids=None, collisions=False, **kwargs):
    r"""Build a test scene.

    Args:
        is_kinematic: shorthand for ``kinematic_ids=(0,)``.
        kinematic_ids: explicit set of kinematic object indices. Use this to cover
            configurations other than "object 0 only" -- multiple kinematic objects, a
            kinematic object that is not first, or an entirely kinematic scene.
    """
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
    scene.set_scene_gravity(torch.tensor([0.0, -9.8, 0.0]))
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
    ref = _trajectory(_make_scene(obj, False, max_ls_steps=max_ls_steps), 15)
    cap = _trajectory(_make_scene(obj, True, max_ls_steps=max_ls_steps), 15)

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
def test_apply_kinematic_bc_zeroes_off_diagonal():
    r"""Direct unit test of the boundary-condition mask on a fully coupled matrix.

    This is the only test that currently distinguishes a correct
    ``apply_kinematic_bc`` from one that zeroes just the diagonal. The scene-level
    kinematic tests cannot: with collisions off, ``_assemble_hessians_capturable``
    writes only the per-object diagonal blocks and ``BMB``/``reg*I`` are themselves
    block-diagonal, so ``H_kf`` is already zero and the off-diagonal term is a no-op.
    Inter-object contact is what makes it load-bearing, by producing genuine ``H_ij``
    blocks for ``i != j``.
    """
    from kaolin.physics.common.optimization_capturable import apply_kinematic_bc

    n = 8
    kin = [1, 4, 5]
    free = [i for i in range(n) if i not in kin]

    torch.manual_seed(0)
    H_th = torch.randn(n, n, device="cuda", dtype=torch.float32) + 5.0 * torch.eye(
        n, device="cuda")
    H = wp.from_torch(H_th.contiguous())

    mask_th = torch.ones(n, device="cuda", dtype=torch.float32)
    mask_th[kin] = 0.0
    apply_kinematic_bc(H, wp.from_torch(mask_th.contiguous()))

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
@pytest.mark.xfail(strict=True, reason=(
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
    ref = _trajectory(_make_scene(obj, False, is_kinematic=True), 20)
    cap = _trajectory(_make_scene(obj, True, is_kinematic=True), 20)

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
    graph_before = dict(scene._graph_dict)
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
    assert scene._graph_dict.keys() == graph_before.keys(), "should not have re-captured"
    assert all(scene._graph_dict[k][0] is graph_before[k][0] for k in graph_before), \
        "graph was re-captured despite only a kinematic transform changing"


@cuda_only
def test_collisions_rejected():
    r"""Inter-object collision detection host-syncs on the contact count."""
    scene = _make_scene(_make_object(), True, collisions=True)
    with pytest.raises(NotImplementedError, match="collision"):
        scene.run_sim_step()


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
    assert scene._graph_dict, "expected a cached graph after the first step"

    scene.set_scene_floor(floor_height=0.5, floor_axis=1,
                          floor_penalty=1e4, flip_floor=False)
    assert not scene._graph_dict, "force setter must clear the graph cache"

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
    iters = int(scene._nm_buf.nm_step_count.numpy()[0])
    assert 0 < iters < scene.max_newton_steps, (
        f"expected early convergence exit, got {iters} of "
        f"{scene.max_newton_steps} iterations")


@cuda_only
def test_newton_iterations_scale_with_difficulty():
    r"""Iteration count must respond to the problem, confirming genuine data dependence."""
    def iters_for(ym, dt, conv_tol):
        scene = SimplicitsScene(device="cuda", timestep=dt, max_newton_steps=8,
                                max_ls_steps=10, conv_tol=conv_tol, capturable=True)
        obj = _make_object(ym=ym)
        T = torch.eye(4, device="cuda", dtype=torch.float32)
        T[1, 3] = 0.55
        scene.add_object(obj, num_qp=96, init_transform=T, apply_qr=False)
        scene.set_scene_gravity(torch.tensor([0.0, -9.8, 0.0]))
        scene.set_scene_floor(floor_height=0.0, floor_axis=1,
                              floor_penalty=1e4, flip_floor=False)
        for _ in range(4):
            scene.run_sim_step()
        return int(scene._nm_buf.nm_step_count.numpy()[0])

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
