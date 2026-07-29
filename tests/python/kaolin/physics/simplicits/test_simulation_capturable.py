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

from kaolin.physics.simplicits import PhysicsPoints, SimplicitsObject, SimplicitsScene

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="capturable path requires CUDA")


def _make_object(n_pts=600, num_handles=4, num_nodes=128, ym=1e6, seed=0):
    torch.manual_seed(seed)
    pts = torch.rand(n_pts, 3, device="cuda", dtype=torch.float32) - 0.5
    phys = PhysicsPoints(pts=pts, yms=ym, prs=0.45, rhos=500.0, appx_vol=1.0)
    return SimplicitsObject.create_with_rkpm(
        physics_points=phys, num_handles=num_handles,
        num_nodes=num_nodes, num_points=n_pts)


def _make_scene(sim_obj, capturable, num_objects=2, num_qp=96, max_ls_steps=10,
                is_kinematic=False, collisions=False, **kwargs):
    scene = SimplicitsScene(device="cuda", timestep=0.03, max_newton_steps=4,
                            max_ls_steps=max_ls_steps, capturable=capturable, **kwargs)
    for i in range(num_objects):
        T = torch.eye(4, device="cuda", dtype=torch.float32)
        T[1, 3] = 0.55 + 1.2 * i
        scene.add_object(sim_obj, num_qp=num_qp, init_transform=T, apply_qr=False,
                         is_kinematic=(is_kinematic and i == 0))
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
def test_kinematic_objects_rejected():
    r"""Kinematic DOFs need the P/Pt projection, which the capturable path lacks."""
    scene = _make_scene(_make_object(), True, is_kinematic=True)
    with pytest.raises(NotImplementedError, match="kinematic"):
        scene.run_sim_step()


@cuda_only
def test_collisions_rejected():
    r"""Inter-object collision detection host-syncs on the contact count."""
    scene = _make_scene(_make_object(), True, collisions=True)
    with pytest.raises(NotImplementedError, match="collision"):
        scene.run_sim_step()


@cuda_only
def test_direct_solve_false_rejected():
    r"""A data-dependent CG iteration count is not capturable, so this must not be
    silently upgraded to a dense solve."""
    with pytest.raises(ValueError, match="direct_solve"):
        SimplicitsScene(device="cuda", capturable=True, direct_solve=False)


@cuda_only
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
def test_newton_loop_is_data_dependent():
    r"""The captured loop must exit on convergence, not run a fixed trip count."""
    scene = _make_scene(_make_object(), True)
    for _ in range(4):
        scene.run_sim_step()
    iters = int(scene._nm_buf.nm_step_count.numpy()[0])
    assert 0 < iters <= scene.max_newton_steps
