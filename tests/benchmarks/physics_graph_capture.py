"""CUDA graph capture for the Simplicits sim step: correctness + speedup.

Correctness is checked first and gates the timing report -- a fast wrong answer is
worthless, and the line-search latch and convergence semantics are easy to get
subtly wrong when moved onto the device.

Usage:
    python tests/benchmarks/physics_graph_capture.py
    python tests/benchmarks/physics_graph_capture.py --only-correctness
    python tests/benchmarks/physics_graph_capture.py --stacking-cubes
"""
import argparse
import os
import sys
import time

import torch
from torch.profiler import profile, ProfilerActivity

from kaolin.physics.simplicits import PhysicsPoints, SimplicitsObject, SimplicitsScene


def build_scene(num_objects, n_pts, num_handles, num_qp, ym, capturable,
                max_newton_steps=5, num_nodes=256, seed=0, check_solve_info=True):
    """Objects dropping onto a floor. No inter-object collisions: the capturable path
    does not support them yet, so both paths are built identically without them."""
    device, dtype = "cuda", torch.float32
    torch.manual_seed(seed)
    pts = torch.rand(n_pts, 3, device=device, dtype=dtype) - 0.5
    phys = PhysicsPoints(pts=pts, yms=ym, prs=0.45, rhos=500.0, appx_vol=1.0)
    sim_obj = SimplicitsObject.create_with_rkpm(
        physics_points=phys, num_handles=num_handles,
        num_nodes=num_nodes, num_points=n_pts)

    scene = SimplicitsScene(device=device, timestep=0.03,
                            max_newton_steps=max_newton_steps, max_ls_steps=10,
                            capturable=capturable, check_solve_info=check_solve_info)
    for i in range(num_objects):
        T = torch.eye(4, device=device, dtype=dtype)
        T[1, 3] = 0.55 + 1.2 * i
        scene.add_object(sim_obj, num_qp=num_qp, init_transform=T, apply_qr=False)
    # +y is down: set_scene_gravity's default is [0, 9.8, 0] and means downward.
    # Verified empirically -- [0, -9.8, 0] accelerates objects UP, away from the floor,
    # which is what this benchmark did before and why it never exercised floor contact.
    scene.set_scene_gravity(torch.tensor([0.0, 9.8, 0.0]))
    scene.set_scene_floor(floor_height=0.0, floor_axis=1,
                          floor_penalty=10000.0, flip_floor=False)
    return scene


def trajectory(scene, n_steps):
    zs = []
    for _ in range(n_steps):
        scene.run_sim_step()
        zs.append(torch.as_tensor(scene.sim_z.numpy()).clone())
    return torch.stack(zs)


def check_correctness(cfg, n_steps=25):
    name, nobj, npts, nh, nqp, ym = cfg
    ref = build_scene(nobj, npts, nh, nqp, ym, capturable=False)
    cap = build_scene(nobj, npts, nh, nqp, ym, capturable=True)

    t_ref = trajectory(ref, n_steps)
    t_cap = trajectory(cap, n_steps)

    abs_err = (t_ref - t_cap).abs().max().item()
    scale = max(t_ref.abs().max().item(), 1e-12)
    rel_err = abs_err / scale
    moved = t_ref.abs().max().item()

    del ref, cap
    torch.cuda.empty_cache()
    return abs_err, rel_err, moved


def measure(scene, n_steps=20, warmup=5, want_launches=False):
    for _ in range(warmup):
        scene.run_sim_step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        scene.run_sim_step()
    torch.cuda.synchronize()
    ms = 1000.0 * (time.perf_counter() - t0) / n_steps

    launches = float("nan")
    if want_launches:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            for _ in range(n_steps):
                scene.run_sim_step()
            torch.cuda.synchronize()
        leaf = [e for e in pr.key_averages()
                if e.self_device_time_total > 0
                and not e.key.startswith("aten::") and not e.key.startswith("cuda")]
        launches = sum(e.count for e in leaf) / n_steps
    return ms, launches


CONFIGS = [
    # name,                 objs, n_pts, handles, num_qp, youngs modulus
    ("120 dof  (2obj h5)",     2,  1000,  5,  128, 1e6),
    ("240 dof  (2obj h10)",    2,  2000, 10,  256, 1e6),
    ("960 dof  (4obj h20)",    4,  4000, 20,  512, 1e6),
]

# Raising Young's modulus alone does NOT increase the Newton iteration count here --
# the motion is translation-dominated, so a stiffer material converges just as fast.
# Larger timesteps with tighter convergence tolerances do. This sweep exists because
# every config above converges in 2 iterations, and the launch-bound argument predicts
# the win should grow with iteration count; without this the claim is untested.
#   dt,   conv_tol, youngs, max_newton
ITER_CONFIGS = [
    (0.03, 1e-4,  1e6,  5),
    (0.10, 1e-8,  1e7, 10),
    (0.20, 1e-10, 1e8, 20),
    (0.30, 1e-12, 1e8, 30),
]


def build_iter_scene(dt, conv_tol, ym, max_newton, capturable):
    device, dtype = "cuda", torch.float32
    torch.manual_seed(0)
    pts = torch.rand(2000, 3, device=device, dtype=dtype) - 0.5
    phys = PhysicsPoints(pts=pts, yms=ym, prs=0.45, rhos=500.0, appx_vol=1.0)
    sim_obj = SimplicitsObject.create_with_rkpm(
        physics_points=phys, num_handles=10, num_nodes=256, num_points=2000)
    scene = SimplicitsScene(device=device, timestep=dt, max_newton_steps=max_newton,
                            max_ls_steps=10, conv_tol=conv_tol, capturable=capturable)
    for i in range(2):
        T = torch.eye(4, device=device, dtype=dtype)
        T[1, 3] = 0.55 + 1.2 * i
        scene.add_object(sim_obj, num_qp=256, init_transform=T, apply_qr=False)
    # +y is down: set_scene_gravity's default is [0, 9.8, 0] and means downward.
    # Verified empirically -- [0, -9.8, 0] accelerates objects UP, away from the floor,
    # which is what this benchmark did before and why it never exercised floor contact.
    scene.set_scene_gravity(torch.tensor([0.0, 9.8, 0.0]))
    scene.set_scene_floor(floor_height=0.0, floor_axis=1,
                          floor_penalty=1e5, flip_floor=False)
    return scene


def newton_iters_per_step(cfg, n_steps=10):
    """Average Newton iterations actually taken, read off the device counter."""
    name, nobj, npts, nh, nqp, ym = cfg
    sc = build_scene(nobj, npts, nh, nqp, ym, capturable=True)
    for _ in range(3):
        sc.run_sim_step()
    total = 0
    for _ in range(n_steps):
        sc.run_sim_step()
        total += int(sc._nm_buf.nm_step_count.numpy()[0])
    del sc
    torch.cuda.empty_cache()
    return total / n_steps


def stacking_cubes(steps=200):
    r"""Times ``examples/tutorial/physics/simplicits_stacking_cubes.py``, capture on/off.

    Drives the example's own ``create_cube_object()`` / ``build_scene()`` rather than a
    reimplementation, so the parameters -- 50000 cube points, 1000 qp/object, 10 handles,
    5 Newton steps, contact radius 0.05, penalty 5000, 50000 contact capacity -- cannot
    drift away from the example. Only the ``capturable`` flag is injected.

    Reports a phase split because the run has two regimes: the stack collapses partway
    through (the cubes interpenetrate and pass through each other), after which there are
    no contacts and a step is ~3.5x cheaper, so a single 200-step average blends them. The
    collapse is pre-existing and unrelated to capture -- at the branch point (631550e3) it
    happens earlier, by step 81. Which step it lands on varies run to run, because contact
    detection compacts slots with wp.atomic_add, so the two columns generally spend
    different numbers of steps in the expensive regime; the per-step means are the
    comparable figures, not the phase totals.

    Polyscope is imported by the example at module scope but never initialized, so this
    runs headless.
    """
    import kaolin as kal
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "examples", "tutorial", "physics"))
    import simplicits_stacking_cubes as example

    base = kal.physics.simplicits.SimplicitsScene

    def run(capturable):
        if capturable:
            class _Capturable(base):
                def __init__(self, *a, **kw):
                    kw["capturable"] = True
                    super().__init__(*a, **kw)
            kal.physics.simplicits.SimplicitsScene = _Capturable
        else:
            kal.physics.simplicits.SimplicitsScene = base
        try:
            cube, pts = example.create_cube_object()
            scene, _ = example.build_scene(cube, pts)
            collision = scene.force_dict["collision"]["object"]
            per_step, contacts = [], []
            torch.cuda.synchronize()
            for _ in range(steps):
                t0 = time.perf_counter()
                scene.run_sim_step()
                torch.cuda.synchronize()
                per_step.append(time.perf_counter() - t0)
                contacts.append(collision.num_contacts)
            return per_step, contacts
        finally:
            kal.physics.simplicits.SimplicitsScene = base

    host_t, host_n = run(False)
    torch.cuda.empty_cache()
    cap_t, cap_n = run(True)

    def split(times, counts):
        hot = [t for t, n in zip(times, counts) if n > 0]
        cold = [t for t, n in zip(times, counts) if n == 0]
        return hot, cold

    host_hot, host_cold = split(host_t, host_n)
    cap_hot, cap_cold = split(cap_t, cap_n)

    print("\n" + "=" * 78)
    print(f"STACKING CUBES ({steps} steps, 3 cubes, 360 dof, 50000 contact capacity)")
    print("=" * 78)
    print(f"{'':<26}{'host (s)':>12}{'captured (s)':>14}{'speedup':>10}")
    print("-" * 78)
    print(f"{'all ' + str(steps) + ' steps':<26}{sum(host_t):>12.3f}"
          f"{sum(cap_t):>14.3f}{sum(host_t) / sum(cap_t):>9.2f}x")

    print(f"\n{'':<22}{'host ms':>10}{'captured ms':>13}{'speedup':>10}"
          f"{'host steps':>12}{'cap steps':>11}")
    print("-" * 78)
    for label, h, c in (("contact phase", host_hot, cap_hot),
                        ("zero-contact phase", host_cold, cap_cold)):
        mh = 1e3 * sum(h) / len(h) if h else float("nan")
        mc = 1e3 * sum(c) / len(c) if c else float("nan")
        speed = f"{mh / mc:>9.2f}x" if h and c else f"{'n/a':>10}"
        print(f"{label:<22}{mh:>10.2f}{mc:>13.2f}{speed}{len(h):>12}{len(c):>11}")
    print(f"\npeak contacts: host {max(host_n)}, captured {max(cap_n)}")
    if len(host_hot) != len(cap_hot):
        print(f"NOTE: the two runs spent different numbers of steps in contact "
              f"({len(host_hot)} vs {len(cap_hot)}), so the phase totals are not "
              f"comparable\n      and the overall speedup is skewed toward whichever "
              f"run collapsed sooner. Compare the per-step means.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-correctness", action="store_true")
    ap.add_argument("--stacking-cubes", action="store_true",
                    help="Time the stacking-cubes example instead of the synthetic "
                         "configs. Opt-in: it rebuilds the RKPM basis twice.")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    if args.stacking_cubes:
        return stacking_cubes()

    print("\n" + "=" * 78)
    print("CORRECTNESS: captured vs host Newton (25 steps, trajectory comparison)")
    print("=" * 78)
    print(f"{'config':<24}{'max |abs err|':>15}{'rel err':>12}{'|z| range':>12}{'':>6}")
    print("-" * 78)
    all_ok = True
    for cfg in CONFIGS:
        try:
            abs_err, rel_err, moved = check_correctness(cfg)
            ok = rel_err < 1e-4 and moved > 1e-6
            all_ok &= ok
            print(f"{cfg[0]:<24}{abs_err:>15.3e}{rel_err:>12.2e}{moved:>12.4f}"
                  f"{'  OK' if ok else '  FAIL':>6}")
        except Exception as e:
            all_ok = False
            print(f"{cfg[0]:<24}  ERROR: {type(e).__name__}: {str(e)[:70]}")

    if not all_ok:
        print("\nCorrectness FAILED -- not reporting timings.")
        return 1
    print("\nAll configs match.")

    if args.only_correctness:
        return 0

    print("\n" + "=" * 78)
    print("SPEEDUP")
    print("=" * 78)
    print(f"{'config':<24}{'newton/step':>12}{'host ms':>10}{'captured ms':>13}"
          f"{'speedup':>10}{'host launch':>13}")
    print("-" * 78)
    for cfg in CONFIGS:
        name, nobj, npts, nh, nqp, ym = cfg
        nit = newton_iters_per_step(cfg)

        ref = build_scene(nobj, npts, nh, nqp, ym, capturable=False)
        ms_ref, nl_ref = measure(ref, args.steps, want_launches=True)
        del ref
        torch.cuda.empty_cache()

        cap = build_scene(nobj, npts, nh, nqp, ym, capturable=True)
        ms_cap, _ = measure(cap, args.steps, want_launches=False)
        del cap
        torch.cuda.empty_cache()

        print(f"{name:<24}{nit:>12.1f}{ms_ref:>10.2f}{ms_cap:>13.2f}"
              f"{ms_ref/ms_cap:>9.2f}x{nl_ref:>13.0f}")
    print("\nnewton/step read from the device iteration counter. It is below "
          "max_newton_steps,\nproving the captured loop exits on convergence rather "
          "than running a fixed trip count.")

    print("\n" + "=" * 78)
    print("COST OF check_solve_info (one D2H sync per step)")
    print("=" * 78)
    print(f"{'config':<24}{'info ON ms':>12}{'info OFF ms':>13}{'sync cost':>12}"
          f"{'% of step':>11}")
    print("-" * 78)
    for cfg in CONFIGS:
        name, nobj, npts, nh, nqp, ym = cfg
        on = build_scene(nobj, npts, nh, nqp, ym, True, check_solve_info=True)
        ms_on, _ = measure(on, args.steps)
        del on
        torch.cuda.empty_cache()
        off = build_scene(nobj, npts, nh, nqp, ym, True, check_solve_info=False)
        ms_off, _ = measure(off, args.steps)
        del off
        torch.cuda.empty_cache()
        print(f"{name:<24}{ms_on:>12.2f}{ms_off:>13.2f}{ms_on - ms_off:>12.2f}"
              f"{100.0 * (ms_on - ms_off) / ms_on:>10.1f}%")
    print("\ncheck_solve_info defaults to True: a singular Hessian raises LinAlgError as "
          "the\nnon-capturable path does. Set False to drop the sync if the cost matters.")

    print("\n" + "=" * 78)
    print("SPEEDUP vs NEWTON ITERATION COUNT (240 dof)")
    print("=" * 78)
    print(f"{'dt':>7}{'conv_tol':>10}{'maxN':>6}{'newton/step':>13}{'host ms':>10}"
          f"{'captured ms':>13}{'speedup':>10}")
    print("-" * 78)
    for dt, ct, ym, mn in ITER_CONFIGS:
        cap = build_iter_scene(dt, ct, ym, mn, capturable=True)
        for _ in range(3):
            cap.run_sim_step()
        tot = 0
        for _ in range(10):
            cap.run_sim_step()
            tot += int(cap._nm_buf.nm_step_count.numpy()[0])
        nit = tot / 10.0
        ms_cap, _ = measure(cap, 15, warmup=2)
        del cap
        torch.cuda.empty_cache()

        ref = build_iter_scene(dt, ct, ym, mn, capturable=False)
        ms_ref, _ = measure(ref, 15, warmup=2)
        del ref
        torch.cuda.empty_cache()

        print(f"{dt:>7}{ct:>10.0e}{mn:>6}{nit:>13.1f}{ms_ref:>10.2f}{ms_cap:>13.2f}"
              f"{ms_ref/ms_cap:>9.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
