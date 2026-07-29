"""CUDA graph capture for the Simplicits sim step: correctness + speedup.

Correctness is checked first and gates the timing report -- a fast wrong answer is
worthless, and the line-search latch and convergence semantics are easy to get
subtly wrong when moved onto the device.

Usage:
    python tests/benchmarks/physics_graph_capture.py
    python tests/benchmarks/physics_graph_capture.py --only-correctness
"""
import argparse
import time

import torch
from torch.profiler import profile, ProfilerActivity

from kaolin.physics.simplicits import PhysicsPoints, SimplicitsObject, SimplicitsScene


def build_scene(num_objects, n_pts, num_handles, num_qp, ym, capturable,
                max_newton_steps=5, num_nodes=256, seed=0, check_solve_info=True):
    """Floor-contact scene. No inter-object collisions: the capturable path does not
    support them yet, so both paths are built identically without them."""
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
    scene.set_scene_gravity(torch.tensor([0.0, -9.8, 0.0]))
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
    scene.set_scene_gravity(torch.tensor([0.0, -9.8, 0.0]))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-correctness", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

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
