import numpy as np
import torch


def sdSphere(p):
    SPHERERAD = 0.5
    return np.linalg.norm(p)-SPHERERAD


def sdLink(p):
    # parameters
    le = 0.2
    r1 = 0.21
    r2 = 0.1
    q = np.array([p[0], max(abs(p[1])-le, 0.0), p[2]])
    return np.linalg.norm(np.array([np.linalg.norm(q[0:2])-r1, q[2]])) - r2


def sdBox(p):
    SDBOXSIZE = [1, 1, 1]
    b = np.array(SDBOXSIZE)
    q = np.absolute(p) - b
    return np.linalg.norm(np.array([max(q[0], 0.0), max(q[1], 0.0), max(q[2], 0.0)])) + min(max(q[0], max(q[1], q[2])), 0.0)


def example_unit_cube_object(num_points=100000, yms=1e5, prs=0.45, rhos=100, DEVICE='cuda', DTYPE=torch.float):
    uniform_points = np.random.uniform(
        [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], size=(num_points, 3))
    sdf_vals = np.apply_along_axis(sdBox, 1, uniform_points)
    # keep points where sd is not positive
    keep_points = np.nonzero(sdf_vals <= 0)[0]
    X0 = uniform_points[keep_points, :]
    X0_sdfval = sdf_vals[keep_points]

    YMs = yms*np.ones(X0.shape[0])
    PRs = prs*np.ones_like(YMs)
    Rhos = rhos*np.ones_like(YMs)

    bb_vol = (np.max(uniform_points[:, 0]) - np.min(uniform_points[:, 0])) * (np.max(uniform_points[:, 1]) - np.min(
        uniform_points[:, 1])) * (np.max(uniform_points[:, 2]) - np.min(uniform_points[:, 2]))
    vol_per_sample = bb_vol / uniform_points.shape[0]
    appx_vol = vol_per_sample*X0.shape[0]
    return torch.tensor(X0, device=DEVICE, dtype=DTYPE), torch.tensor(X0_sdfval, device=DEVICE, dtype=DTYPE), torch.tensor(YMs, device=DEVICE, dtype=DTYPE), torch.tensor(PRs, device=DEVICE, dtype=DTYPE), torch.tensor(Rhos, device=DEVICE, dtype=DTYPE), torch.tensor(appx_vol, device=DEVICE, dtype=DTYPE)


def example_unit_sphere_object(num_points=100000, yms=1e6, prs=0.45, rhos=1000, DEVICE='cuda', DTYPE=torch.float):
    uniform_points = np.random.uniform(
        [-3, -3, -3], [3, 3, 3], size=(num_points, 3))
    sdf_vals = np.apply_along_axis(sdSphere, 1, uniform_points)
    # keep points where sd is not positive
    keep_points = np.nonzero(sdf_vals <= 0)[0]
    X0 = uniform_points[keep_points, :]
    X0_sdfval = sdf_vals[keep_points]

    YMs = yms*np.ones(X0.shape[0])
    PRs = prs*np.ones_like(YMs)
    Rhos = rhos*np.ones_like(YMs)

    bb_vol = (np.max(uniform_points[:, 0]) - np.min(uniform_points[:, 0])) * (np.max(uniform_points[:, 1]) - np.min(
        uniform_points[:, 1])) * (np.max(uniform_points[:, 2]) - np.min(uniform_points[:, 2]))
    vol_per_sample = bb_vol / uniform_points.shape[0]
    appx_vol = vol_per_sample*X0.shape[0]

    return torch.tensor(X0, device=DEVICE, dtype=DTYPE), torch.tensor(X0_sdfval, device=DEVICE, dtype=DTYPE), torch.tensor(YMs, device=DEVICE, dtype=DTYPE), torch.tensor(PRs, device=DEVICE, dtype=DTYPE), torch.tensor(Rhos, device=DEVICE, dtype=DTYPE), torch.tensor(appx_vol, device=DEVICE, dtype=DTYPE)
