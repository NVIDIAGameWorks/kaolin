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

import numpy as np
import torch
from typing import List, Iterator, Union, Callable, Optional, Iterator
from scipy.special import comb
from kaolin.math import quat
from kaolin.render.camera import Camera


__all__ = [
    'loop_camera_path_generator',
    'camera_path_generator'
]


def _smoothstep(
        x: float,
        x_min: float = 0.0,
        x_max: float = 1.0,
        N: int = 3
):
    """Generalized smoothstep polynomials function, used for smooth interpolation of point x in [x_min, x_max].

    This function implements a generalized smoothstep that creates a smooth transition
    between 0 and 1 using higher-order polynomials. The resulting curve has zero 1st and 2nd
    derivatives at both endpoints, ensuring smooth transitions.

    The polynomial order is (2N + 1), where N controls the "smoothness" of the transition, i.e:
    - N = 1: 3rd order (classic smoothstep)
    - N = 3: 7th order (default, smoother transition)

    Args:
        x (float):
            Input value to be smoothly interpolated
        x_min (float, optional):
            Minimum value of the input range. Defaults to 0.0.
        x_max (float, optional):
            Maximum value of the input range. Defaults to 1.0.
        N (int, optional):
            Smoothness factor, determines polynomial order :math:`(2N + 1).`
            Defaults to :math:`3` (7th order polynomial).

    Returns:
        (float):
            Smoothly interpolated value in :math:`[0, 1]`. The interpolation value is clamped to :math:`[0, 1]`.

    .. seealso::

        https://en.wikipedia.org/wiki/Smoothstep#Generalization_to_higher-order_equations
    """
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)
    return result


def _catmull_rom(
        p0: Union[torch.Tensor, float, int],
        p1: Union[torch.Tensor, float, int],
        p2: Union[torch.Tensor, float, int],
        p3: Union[torch.Tensor, float, int],
        t: float
):
    """Interpolates a scalar or n-dimensional point on a Catmull-Rom spline curve.
    Catmull-Rom splines are C1 continuous cubic splines that guarantee to pass through their control points.
    The spline is defined by four control points, where the interpolation occurs between p1 and p2, while p0 and p3
    influence the tangents.

    Args:
        p0 (Union[torch.Tensor, float, int]):
            First control point, influences incoming tangent at p1
        p1 (Union[torch.Tensor, float, int]):
            Second control point, spline passes through this point
        p2 (Union[torch.Tensor, float, int]):
            Third control point, spline passes through this point
        p3 (Union[torch.Tensor, float, int]):
            Fourth control point, influences outgoing tangent at p2
        t (float):
            Interpolation parameter in [0, 1], where:
            - **t = 0** gives p1
            - **t = 1** gives p2
            - **0 < t < 1** gives smooth interpolation between p1 and p2

    Returns:
        (Union[torch.Tensor, float, int]):
            Interpolated point of the same type as input control points.
            For tensors, the interpolation is applied element-wise.
    """
    q_t = 0.5 * (
            (2.0 * p1) +
            (-p0 + p2) * t +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3
    )
    return q_t


def _lerp(
        a: Union[torch.Tensor, float, int],
        b: Union[torch.Tensor, float, int],
        t: float
) -> Union[torch.Tensor, float, int]:
    """Computes LERP, the Linear Interpolation function between two tensors, a and b.

    Args:
        a (torch.Tensor, float, int):
            Start tensor of arbitrary shape.
        b (torch.Tensor, float, int):
            End tensor, of similar shape to a.
        t (float):
            Interpolation factor, a scalar between 0 and 1 that determines where the result lies on the
            interpolation curve.

    Returns:
        (torch.Tensor):
            Interpolated tensor, same shape as a and b.
    """
    return a * (1 - t) + b * t


# TODO (operel): Move under kaolin.math.quat.quaternion
def _quaternion_angular_distance(q1, q2):
    """ Computes the angular distance between two unit quaternions q1 and q2,
    representing the minimum rotation angle needed to align them. The distance is
    invariant to quaternion double-cover (meaning q and -q represent the same rotation).

    The returned angle is in the range :math:`[0, pi]` radians, where:
    - 0 means the quaternions represent the same rotation
    - pi means the quaternions represent opposite rotations

    The formula used is :math:`theta = arccos(2(q1·q2)^2 - 1)`
    This gives the absolute angular difference regardless of rotation direction.

    Input quaternions are assumed to be normalized unit quaternions.

    Args:
        q1 (torch.Tensor):
            First unit quaternion of shape :math:`(4,)`
        q2 (torch.Tensor):
            Second unit quaternion of shape :math:`(4,)`

    Returns:
        torch.Tensor: Angular distance in radians, shape :math:`(1,).`
        Returns 0 if the quaternions represent the same or nearly same rotation
        (within eps tolerance).
    """
    q_dot = torch.dot(q1, q2)
    q_dot = torch.clamp(q_dot, -1.0, 1.0)
    return torch.acos(2 * q_dot * q_dot - 1.0)


def _catmull_rom_q(q0, q1, q2, q3, t, alpha=0.5, eps=1e-6):
    """Interpolates a quaternion using a Catmull-Rom spline formulation.
    Similar to _catmull_rom(), this function ensures the interpolation results in a quaternion on the unit sphere.

    Quaternion interpolation is computed using a pyramid formulation of spherical linear
    interpolation (SLERP) operations. The interpolation is performed between four control quaternions,
    with the result guaranteed to pass through q1 and q2. The spacing between quaternions is controlled
    by their angular distances raised to a power alpha, allowing for different parameterizations.
    Therefore, the interpolation may slow down near control points.

    All input quaternions are assumed to be normalized unit quaternions.

    .. seealso::

        https://en.wikipedia.org/wiki/Centripetal_Catmull-Rom_spline

    Args:
        q0 (torch.Tensor):
            First control quaternion of shape :math:`(4,)`, influences the "approach" to q1
        q1 (torch.Tensor):
            Second control quaternion of shape :math:`(4,)`, interpolation will pass through this point
        q2 (torch.Tensor):
            Third control quaternion of shape :math:`(4,)`, interpolation will pass through this point
        q3 (torch.Tensor):
            Fourth control quaternion of shape :math:`(4,)`, influences the "exit" from q2
        t (float):
            Interpolation parameter in :math:`[0,1]`, determines position along the spline
        alpha (float, optional):
            Controls the parameterization of the curve:
            **alpha = 0.0**: uniform spacing (constant speed, but may be too fast/slow in places)
            **alpha = 0.5**: centripetal (provides good balance of speed and smoothness)
            **alpha = 1.0**: chordal/arc-length (most uniform speed but may overshoot)
            Defaults to :math:`0.5` (centripetal).
        eps (float, optional):
            Small value to avoid division by zero in angle calculations. Defaults to :math:`1e-6`.
    """
    t0 = 0.0
    t1 = _quaternion_angular_distance(q0, q1) ** alpha + t0
    t2 = _quaternion_angular_distance(q1, q2) ** alpha + t1
    t3 = _quaternion_angular_distance(q2, q3) ** alpha + t2

    # Scale interpolation arg to [t1, t2]
    t = t * (t2 - t1) + t1

    tA1 = (t - t0) / (t1 - t0) if abs(t1 - t0) > eps else t0
    tA2 = (t - t1) / (t2 - t1) if abs(t2 - t1) > eps else t1
    tA3 = (t - t2) / (t3 - t2) if abs(t3 - t2) > eps else t2
    A1 = _slerp_q(q0, q1, t=tA1, eps=eps)
    A2 = _slerp_q(q1, q2, t=tA2, eps=eps)
    A3 = _slerp_q(q2, q3, t=tA3, eps=eps)

    tB1 = (t - t0) / (t2 - t0) if abs(t2 - t0) > eps else t0
    tB2 = (t - t1) / (t3 - t1) if abs(t3 - t1) > eps else t1
    B1 = _slerp_q(A1, A2, t=tB1, eps=eps)
    B2 = _slerp_q(A2, A3, t=tB2, eps=eps)

    tC = (t - t1) / (t2 - t1) if abs(t2 - t1) > eps else t1
    C = _slerp_q(B1, B2, t=tC, eps=eps)

    # Quaternion should be normalized, but enforce as a failsafe
    return quat.quat_unit(C)


def _slerp_q(a: torch.Tensor, b: torch.Tensor, t: float, eps=1e-6):
    """Computes SLERP, the Spherical Linear Interpolation function between two quaternions, a and b.

    Args:
        a (torch.Tensor):
            Start unit quaternion of shape :math:`(4,)`.
        b (torch.Tensor):
            End unit quaternion of shape :math:`(4,)`.
        t (float):
            Interpolation factor, a scalar between 0 and 1 that determines where the result lies on the
            interpolation curve.
        eps (float):
            Numerical precision parameter, to avoid division by zero if the angle between a,b is very small.
    Returns:
        (torch.Tensor):
            Interpolated unit quaternion of shape :math:`(4,)`.
    """
    dot = torch.dot(a, b)
    # If dot is negative, slerp will take the longer path along the sphere.
    # In that case, negate to take the shortest path
    if dot < 0.0:
        dot = -dot
        b = -b
    theta = torch.acos(torch.clamp(dot, -1.0, 1.0))  # Angle between a, b
    # If angle is too small, resort to LeRP to avoid division by zero
    if theta < eps:
        return (1.0 - t) * a + t * b
    sin_theta = torch.sin(theta)
    wa = torch.sin((1.0 - t) * theta) / sin_theta
    wb = torch.sin(t * theta) / sin_theta
    return wa * a + wb * b


def _lerp_q(a: torch.Tensor, b: torch.Tensor, t: float):
    """Computes LERP, the Linear Interpolation function between two quaternions, a and b.
    Args:
        a (torch.Tensor):
            Start unit quaternion of shape :math:`(4,)`.
        b (torch.Tensor):
            End unit quaternion of shape :math:`(4,)`.
        t (float):
            Interpolation factor, a scalar between 0 and 1 that determines where the result lies on the
            interpolation curve.
    Returns:
        (torch.Tensor):
            Interpolated unit quaternion of shape :math:`(4,)`.
    """
    dot = torch.dot(a, b)
    # If dot is negative, lerp will take the longer path along the sphere.
    # Negate to take the shortest path here
    if dot < 0.0:
        b = -b
    return a * (1 - t) + b * t


def _interpolate_camera_on_polynomial_path(
    trajectory: List[Camera],
    timestep: int,
    frames_between_cameras: int = 60,
    N: int = 3
) -> Camera:
    """Interpolates a camera from a smoothed path formed by a trajectory of cameras.

    The trajectory is assumed to have a list of at least 2 cameras, where the first and last cameras form a
    looped path.

    The interpolation is done using a generalized polynomial function.

    Args:
        trajectory (List[kaolin.render.camera.Camera]):
            A trajectory of camera nodes, used to form a continuous path.
        timestep (int):
            Timestep used to interpolate a point on the smoothed trajectory.
            `timestep` can take any integer value to support continuous animations.
            e.g. if `timestep > len(trajectory) X frames_between_cameras`, the timestep will fold over using a
            modulus op.
        frames_between_cameras (int):
            Number of interpolated points generated between each pair of cameras on the
            trajectory. In essence, this value controls how detailed, or smooth the path is.
        N (int):
            Determines the order polynomial, where the exact order is :math:`2N + 1`.

    Returns:
        (kaolin.render.camera.Camera):
            An interpolated camera object formed by the cameras trajectory.
    """
    traj_idx = (timestep // frames_between_cameras) % len(trajectory)
    cam1 = trajectory[traj_idx]
    cam2 = trajectory[traj_idx + 1]
    assert cam1.lens_type == cam2.lens_type, \
        ('_interpolate_camera_on_polynomial_path() only support interpolation of cameras with the same lens type, '
         f'but received cameras of types {cam1.lens_type} and {cam2.lens_type}.')
    eps = torch.finfo(cam1.dtype).eps
    # Add +1 here, as we assume Xs won't be exactly equal to 1. When skipping to the next key-cam,
    # traj_idx would already increase by 1 and Xs is 0.
    Xs = _smoothstep(np.linspace(0.0, 1.0, frames_between_cameras + 1), N=N)
    x = Xs[timestep % frames_between_cameras]  # Interpolation variable

    # Extrinsics
    q1 = quat.quat_from_rot33(cam1.R).squeeze(0)
    q2 = quat.quat_from_rot33(cam2.R).squeeze(0)
    q = _slerp_q(q1, q2, x, eps=eps)
    cam_R = quat.rot33_from_quat(q.unsqueeze(0))
    cam_t = _lerp(cam1.t, cam2.t, x)
    view_matrix = torch.eye(4, dtype=q1.dtype, device=q1.device).unsqueeze(0)
    view_matrix[0, :3, :3] = cam_R
    view_matrix[0, :3, 3] = cam_t[0, :, 0]

    # Intrinsics
    intrinsics = dict()
    if cam1.lens_type == 'pinhole' and cam2.lens_type == 'pinhole':
        intrinsics['fov'] = (cam1.fov(in_degrees=False).item(), cam2.fov(in_degrees=False).item())
    elif cam1.lens_type == 'ortho' and cam2.lens_type == 'ortho':
        intrinsics['fov_distance'] = (cam1.fov_distance.item(), cam2.fov_distance.item())
    intrinsics = {intr_name: _lerp(intr_val[0], intr_val[1], x) for intr_name, intr_val in intrinsics.items()}
    width = round(_lerp(cam1.width, cam2.width, x))
    height = round(_lerp(cam1.height, cam2.height, x))

    cam = Camera.from_args(
        view_matrix=view_matrix,
        width=width, height=height,
        device=cam1.device,
        dtype=cam1.dtype,
        **intrinsics
    )

    return cam


def _interpolate_camera_on_spline_path(
    trajectory: List[Camera],
    timestep: int,
    frames_between_cameras: int = 60
) -> Camera:
    """Interpolates a camera from a linear path formed by a trajectory of cameras.
    The trajectory is assumed to have a list of at least 4 cameras.
    The interpolation is done using Catmull-Rom Splines that guarantee to pass through the control points.

    Args:
        trajectory (List[kaolin.render.camera.Camera]):
            A trajectory of camera nodes, used to form a continuous path.
        timestep (int):
            Timestep used to interpolate a point on the smoothed trajectory.
            `timestep` can take any integer value to support continuous animations.
            e.g. if `timestep > len(trajectory) X frames_between_cameras`, the timestep will fold over using a
            modulus.
        frames_between_cameras (int):
            Number of interpolated points generated between each pair of cameras on the
            trajectory. In essence, this value controls how detailed, or smooth the path is.

    Returns:
        (kaolin.render.camera.Camera):
            An interpolated camera object formed by the cameras trajectory.
    """
    traj_idx = (timestep // frames_between_cameras) % len(trajectory)
    traj_idx = min(max(traj_idx, 0), len(trajectory) - 3)
    cam1 = trajectory[traj_idx - 1]
    cam2 = trajectory[traj_idx]
    cam3 = trajectory[traj_idx + 1]
    cam4 = trajectory[traj_idx + 2]
    assert cam1.lens_type == cam2.lens_type == cam3.lens_type == cam4.lens_type, \
        ('_interpolate_camera_on_spline_path() only support interpolation of cameras with the same lens type, '
         f'but received cameras of types {cam1.lens_type}, {cam2.lens_type}, {cam3.lens_type}, {cam4.lens_type}.')
    eps = torch.finfo(cam1.dtype).eps
    # Add +1 here, as we assume Xs won't be exactly equal to 1. When skipping to the next key-cam,
    # traj_idx would already increase by 1 and Xs is 0.
    Xs = np.linspace(0.0, 1.0, frames_between_cameras + 1)
    x = Xs[timestep % frames_between_cameras]  # Interpolation variable

    # Extrinsics
    q1 = quat.quat_from_rot33(cam1.R).squeeze(0)
    q2 = quat.quat_from_rot33(cam2.R).squeeze(0)
    q3 = quat.quat_from_rot33(cam3.R).squeeze(0)
    q4 = quat.quat_from_rot33(cam4.R).squeeze(0)
    q = _catmull_rom_q(q1, q2, q3, q4, x, eps=eps)
    cam_R = quat.rot33_from_quat(q.unsqueeze(0))
    cam_t = _catmull_rom(cam1.t, cam2.t, cam3.t, cam4.t, x)
    view_matrix = torch.eye(4, dtype=q1.dtype, device=q1.device).unsqueeze(0)
    view_matrix[0, :3, :3] = cam_R
    view_matrix[0, :3, 3] = cam_t[0, :, 0]

    # Intrinsics
    intrinsics = dict()
    if cam1.lens_type == 'pinhole':
        intrinsics['fov'] = (
            cam1.fov(in_degrees=False).item(),
            cam2.fov(in_degrees=False).item(),
            cam3.fov(in_degrees=False).item(),
            cam4.fov(in_degrees=False).item()
        )
    elif cam1.lens_type == 'ortho':
        intrinsics['fov_distance'] = \
            (cam1.fov_distance.item(),
             cam2.fov_distance.item(),
             cam3.fov_distance.item(),
             cam4.fov_distance.item()
        )
    else:
        raise ValueError(f'Unsupported lens type {cam1.lens_type}')

    intrinsics = {intr_name: _catmull_rom(intr_val[0], intr_val[1], intr_val[2], intr_val[3], x)
                  for intr_name, intr_val in intrinsics.items()}
    width = round(_catmull_rom(cam1.width, cam2.width, cam3.width, cam4.width, x))
    height = round(_catmull_rom(cam1.height, cam2.height, cam3.height, cam4.height, x))

    # Create camera from view matrix
    cam = Camera.from_args(
        view_matrix=view_matrix,
        width=width, height=height,
        device=cam1.device,
        dtype=cam1.dtype,
        **intrinsics
    )

    return cam


def _get_interpolator(interpolation: str, trajectory: List[Camera]) -> Callable:
    """Returns an interpolator function based on the interpolation type and trajectory.

    Args:
        interpolation (str):
            Type of interpolation function used:
            'polynomial' uses a smoothstep polynomial function which tends to overshoot around the keyframes.
            This interpolator is fitting for paths orbiting an object of interest.
            'catmull_rom' uses a spline defined by 4 control points, guaranteed to pass precisely through the keyframes.

    Returns:
        Callable:
            An interpolator function that takes a trajectory, timestep, and frames_between_cameras,
            and returns a camera.
    """
    if interpolation == 'polynomial':
        interpolator = _interpolate_camera_on_polynomial_path
        if len(trajectory) < 2:
            raise ValueError("For polynomial interpolation, cameras trajectory must have at least 2 cameras.")
    elif interpolation == 'catmull_rom':
        interpolator = _interpolate_camera_on_spline_path
        if len(trajectory) < 4:
            raise ValueError("For catmull_rom interpolation, cameras trajectory must have at least 4 cameras.")
    else:
        raise ValueError("Unknown interpolation function specified. Valid options: 'polynomial', 'catmull_rom'.")
    return interpolator


def loop_camera_path_generator(
    trajectory: List[Camera],
    frames_between_cameras: int = 60,
    interpolation: str = 'polynomial',
    repeat: Optional[int] = None
) -> Iterator[Camera]:
    """A generator function for returning continuous camera objects an on a smoothed path interpolated
    from a trajectory of cameras.

    The trajectory is assumed to have a list of at least 2 cameras, where the first and last cameras form a
    looped path. Certain interpolation modes (i.e. catmull_rom) may require additional cameras.
    If repeat is None, this generator is therefore never exhausted, and can be invoked infinitely to generate continuous
    camera motion. Otherwise the loop will repeat a finite number of times.

    Args:
        trajectory (List[kaolin.render.camera.Camera]):
            A trajectory of camera nodes, used to form a continuous path.
        frames_between_cameras (int): Number of interpolated points generated between each pair of cameras on the
            trajectory. In essence, this value controls how detailed, or smooth the path is.
        interpolation (str): Type of interpolation function used:
            'polynomial' uses a smoothstep polynomial function which tends to overshoot around the keyframes.
            This interpolator is fitting for paths orbiting an object of interest.
            'catmull_rom' uses a spline defined by 4 control points, guaranteed to pass precisely through the keyframes.
        repeat (int, Optional):
            If specified, will limit the number of loops. Passing None results in an infinite loop.

    Returns:
        (Iterator[kaolin.render.camera.Camera]):
            An interpolated camera object formed by the cameras trajectory.
    """
    interpolator = _get_interpolator(interpolation, trajectory)

    _trajectory = [trajectory[-1]] + trajectory + [trajectory[0], trajectory[1]]
    frames_between_cameras += 1
    timestep = frames_between_cameras

    iter = 0
    iters_per_round = len(trajectory) * frames_between_cameras

    while True:
        yield interpolator(_trajectory, timestep, frames_between_cameras)
        timestep = max(
            (timestep + 1) % ((len(trajectory) + 1) * frames_between_cameras), frames_between_cameras
        )

        iter += 1
        if repeat is not None and iter > (iters_per_round * repeat):
            break


def camera_path_generator(
    trajectory: List[Camera],
    frames_between_cameras: int = 60,
    interpolation: str = 'catmull_rom'
) -> Iterator[Camera]:
    """A finite generator function for returning continuous camera objects an o path interpolated
    from a trajectory of cameras.

    This generator is exhausted after it returns the last point on the path.
    If interpolation is 'polynomial' - the trajectory is assumed to have a list of at least 2 cameras.
    If interpolation is 'catmull_rom' - the trajectory is assumed to have a list of at least 4 cameras.

    Args:
        trajectory (List[kaolin.render.camera.Camera]): A trajectory of camera nodes, used to form a continuous path.
            frames_between_cameras (int): Number of interpolated points generated between each pair of cameras on the
            trajectory. In essence, this value controls how detailed, or smooth the path is.
        interpolation (str): Type of interpolation function used:
            'polynomial' uses a smoothstep polynomial function which tends to overshoot around the keyframes.
            This interpolator is fitting for paths orbiting an object of interest.
            'catmull_rom' uses a spline defined by 4 control points, guaranteed to pass precisely through the keyframes.

    Returns:
        (Iterator[kaolin.render.camera.Camera]):
            An interpolated camera object formed by the cameras trajectory.
    """
    interpolator = _get_interpolator(interpolation, trajectory)

    _trajectory = [trajectory[0]] + trajectory + [trajectory[-1], trajectory[-1]]
    frames_between_cameras += 1
    timestep = frames_between_cameras

    while True:
        yield interpolator(_trajectory, timestep, frames_between_cameras)
        timestep += 1

        traj_idx = (timestep // frames_between_cameras) % len(_trajectory)
        if traj_idx == len(_trajectory) - 3:
            break

    yield interpolator(_trajectory, timestep, frames_between_cameras)
