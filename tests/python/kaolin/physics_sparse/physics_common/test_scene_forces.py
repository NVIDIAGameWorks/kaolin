import pytest

import torch
import warp as wp
import warp.sparse as wps
import kaolin.physics as physics
import kaolin.physics_sparse as physics_sparse
from kaolin.physics_sparse.physics_common import gravity_energy, gravity_gradient, floor_energy, floor_gradient, floor_hessian


@wp.kernel
def gravity_energy_kernel(
    g: wp.vec3,                             # acceleration due to gravity
    rho: wp.array(dtype=wp.float32),        # density at each point
    vol: wp.array(dtype=wp.float32),        # volume at each point
    dx: wp.array(dtype=wp.vec3),            # dx
    x0: wp.array(dtype=wp.vec3),            # x0
    coeff: wp.float32,
    energy: wp.array(dtype=wp.float32)      # Array of size 1
):
    # Get thread index
    tid = wp.tid()
    wp.atomic_add(energy, 0, coeff * gravity_energy(
        g, rho[tid], vol[tid], dx[tid], x0[tid]))


@wp.kernel
def gravity_gradient_kernel(
    g: wp.vec3,                        # acceleration due to gravity
    rho: wp.array(dtype=wp.float32),   # density at each point
    vol: wp.array(dtype=wp.float32),   # volume at each point
    dx: wp.array(dtype=wp.vec3),       # dx
    x0: wp.array(dtype=wp.vec3),       # x0
    coeff: wp.float32,
    f: wp.array(dtype=wp.vec3)         # output forces
):
    # Get thread index
    tid = wp.tid()

    # Each point contributes proportionally to its density
    point_mass = rho[tid] * vol[tid]

    # Compute gravitational force (F = mg)
    # Write force components to output array
    f[tid] += coeff * gravity_gradient(g, rho[tid], vol[tid], dx[tid], x0[tid])


@wp.kernel
def floor_energy_kernel(
    floor_height: wp.float32,
    floor_axis: wp.int32,
    flip_floor: wp.int32,
    vol: wp.array(dtype=wp.float32),
    dx: wp.array(dtype=wp.vec3),
    x0: wp.array(dtype=wp.vec3),
    coeff: wp.float32,
    energy: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    wp.atomic_add(energy, 0, coeff * floor_energy(
        floor_height, floor_axis, flip_floor, vol[tid], dx[tid], x0[tid]))


@wp.kernel
def floor_gradient_kernel(
    floor_height: wp.float32,
    floor_axis: wp.int32,
    flip_floor: wp.int32,
    vol: wp.array(dtype=wp.float32),
    dx: wp.array(dtype=wp.vec3),
    x0: wp.array(dtype=wp.vec3),
    coeff: wp.float32,
    f: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    wp.atomic_add(f, tid, coeff * floor_gradient(
        floor_height, floor_axis, flip_floor, vol[tid], dx[tid], x0[tid]))


@wp.kernel
def floor_hessian_kernel(
    floor_height: wp.float32,
    floor_axis: wp.int32,
    flip_floor: wp.int32,
    vol: wp.array(dtype=wp.float32),
    dx: wp.array(dtype=wp.vec3),
    x0: wp.array(dtype=wp.vec3),
    coeff: wp.float32,
    H_blocks: wp.array(dtype=wp.mat33)
):
    tid = wp.tid()
    wp.atomic_add(H_blocks, tid, coeff * floor_hessian(
        floor_height, floor_axis, flip_floor, vol[tid], dx[tid], x0[tid]))


class Gravity:
    def __init__(self, g, integration_pt_density, integration_pt_volume):
        # warp constant [x, y, z] acceleration due to gravity
        self.g = g
        # warp constant [N] density at each point
        self.integration_pt_density = integration_pt_density
        # warp constant [N] volume at each point
        self.integration_pt_volume = integration_pt_volume

        # warp constant [N, 3] gradients
        self.gradients = None
        # warp constant [N, 9] hessians
        self.hessians_blocks = wp.zeros(
            integration_pt_volume.shape[0], dtype=wp.mat33, device=integration_pt_volume.device)
        self.sparse_hessian = None

    def energy(self, wp_dx, wp_x0, coeff=1.0):
        wp_energy = wp.zeros(1, dtype=wp.float32, device=wp_dx.device)
        wp.launch(
            kernel=gravity_energy_kernel,
            dim=wp_dx.shape[0],
            inputs=[self.g,  # [gx, gy, gz] Acceleration due to gravity
                    self.integration_pt_density,
                    self.integration_pt_volume,
                    wp_dx,
                    wp_x0,
                    coeff],
            outputs=[wp_energy],
            adjoint=False
        )
        return wp_energy

    def gradient(self, wp_dx, wp_x0, coeff=1.0):
        if self.gradients is None:
            self.gradients = wp.zeros(
                wp_dx.shape[0], dtype=wp.vec3, device=wp_dx.device)
            wp.launch(
                kernel=gravity_gradient_kernel,
                dim=wp_dx.shape[0],
                inputs=[self.g,  # [gx, gy, gz] Acceleration due to gravity
                        self.integration_pt_density,
                        self.integration_pt_volume,
                        wp_dx,
                        wp_x0,
                        coeff],
                outputs=[self.gradients],
                adjoint=False
            )

        return self.gradients

    def hessian(self, wp_dx, wp_x0, coeff=1.0):
        if self.sparse_hessian is None:
            # [N, N] sparse zero hessian
            self.sparse_hessian = wps.bsr_zeros(
                wp_dx.shape[0], wp_dx.shape[0], block_type=wp.mat33)

        return self.sparse_hessian


class Floor:
    def __init__(self,
                 floor_height,
                 floor_axis,
                 flip_floor,
                 integration_pt_volume):
        self.floor_height = floor_height
        self.floor_axis = floor_axis
        self.flip_floor = flip_floor
        self.integration_pt_volume = integration_pt_volume

        # pre-allocated gradients and hessians
        self.gradients = wp.zeros(
            integration_pt_volume.shape[0], dtype=wp.vec3, device=integration_pt_volume.device)
        self.hessians_blocks = wp.zeros(
            integration_pt_volume.shape[0], dtype=wp.mat33, device=integration_pt_volume.device)

    def energy(self, wp_dx, wp_x0, coeff=1.0):
        wp_energy = wp.zeros(1, dtype=wp.float32, device=wp_dx.device)
        wp.launch(
            kernel=floor_energy_kernel,
            dim=wp_dx.shape[0],
            inputs=[self.floor_height,
                    self.floor_axis,
                    self.flip_floor,
                    self.integration_pt_volume,
                    wp_dx,
                    wp_x0,
                    coeff],
            outputs=[wp_energy],
            adjoint=False
        )
        return wp_energy

    def gradient(self, wp_dx, wp_x0, coeff=1.0):
        self.gradients.zero_()
        wp.launch(
            kernel=floor_gradient_kernel,
            dim=wp_dx.shape[0],
            inputs=[self.floor_height,
                    self.floor_axis,
                    self.flip_floor,
                    self.integration_pt_volume,
                    wp_dx,
                    wp_x0,
                    coeff],
            outputs=[self.gradients],
            adjoint=False
        )
        return self.gradients

    def hessian(self, wp_dx, wp_x0, coeff=1.0):
        self.hessians_blocks.zero_()
        wp.launch(
            kernel=floor_hessian_kernel,
            dim=wp_dx.shape[0],
            inputs=[self.floor_height,
                    self.floor_axis,
                    self.flip_floor,
                    self.integration_pt_volume,
                    wp_dx,
                    wp_x0,
                    coeff],
            outputs=[self.hessians_blocks],
            adjoint=False
        )
        wp_H = wps.bsr_zeros(
            wp_dx.shape[0], wp_dx.shape[0], block_type=wp.mat33)
        wps.bsr_set_diag(wp_H, self.hessians_blocks)
        return wp_H


def setup_gravity(device, dtype):
    # Test parameters
    N = 100
    vol = 1.0
    gravity_acc = torch.tensor([0, -9.81, 0], device=device)

    # Generate test points and density
    x0 = torch.randn((N, 3), device=device, dtype=dtype)
    x0[:, 1] += 0.5
    dx = 1e-2*torch.randn((N, 3), device=device, dtype=dtype)
    x = dx + x0

    integration_pt_density = torch.ones((N, 1), device=device, dtype=dtype)

    integration_pt_volume = (vol/N)*torch.ones((N, 1),
                                               device=device, dtype=dtype)

    gravity_object = physics.utils.Gravity(
        rhos=integration_pt_density, acceleration=gravity_acc)

    # Setup warp gravity struct
    gravity_struct = Gravity(g=wp.vec3(0, -9.81, 0),
                             integration_pt_density=wp.from_torch(
        integration_pt_density.flatten(), dtype=wp.float32),
        integration_pt_volume=wp.from_torch(
        integration_pt_volume.flatten(), dtype=wp.float32)
    )

    return x0, dx, integration_pt_volume, gravity_object, gravity_struct


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_wp_gravity_energy(device, dtype):
    x0, dx, integration_pt_vol, gravity_object, gravity_struct = setup_gravity(
        device, dtype)

    # Calculate expected energy using Kaolin physics
    expected_gravity_energy = gravity_object.energy(
        (dx + x0), integration_pt_vol)

    wp_points = wp.from_torch(x0, dtype=wp.vec3)
    wp_delta = wp.from_torch(dx, dtype=wp.vec3)
    wp_energy = gravity_struct.energy(
        wp_delta, wp_points)

    print(expected_gravity_energy.sum(), wp.to_torch(wp_energy).sum())
    # Compare results
    assert torch.allclose(expected_gravity_energy.sum(),
                          wp.to_torch(wp_energy).sum())


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_wp_gravity_gradient(device, dtype):
    x0, dx, integration_pt_vol, gravity_object, gravity_struct = setup_gravity(
        device, dtype)

    # Calculate expected gradient using Kaolin physics
    expected_gravity_gradient = gravity_object.gradient(
        (dx + x0), integration_pt_vol)

    wp_points = wp.from_torch(x0, dtype=wp.vec3)
    wp_delta = wp.from_torch(dx, dtype=wp.vec3)
    wp_gradients = gravity_struct.gradient(
        wp_delta, wp_points)

    # Compare results
    assert torch.allclose(expected_gravity_gradient,
                          wp.to_torch(wp_gradients))


def setup_floor(device, dtype):
    # Test parameters
    N = 100
    vol = 1.0
    floor_height = 0.0
    floor_axis = 1
    flip_floor = 0

    # Generate test points and density
    x0 = torch.randn((N, 3), device=device, dtype=dtype)
    dx = 1e-2*torch.randn((N, 3), device=device, dtype=dtype)
    x = dx + x0

    integration_pt_density = torch.ones((N, 1), device=device, dtype=dtype)

    integration_pt_volume = (vol/N)*torch.ones((N, 1),
                                               device=device, dtype=dtype)

    floor_object = physics.utils.Floor(
        floor_height=floor_height, floor_axis=floor_axis, flip_floor=flip_floor)

    # Setup warp floor struct
    floor_struct = Floor(floor_height=floor_height,
                         floor_axis=floor_axis,
                         flip_floor=flip_floor,
                         integration_pt_volume=wp.from_torch(
                             integration_pt_volume.flatten(), dtype=wp.float32))
    return x0, dx, integration_pt_volume, floor_object, floor_struct


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_wp_floor_energy(device, dtype):
    x0, dx, integration_pt_vol, floor_object, floor_struct = setup_floor(
        device, dtype)

    # Calculate expected energy using Kaolin physics
    expected_floor_energy = floor_object.energy(
        (dx + x0), integration_pt_vol[0])

    wp_points = wp.from_torch(x0, dtype=wp.vec3)
    wp_delta = wp.from_torch(dx, dtype=wp.vec3)
    wp_energy = floor_struct.energy(
        wp_delta, wp_points)

    print(wp.to_torch(wp_energy).sum())
    # Compare results
    assert torch.allclose(expected_floor_energy.sum(),
                          wp.to_torch(wp_energy).sum())


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_wp_floor_gradient(device, dtype):
    x0, dx, integration_pt_vol, floor_object, floor_struct = setup_floor(
        device, dtype)

    # Calculate expected gradient using Kaolin physics
    expected_floor_gradient = floor_object.gradient(
        (dx + x0), integration_pt_vol)

    wp_points = wp.from_torch(x0, dtype=wp.vec3)
    wp_delta = wp.from_torch(dx, dtype=wp.vec3)
    wp_gradients = floor_struct.gradient(
        wp_delta, wp_points)

    # Compare results
    assert torch.allclose(expected_floor_gradient,
                          wp.to_torch(wp_gradients))


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_wp_floor_hessian(device, dtype):
    x0, dx, integration_pt_vol, floor_object, floor_struct = setup_floor(
        device, dtype)

    # Calculate expected hessian using Kaolin physics
    expected_floor_hessian = floor_object.hessian(
        (dx + x0), integration_pt_vol[0]).transpose(1, 2).reshape(dx.shape[0]*dx.shape[1], dx.shape[0]*dx.shape[1])  # reshape to 3N x 3N size

    wp_points = wp.from_torch(x0, dtype=wp.vec3)
    wp_delta = wp.from_torch(dx, dtype=wp.vec3)
    wp_hessian = floor_struct.hessian(
        wp_delta, wp_points)

    print(wp_hessian.shape, expected_floor_hessian.shape)

    wp_hessian_dense = physics_sparse.physics_common.bsr_to_torch(wp_hessian)
    # Compare results
    assert torch.allclose(expected_floor_hessian,
                          wp_hessian_dense.to_dense())
