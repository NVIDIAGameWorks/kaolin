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

import torch
import warp as wp
import warp.sparse as wps

__all__ = ['Gravity',
           'Floor',
           'Boundary']


@wp.func
def gravity_energy_wp_func(g: wp.vec3,
                   rho: wp.float32,
                   vol: wp.float32,
                   dx: wp.vec3,
                           x0: wp.vec3):  # pragma: no cover
    r"""Returns gravitational potential energy at each integration primitive, :math:`mgh` where :math:`h=x_0+dx` and :math:`m=rho*vol`.

    Args:
        g (wp.vec3): Gravity acceleration (0, -9.81, 0)
        rho (wp.float32): Density
        vol (wp.float32): Volume
        dx (wp.vec3): Delta in position
        x0 (wp.vec3): Rest position

    Returns:
        Scalar: Gravity energy
    """
    return wp.dot(g, dx + x0) * rho * vol


@wp.func
def gravity_gradient_wp_func(g: wp.vec3,
                     rho: wp.float32,
                     vol: wp.float32,
                     dx: wp.vec3,
                             x0: wp.vec3):  # pragma: no cover
    r"""Returns gravitational force at each integration primitive :math:`mg` where :math:`m=rho*vol`.

    Args:
        g (wp.vec3): Gravity acceleration (0, -9.81, 0)
        rho (wp.float32): Density
        vol (wp.float32): Volume
        dx (wp.vec3): Delta in position
        x0 (wp.vec3): Rest position

    Returns:
        wp.vec3: Gravity force
    """
    return g * rho * vol


@wp.func
def gravity_hessian_wp_func(g: wp.vec3,
                    rho: wp.float32,
                    vol: wp.float32,
                    dx: wp.vec3,
                            x0: wp.vec3):  # pragma: no cover
    r"""Returns gravity hessian matrix at each integration primitive which is 0.

    Args:
        g (wp.vec3): Gravity acceleration (0, -9.81, 0)
        rho (wp.float32): Density
        vol (wp.float32): Volume
        dx (wp.vec3): Delta in position
        x0 (wp.vec3): Rest position

    Returns:
        wp.mat33: Gravity hessian
    """
    return wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def gravity_energy_wp_kernel(
    g: wp.vec3,                             # acceleration due to gravity
    rho: wp.array(dtype=wp.float32),        # density at each point
    vol: wp.array(dtype=wp.float32),        # volume at each point
    dx: wp.array(dtype=wp.vec3),            # dx
    x0: wp.array(dtype=wp.vec3),            # x0
    coeff: wp.float32,
    energy: wp.array(dtype=wp.float32)      # Array of size 1
):  # pragma: no cover
    # Get thread index
    tid = wp.tid()
    wp.atomic_add(energy, 0, coeff * gravity_energy_wp_func(
        g, rho[tid], vol[tid], dx[tid], x0[tid]))


@wp.kernel
def gravity_gradient_wp_kernel(
    g: wp.vec3,                        # acceleration due to gravity
    rho: wp.array(dtype=wp.float32),   # density at each point
    vol: wp.array(dtype=wp.float32),   # volume at each point
    dx: wp.array(dtype=wp.vec3),       # dx
    x0: wp.array(dtype=wp.vec3),       # x0
    coeff: wp.float32,
    f: wp.array(dtype=wp.vec3)         # output forces
):  # pragma: no cover
    
    # Get thread index
    tid = wp.tid()

    # Each point contributes proportionally to its density
    point_mass = rho[tid] * vol[tid]

    # Compute gravitational force (F = mg)
    # Write force components to output array
    f[tid] += coeff * gravity_gradient_wp_func(g, rho[tid], vol[tid], dx[tid], x0[tid])


@wp.func
def floor_energy_wp_func(floor_height: wp.float32,
                 floor_axis: wp.int32,
                 flip_floor: wp.int32,
                 vol: wp.float32,
                 dx: wp.vec3,
                         x0: wp.vec3):  # pragma: no cover
    r"""Return floor energy at each integration primitive.

    Args:
        floor_height (wp.float32): Floor height
        floor_axis (wp.int32): Floor axis (0-x, or 1-y, or 2-z)
        flip_floor (wp.int32): Flips the direction of the floor (0-False, 1-True)
        vol (wp.float32): Volume
        dx (wp.vec3): Delta in position
        x0 (wp.vec3): Rest position

    Returns:
        Scalar: Floor energy
    """

    x = dx + x0
    p = float(x[floor_axis])
    if flip_floor == 0:
        if p < floor_height:
            return vol*(p - float(floor_height))**2.0
    else:
        if p > floor_height:
            return vol*(p - float(floor_height))**2.0

    return 0.0


@wp.func
def floor_gradient_wp_func(floor_height: wp.float32,
                   floor_axis: wp.int32,
                   flip_floor: wp.int32,
                   vol: wp.float32,
                   dx: wp.vec3,
                           x0: wp.vec3):  # pragma: no cover
    r"""Return floor gradient at each integration primitive.

    Args:
        floor_height (wp.float32): Floor height
        floor_axis (wp.int32): Floor axis (0-x, or 1-y, or 2-z)
        flip_floor (wp.int32): Flips the direction of the floor (0-False, 1-True)
        vol (wp.float32): Volume
        dx (wp.vec3): Delta in position
        x0 (wp.vec3): Rest position

    Returns:
        wp.vec3: Floor gradient
    """

    x = dx + x0
    p = float(x[floor_axis])
    force = wp.vec3(0.0, 0.0, 0.0)
    force[floor_axis] = vol*2.0*(p - floor_height)

    if flip_floor == 0:
        if p < floor_height:
            return 1.0*force
        else:
            pass
    else:
        if p > floor_height:
            return -1.0*force
        else:
            pass

    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def floor_hessian_wp_func(floor_height: wp.float32,
                  floor_axis: wp.int32,
                  flip_floor: wp.int32,
                  vol: wp.float32,
                  dx: wp.vec3,
                          x0: wp.vec3):  # pragma: no cover
    r"""Return floor hessian at each integration primitive.

    Args:
        floor_height (wp.float32): Floor height
        floor_axis (wp.int32): Floor axis (0-x, or 1-y, or 2-z)
        flip_floor (wp.int32): Flips the direction of the floor (0-False, 1-True)
        vol (wp.float32): Volume
        dx (wp.vec3): Delta in position
        x0 (wp.vec3): Rest position

    Returns:
        wp.mat33: Floor hessian
    """

    x = dx + x0
    p = float(x[floor_axis])
    local_hess = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    local_hess[floor_axis, floor_axis] = vol*2.0

    if flip_floor == 0:
        if p < floor_height:
            return 1.0*local_hess
        else:
            pass
    else:
        if p > floor_height:
            return -1.0*local_hess
        else:
            pass
    return wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def floor_energy_wp_kernel(
        floor_height: wp.float32,
        floor_axis: wp.int32,
        flip_floor: wp.int32,
        vol: wp.array(dtype=wp.float32),
        dx: wp.array(dtype=wp.vec3),
        x0: wp.array(dtype=wp.vec3),
        coeff: wp.float32,
        energy: wp.array(dtype=wp.float32)):  # pragma: no cover

    tid = wp.tid()
    wp.atomic_add(energy, 0, coeff * floor_energy_wp_func(
        floor_height, floor_axis, flip_floor, vol[tid], dx[tid], x0[tid]))


@wp.kernel
def floor_gradient_wp_kernel(
        floor_height: wp.float32,
        floor_axis: wp.int32,
        flip_floor: wp.int32,
        vol: wp.array(dtype=wp.float32),
        dx: wp.array(dtype=wp.vec3),
        x0: wp.array(dtype=wp.vec3),
        coeff: wp.float32,
        f: wp.array(dtype=wp.vec3)):  # pragma: no cover

    tid = wp.tid()
    f[tid] += coeff * floor_gradient_wp_func(
        floor_height, floor_axis, flip_floor, vol[tid], dx[tid], x0[tid])


@wp.kernel
def floor_hessian_wp_kernel(
        floor_height: wp.float32,
        floor_axis: wp.int32,
        flip_floor: wp.int32,
        vol: wp.array(dtype=wp.float32),
        dx: wp.array(dtype=wp.vec3),
        x0: wp.array(dtype=wp.vec3),
        coeff: wp.float32,
        H_blocks: wp.array(dtype=wp.mat33)):  # pragma: no cover

    tid = wp.tid()
    H_blocks[tid] += coeff * floor_hessian_wp_func(
        floor_height, floor_axis, flip_floor, vol[tid], dx[tid], x0[tid])


@wp.func
def boundary_energy_wp_func(pin_pos: wp.vec3,
                    vol: wp.float32,
                    dx: wp.vec3,
                            x0: wp.vec3):  # pragma: no cover
    r"""Return boundary energy at each integration primitive.

    Args:
        pin_pos (wp.vec3): Pinned vertex position
        vol (wp.float32): Volume
        dx (wp.vec3): Delta in position
        x0 (wp.vec3): Rest position

    Returns:
        wp.float32: Boundary energy
    """
    current_pos = dx + x0
    sq_norm = wp.dot(current_pos - pin_pos, current_pos - pin_pos)
    return sq_norm


@wp.func
def boundary_gradient_wp_func(pin_pos: wp.vec3,
                      vol: wp.float32,
                      dx: wp.vec3,
                              x0: wp.vec3):  # pragma: no cover
    r"""Return boundary gradient at each integration primitive.

    Args:
        pin_pos (wp.vec3): Pinned vertex position
        vol (wp.float32): Volume
        dx (wp.vec3): Delta in position
        x0 (wp.vec3): Rest position

    Returns:
        wp.vec3: Boundary gradient
    """
    current_pos = dx + x0
    grad = 2.0 * (current_pos - pin_pos)
    return grad


@wp.func
def boundary_hessian_wp_func(pin_pos: wp.vec3,
                     vol: wp.float32,
                     dx: wp.vec3,
                             x0: wp.vec3):  # pragma: no cover
    r"""Return boundary hessian at each integration primitive.

    Args:
        pin_pos (wp.vec3): Pinned vertex position
        vol (wp.float32): Volume
        dx (wp.vec3): Delta in position
        x0 (wp.vec3): Rest position

    Returns:
        wp.mat33: Boundary hessian
    """
    hess = 2.0 * wp.identity(3, dtype=wp.float32)
    return hess


@wp.kernel
def boundary_energy_wp_kernel(
    pinned_vertices: wp.array(dtype=wp.vec3),
    pinned_indices: wp.array(dtype=wp.int32),
    vol: wp.array(dtype=wp.float32),
    dx: wp.array(dtype=wp.vec3),
    x0: wp.array(dtype=wp.vec3),
    coeff: wp.float32,
    energy: wp.array(dtype=wp.float32)
):  # pragma: no cover
    tid = wp.tid()  # pinned point index
    pid = pinned_indices[tid]  # global index of pinned point

    pinned_pos = pinned_vertices[tid]
    wp.atomic_add(energy, 0, coeff * boundary_energy_wp_func(
        pinned_pos, vol[pid], dx[pid], x0[pid]))


@wp.kernel
def boundary_gradient_wp_kernel(
    pinned_vertices: wp.array(dtype=wp.vec3),
    pinned_indices: wp.array(dtype=wp.int32),
    vol: wp.array(dtype=wp.float32),
    dx: wp.array(dtype=wp.vec3),
    x0: wp.array(dtype=wp.vec3),
    coeff: wp.float32,
    f: wp.array(dtype=wp.vec3)
):  # pragma: no cover
    tid = wp.tid()  # pinned point index
    pid = pinned_indices[tid]  # global index of pinned point

    pinned_pos = pinned_vertices[tid]
    f[pid] += coeff * \
        boundary_gradient_wp_func(pinned_pos, vol[pid], dx[pid], x0[pid])


@wp.kernel
def boundary_hessian_wp_kernel(
    pinned_vertices: wp.array(dtype=wp.vec3),
    pinned_indices: wp.array(dtype=wp.int32),
    vol: wp.array(dtype=wp.float32),
    dx: wp.array(dtype=wp.vec3),
    x0: wp.array(dtype=wp.vec3),
    coeff: wp.float32,
    H_blocks: wp.array(dtype=wp.mat33)
):  # pragma: no cover
    tid = wp.tid()  # pinned point index
    pid = pinned_indices[tid]  # global index of pinned point

    pinned_pos = pinned_vertices[tid]
    H_blocks[pid] += coeff * boundary_hessian_wp_func(
        pinned_pos, vol[pid], dx[pid], x0[pid])


class Gravity:
    r""" Gravity class acts as a wrapper for the gravity energy, gradient, and hessian kernels.
    """

    def __init__(self, g, integration_pt_density, integration_pt_volume):
        r""" Initializes a Gravity object.
        
        Args:
            g (wp.vec3): Gravity acceleration (0, -9.81, 0)
            integration_pt_density (wp.array): Density at each point of size :math:`(\text{num_pts})`
            integration_pt_volume (wp.array): Volume at each point of size :math:`(\text{num_pts})`
        """

        # warp constant [x, y, z] acceleration due to gravity
        self.g = g
        # warp constant [N] density at each point
        self.integration_pt_density = integration_pt_density
        # warp constant [N] volume at each point
        self.integration_pt_volume = integration_pt_volume

        # warp constant [N, 9] hessians
        self.hessians_blocks = wp.zeros(
            integration_pt_volume.shape[0], dtype=wp.mat33, device=integration_pt_volume.device)
        self.sparse_hessian = None

    def energy(self, dx, x0, coeff, energy=None):
        r""" Returns the gravity energy at each integration primitive.
        
        Args:
            dx (wp.array): Delta in position of size :math:`(\text{num_pts}, 3)`
            x0 (wp.array): Rest position of size :math:`(\text{num_pts}, 3)`
            coeff (wp.float32): Coefficient
        
        Returns:
            wp.array(dtype=wp.float32): Gravity energy array of size :math:`1`
        """

        if energy is None:
            energy = wp.zeros(1, dtype=float)

        wp.launch(
            kernel=gravity_energy_wp_kernel,
            dim=dx.shape[0],
            inputs=[self.g,  # [gx, gy, gz] Acceleration due to gravity
                    self.integration_pt_density,
                    self.integration_pt_volume,
                    dx,
                    x0,
                    coeff],
            outputs=[energy],
            adjoint=False
        )

        return energy

    def gradient(self, dx, x0, coeff, gradients):
        r""" Returns the gravity gradient at each integration primitive.
        
        Args:
            dx (wp.array): Delta in position of size :math:`(\text{num_pts}, 3)`
            x0 (wp.array): Rest position of size :math:`(\text{num_pts}, 3)`
            coeff (wp.float32): Coefficient
        
        Returns:
            wp.array(dtype=wp.vec3): Gravity gradient of size :math:`(\text{num_pts}, 3)`
        """
        if gradients is None:
            gradients = wp.zeros_like(dx)
        wp.launch(
            kernel=gravity_gradient_wp_kernel,
            dim=dx.shape[0],
            inputs=[self.g,  # [gx, gy, gz] Acceleration due to gravity
                    self.integration_pt_density,
                    self.integration_pt_volume,
                    dx,
                    x0,
                    coeff],
            outputs=[gradients],
            adjoint=False
        )
        return gradients

    def hessian(self, dx, x0, coeff):
        r""" Returns the gravity hessian at each integration primitive.
        
        Args:
            dx (wp.array): Delta in position of size :math:`(\text{num_pts}, 3)`
            x0 (wp.array): Rest position of size :math:`(\text{num_pts}, 3)`
            coeff (wp.float32): Coefficient of hessian
        
        Returns:
            wp.array(dtype=wp.mat33): Gravity hessian of size :math:`(\text{num_pts}, 3, 3)`
        """
        return self.hessians_blocks


class Floor:
    r""" Floor class acts as a wrapper for the floor energy, gradient, and hessian kernels.
    """

    def __init__(self,
                 floor_height,
                 floor_axis,
                 flip_floor,
                 integration_pt_volume):
        r""" Initializes a Floor object.
        
        Args:
            floor_height (wp.float32): Floor height
            floor_axis (wp.int32): Floor axis (0-x, or 1-y, or 2-z)
            flip_floor (wp.int32): Flips the direction of the floor (0-False, 1-True)
            integration_pt_volume (wp.array): Volume at each point of size :math:`(\text{num_pts},)`
        """

        self.floor_height = floor_height
        self.floor_axis = floor_axis
        self.flip_floor = flip_floor
        self.integration_pt_volume = integration_pt_volume

        # pre-allocated hessians
        self.hessians_blocks = wp.zeros(
            integration_pt_volume.shape[0], dtype=wp.mat33, device=integration_pt_volume.device)

    def energy(self, dx, x0, coeff, energy=None):
        r""" Returns the floor energy at each integration primitive.
        
        Args:
            dx (wp.array): Delta in position of size :math:`(\text{num_pts}, 3)`
            x0 (wp.array): Rest position of size :math:`(\text{num_pts}, 3)`
            coeff (wp.float32): Coefficient
        
        Returns:
            wp.array(dtype=wp.float32): Floor energy array of size :math:`1`
        """

        if energy is None:
            energy = wp.zeros(1, dtype=float)
        wp.launch(
            kernel=floor_energy_wp_kernel,
            dim=dx.shape[0],
            inputs=[self.floor_height,
                    self.floor_axis,
                    self.flip_floor,
                    self.integration_pt_volume,
                    dx,
                    x0,
                    coeff],
            outputs=[energy],
            adjoint=False
        )
        return energy

    def gradient(self, dx, x0, coeff, gradients):
        r""" Returns the floor gradient at each integration primitive.
        
        Args:
            dx (wp.array): Delta in position of size :math:`(\text{num_pts}, 3)`
            x0 (wp.array): Rest position of size :math:`(\text{num_pts}, 3)`
            coeff (wp.float32): Coefficient
        
        Returns:
            wp.array(dtype=wp.vec3): Floor gradient of size :math:`(\text{num_pts}, 3)`
        """
        if gradients is None:
            gradients = wp.zeros_like(dx)
        wp.launch(
            kernel=floor_gradient_wp_kernel,
            dim=dx.shape[0],
            inputs=[self.floor_height,
                    self.floor_axis,
                    self.flip_floor,
                    self.integration_pt_volume,
                    dx,
                    x0,
                    coeff],
            outputs=[gradients],
            adjoint=False
        )
        return gradients

    def hessian(self, dx, x0, coeff):
        r""" Returns the floor hessian at each integration primitive.
        
        Args:
            dx (wp.array): Delta in position of size :math:`(\text{num_pts}, 3)`
            x0 (wp.array): Rest position of size :math:`(\text{num_pts}, 3)`
            coeff (wp.float32): Coefficient
        
        Returns:
            wp.array(dtype=wp.mat33): Floor hessian as an array of wp.mat33 block-wise matrices of size :math:`(\text{num_pts}, 3, 3)`
        """
        self.hessians_blocks.zero_()
        wp.launch(
            kernel=floor_hessian_wp_kernel,
            dim=dx.shape[0],
            inputs=[self.floor_height,
                    self.floor_axis,
                    self.flip_floor,
                    self.integration_pt_volume,
                    dx,
                    x0,
                    coeff],
            outputs=[self.hessians_blocks],
            adjoint=False
        )
        return self.hessians_blocks
        # H = wps.bsr_zeros(
        #     dx.shape[0], dx.shape[0], block_type=wp.mat33)

        # wps.bsr_set_diag(H, self.hessians_blocks)
        # torch.cuda.synchronize()
        # return H


class Boundary:
    r""" Boundary class acts as a wrapper for the boundary energy, gradient, and hessian kernels for all sample points in the scene.
    """

    def __init__(self, integration_pt_volume):
        r""" Initializes a Boundary object.
        
        Args:
            integration_pt_volume (wp.array(dtype=wp.float32)): Volume distributed across each point of size :math:`(\text{num_pts},)`
        """

        self.integration_pt_volume = integration_pt_volume
        self.pinned_vertices = None
        self.pinned_indices = None

        # pre-allocated hessians
        self.hessians_blocks = wp.zeros(
            integration_pt_volume.shape[0], dtype=wp.mat33, device=integration_pt_volume.device)

    def set_pinned(self, indices, pinned_x):
        r""" Sets the pinned vertices and indices.
        
        Args:
            indices (wp.array(dtype=wp.int32)): Indices of the pinned vertices of size :math:`(\text{num_pinned_pts},)`
            pinned_x (wp.array(dtype=wp.vec3)): Pinned vertices of size :math:`(\text{num_pinned_pts}, 3)`
        """
        assert pinned_x.shape[0] == indices.shape[0]
        self.pinned_indices = indices
        self.pinned_vertices = pinned_x

    def energy(self, dx, x0, coeff, energy=None):
        r""" Returns the boundary energy at each integration primitive.
        
        Args:
            dx (wp.array(dtype=wp.vec3)): Delta in position of size :math:`(\text{num_pts}, 3)`
            x0 (wp.array(dtype=wp.vec3)): Rest position of size :math:`(\text{num_pts}, 3)`
            coeff (float): Coefficient
            
        Returns:
            wp.array(dtype=wp.float32): Boundary energy array of size :math:`1`
        """

        if energy is None:
            energy = wp.zeros(1, dtype=float)

        wp.launch(
            kernel=boundary_energy_wp_kernel,
            dim=self.pinned_indices.shape[0],
            inputs=[self.pinned_vertices,
                    self.pinned_indices,
                    self.integration_pt_volume,
                    dx,
                    x0,
                    coeff],
            outputs=[energy],
            adjoint=False)

        return energy

    def gradient(self, dx, x0, coeff, gradients):
        r""" Returns the boundary gradient at each integration primitive.
        
        Args:
            dx (wp.array(dtype=wp.vec3)): Delta in position of size :math:`(\text{num_pts}, 3)`
            x0 (wp.array(dtype=wp.vec3)): Rest position of size :math:`(\text{num_pts}, 3)`
            coeff (float): Coefficient
        
        Returns:
            wp.array(dtype=wp.vec3): Boundary gradient of size :math:`(\text{num_pts}, 3)`
        """
        if gradients is None:
            gradients = wp.zeros_like(dx)
        wp.launch(
            kernel=boundary_gradient_wp_kernel,
            dim=self.pinned_indices.shape[0],
            inputs=[self.pinned_vertices,
                    self.pinned_indices,
                    self.integration_pt_volume,
                    dx,
                    x0,
                    coeff],
            outputs=[gradients],
            adjoint=False)
        return gradients

    def hessian(self, dx, x0, coeff):
        r""" Returns the boundary hessian at each integration primitive.
        
        Args:
            dx (wp.array(dtype=wp.vec3)): Delta in position of size :math:`(\text{num_pts}, 3)`
            x0 (wp.array(dtype=wp.vec3)): Rest position of size :math:`(\text{num_pts}, 3)`
            coeff (float): Coefficient
        
        Returns:
            wp.array(dtype=wp.mat33): Boundary hessian as a wp.mat33 block-wise sparse matrix of size :math:`(\text{num_pts}, 3, 3)`
        """
        self.hessians_blocks.zero_()
        wp.launch(
            kernel=boundary_hessian_wp_kernel,
            dim=self.pinned_indices.shape[0],
            inputs=[self.pinned_vertices, self.pinned_indices,
                    self.integration_pt_volume, dx, x0, coeff],
            outputs=[self.hessians_blocks],
            adjoint=False)

        return self.hessians_blocks