# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import pytest
import torch
import warp as wp
import kaolin
from kaolin.physics.common import Gravity, Floor, Boundary


@pytest.fixture(params=['no_displacement', 'y_displacement', 'xy_displacement'])
def object_points(request):
    num_points = 20
    dx = torch.zeros(num_points, 3)

    if request.param == 'y_displacement':
        dx[:, 1] = 0.1
    elif request.param == 'xy_displacement':
        dx[:, 0] = 0.1
        dx[:, 1] = 0.2

    return {
        'x0': wp.array(torch.rand(num_points, 3), dtype=wp.vec3),
        'dx': wp.array(dx, dtype=wp.vec3), 
        'density': wp.array(torch.ones(num_points), dtype=wp.float32),
        'volume': wp.array(torch.ones(num_points)/num_points, dtype=wp.float32)
    }
    
    
# Gravity Tests
@pytest.mark.parametrize("object_points", [
    "object_points_no_displacement",
    "object_points_y_displacement",
    "object_points_xy_displacement"
], indirect=True)
def test_gravity_energy(object_points):
    
    g = wp.vec3(0.0, -9.81, 0.0)
    x0 = object_points['x0'] 
    dx = object_points['dx']
    density = object_points['density']
    volume = object_points['volume']
    
    gravity = Gravity(g, density, volume)
    energy = gravity.energy(dx, x0, 1.0)
    
    # Analytically calculate gravity energy
    points = wp.to_torch(x0) + wp.to_torch(dx)
    masses = wp.to_torch(density) * wp.to_torch(volume)
    expected_energy = 0.0
    for i in range(len(points)):
        expected_energy += masses[i] * (-9.81 * points[i,1])  # g.dot(p) where g = (0,-9.81,0)
    
    kaolin.utils.testing.check_allclose(wp.to_torch(energy)[0], expected_energy, rtol=1e-5), \
        "Gravity energy doesn't match analytical calculation"


@pytest.mark.parametrize("object_points", [
    "object_points_no_displacement",
    "object_points_y_displacement",
    "object_points_xy_displacement"
], indirect=True)
def test_gravity_gradient(object_points):
    g = wp.vec3(0.0, -9.81, 0.0)
    
    x0 = object_points['x0'] 
    dx = object_points['dx']
    density = object_points['density']
    volume = object_points['volume']
    
    gravity = Gravity(g, density, volume)
    gradient = gravity.gradient(dx, x0, 1.0, None)
    
    torch_gradient = wp.to_torch(gradient)
    torch_density = wp.to_torch(density)
    torch_volume = wp.to_torch(volume)
    for i in range(len(torch_gradient)):
        kaolin.utils.testing.check_allclose(torch_gradient[i, :], (torch_density[i]*torch_volume[i])*torch.tensor([0.0, -9.81, 0.0], device=torch_gradient.device), rtol=1e-5), "Gravity gradient y-component should be -9.81"


@pytest.mark.parametrize("object_points", [
    "object_points_no_displacement",
    "object_points_y_displacement",
    "object_points_xy_displacement"
], indirect=True)
def test_gravity_hessian(object_points):
    g = wp.vec3(0.0, -9.81, 0.0)
        
    x0 = object_points['x0'] 
    dx = object_points['dx']
    density = object_points['density']
    volume = object_points['volume']
    
    gravity = Gravity(g, density, volume)
    hessian = gravity.hessian(dx, x0, 1.0)
    assert (wp.to_torch(hessian) == 0).all(), "Gravity hessian should be zero"

# Floor Tests


@pytest.mark.parametrize("object_points", [
    "object_points_no_displacement",
    "object_points_y_displacement",
    "object_points_xy_displacement"
], indirect=True)
@pytest.mark.parametrize("floor_height", [-1.0, 0.0, 1.0])
def test_floor_energy(object_points, floor_height):
    floor_axis = 1
    flip_floor = 0
    floor = Floor(floor_height, floor_axis, flip_floor, object_points['volume'])  # y-axis floor at 0
    energy = floor.energy(object_points['dx'], object_points['x0'], 1.0)
    
    # Calculate expected energy analytically
    x0_torch = wp.to_torch(object_points['x0'])
    dx_torch = wp.to_torch(object_points['dx'])
    volume_torch = wp.to_torch(object_points['volume'])
    
    # Floor energy is sum of volume * penetration depth squared
    expected_energy = torch.as_tensor(0.0).to(x0_torch.device)
    for i in range(len(x0_torch)):
        points = x0_torch[i] + dx_torch[i]
        if points[floor_axis] < floor_height:
            penetration = (points[floor_axis] - floor_height)  # depth below floor
            expected_energy += volume_torch[i] * penetration**2

    kaolin.utils.testing.check_allclose(wp.to_torch(energy)[0], expected_energy, rtol=1e-5), "Floor energy does not match analytical calculation"


@pytest.mark.parametrize("object_points", [
    "object_points_no_displacement",
    "object_points_y_displacement",
    "object_points_xy_displacement"
], indirect=True)
@pytest.mark.parametrize("floor_height", [-1.0, 0.0, 1.0])
def test_floor_gradient(object_points, floor_height):
    floor_axis = 1
    flip_floor = 0
    floor = Floor(floor_height, floor_axis, flip_floor, object_points['volume'])
    gradient = floor.gradient(object_points['dx'], object_points['x0'], 1.0, None)
    
    torch_gradient = wp.to_torch(gradient)
    
    # Calculate expected gradient analytically
    x0_torch = wp.to_torch(object_points['x0'])
    dx_torch = wp.to_torch(object_points['dx']) 
    volume_torch = wp.to_torch(object_points['volume'])
    
    # Floor gradient is -k * d * V for points below floor
    # where d is penetration depth and V is volume
    expected_gradient = torch.zeros_like(torch_gradient)
    for i in range(len(x0_torch)):
        points = x0_torch[i] + dx_torch[i]
        if points[floor_axis] < floor_height:
            penetration = (points[floor_axis] - floor_height)  # depth below floor
            expected_gradient[i, floor_axis] = 2.0*penetration * volume_torch[i]
            
    kaolin.utils.testing.check_allclose(torch_gradient, expected_gradient, rtol=1e-5), "Floor gradient does not match analytical calculation"


@pytest.mark.parametrize("object_points", [
    "object_points_no_displacement",
    "object_points_y_displacement",
    "object_points_xy_displacement"
], indirect=True)
@pytest.mark.parametrize("floor_height", [-1.0, 0.0, 1.0])
def test_floor_hessian(object_points, floor_height):
    floor_axis = 1
    flip_floor = 0
    floor = Floor(floor_height, floor_axis, flip_floor, object_points['volume'])
    hessian = floor.hessian(object_points['dx'], object_points['x0'], 1.0)
    hess_torch = wp.to_torch(hessian)
    
    # Calculate expected hessian analytically
    x0_torch = wp.to_torch(object_points['x0'])
    dx_torch = wp.to_torch(object_points['dx']) 
    volume_torch = wp.to_torch(object_points['volume'])
    
    # Floor hessian is 2 * V for points below floor
    # where V is volume
    expected_hessian = torch.zeros_like(hess_torch)
    for i in range(len(x0_torch)):
        points = x0_torch[i] + dx_torch[i]
        if points[floor_axis] < floor_height:
            expected_hessian[i, floor_axis, floor_axis] = 2.0 * volume_torch[i]
    kaolin.utils.testing.check_allclose(hess_torch, expected_hessian, rtol=1e-5), "Floor hessian does not match analytical calculation"



# Boundary Tests
@pytest.mark.parametrize("object_points", [
    "object_points_no_displacement",
    "object_points_y_displacement",
    "object_points_xy_displacement"
], indirect=True)
def test_boundary_energy(object_points):
    # Calculate expected energy analytically
    x0_torch = wp.to_torch(object_points['x0'])
    dx_torch = wp.to_torch(object_points['dx'])
    volume_torch = wp.to_torch(object_points['volume'])
    
    # Function which returns a boolean vector of length x0_torch.shape[0] where pinned indices are 1
    # Pinned indices are the indices of points that are within 10% of the top of the bounding box in the y direction
    pinned_fcn = lambda x: x[:, 1] > 0.9*x[:, 1].max()
    pinned_indices = torch.nonzero(pinned_fcn(x0_torch), as_tuple=False).squeeze(1)
    
    # Calculate expected energy
    expected_energy = torch.as_tensor(0.0).to(x0_torch.device)
    for i in range(pinned_indices.shape[0]):
        idx = pinned_indices[i]
        pinned_pos = x0_torch[idx]
        n = dx_torch[idx] + x0_torch[idx] - pinned_pos
        expected_energy += torch.dot(n, n)
    
    
    boundary = Boundary(object_points['volume'])
    boundary.set_pinned(indices=wp.from_torch(pinned_indices.to(torch.int32), dtype=wp.int32), 
                        pinned_x=wp.from_torch(x0_torch[pinned_indices], dtype=wp.vec3))
    
    energy = boundary.energy(object_points['dx'], object_points['x0'], 1.0)
    
    assert torch.allclose(wp.to_torch(energy)[0], expected_energy, rtol=1e-4), "Boundary energy does not match analytical calculation"


@pytest.mark.parametrize("object_points", [
    "object_points_no_displacement",
    "object_points_y_displacement",
    "object_points_xy_displacement"
], indirect=True)
def test_boundary_gradient(object_points):
    # Calculate expected energy analytically
    x0_torch = wp.to_torch(object_points['x0'])
    dx_torch = wp.to_torch(object_points['dx'])
    volume_torch = wp.to_torch(object_points['volume'])

    # Function which returns a boolean vector of length x0_torch.shape[0] where pinned indices are 1
    # Pinned indices are the indices of points that are within 10% of the top of the bounding box in the y direction
    def pinned_fcn(x): return x[:, 1] > 0.9*x[:, 1].max()
    pinned_indices = torch.nonzero(
        pinned_fcn(x0_torch), as_tuple=False).squeeze(1)

    # Calculate expected gradient analytically
    expected_gradient = torch.zeros_like(x0_torch)
    for i in range(pinned_indices.shape[0]):
        idx = pinned_indices[i]
        pinned_pos = x0_torch[idx]
        n = dx_torch[idx] + x0_torch[idx] - pinned_pos
        expected_gradient[idx] += 2.0 * n

    boundary = Boundary(object_points['volume'])
    boundary.set_pinned(indices=wp.from_torch(pinned_indices.to(torch.int32), dtype=wp.int32),
                        pinned_x=wp.from_torch(x0_torch[pinned_indices], dtype=wp.vec3))

    gradient = boundary.gradient(object_points['dx'], object_points['x0'], 1.0, None)

    assert torch.allclose(wp.to_torch(gradient), expected_gradient, rtol=1e-4), "Boundary gradient does not match analytical calculation"


@pytest.mark.parametrize("object_points", [
    "object_points_no_displacement",
    "object_points_y_displacement",
    "object_points_xy_displacement"
], indirect=True)
def test_boundary_hessian(object_points):
    # Calculate expected energy analytically
    x0_torch = wp.to_torch(object_points['x0'])
    dx_torch = wp.to_torch(object_points['dx'])
    volume_torch = wp.to_torch(object_points['volume'])

    # Function which returns a boolean vector of length x0_torch.shape[0] where pinned indices are 1
    # Pinned indices are the indices of points that are within 10% of the top of the bounding box in the y direction
    def pinned_fcn(x): return x[:, 1] > 0.9*x[:, 1].max()
    pinned_indices = torch.nonzero(
        pinned_fcn(x0_torch), as_tuple=False).squeeze(1)

    boundary = Boundary(object_points['volume'])
    boundary.set_pinned(indices=wp.from_torch(pinned_indices.to(torch.int32), dtype=wp.int32),
                        pinned_x=wp.from_torch(x0_torch[pinned_indices], dtype=wp.vec3))

    hessian = boundary.hessian(object_points['dx'], object_points['x0'], 1.0)

    # Floor hessian is 2 * V for points below floor
    # where V is volume
    expected_hessian = torch.zeros_like(wp.to_torch(hessian))
    for i in range(pinned_indices.shape[0]):
        idx = pinned_indices[i]
        expected_hessian[idx, 0, 0] = 2.0
        expected_hessian[idx, 1, 1] = 2.0
        expected_hessian[idx, 2, 2] = 2.0
    
    assert torch.allclose(wp.to_torch(hessian), expected_hessian,
                          rtol=1e-5), "Boundary hessian does not match analytical calculation"
