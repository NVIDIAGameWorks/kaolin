# ====================================================================================================================
# The following snippet demonstrates how cameras can be used for optimizing specific extrinsic / intrinsic parameters
# ====================================================================================================================

import torch
import torch.optim as optim
from kaolin.render.camera import Camera

# Create simple perspective camera
cam = Camera.from_args(
    eye=torch.tensor([4.0, 4.0, 4.0]),
    at=torch.tensor([0.0, 0.0, 0.0]),
    up=torch.tensor([0.0, 1.0, 0.0]),
    width=800, height=600, focal_x=300.0
)

# When requires_grad is on, the camera will automatically switch to differentiation friendly backend
# (implicitly calling cam.switch_backend('matrix_6dof_rotation') )
cam.requires_grad_(True)

# Constraint camera to optimize only fov and camera position (cannot rotate)
ext_mask, int_mask = cam.gradient_mask('t', 'focal_x', 'focal_y')
ext_params, int_params = cam.parameters()
ext_params.register_hook(lambda grad: grad * ext_mask.float())
grad_scale = 1e5    # Used to move the projection matrix elements faster
int_params.register_hook(lambda grad: grad * int_mask.float() * grad_scale)

# Make the camera a bit noisy
# Currently can't copy the camera here after requires_grad is true because we're still missing a camera.detach() op
target = Camera.from_args(
    eye=torch.tensor([4.0, 4.0, 4.0]),
    at=torch.tensor([0.0, 0.0, 0.0]),
    up=torch.tensor([0.0, 1.0, 0.0]),
    width=800, height=600, focal_x=300.0
)
target.t = target.t + torch.randn_like(target.t)
target.focal_x = target.focal_x + torch.randn_like(target.focal_x)
target.focal_y = target.focal_y + torch.randn_like(target.focal_y)
target_mat = target.view_projection_matrix()

# Save for later so we have some comparison of what changed
initial_view = cam.view_matrix().detach().clone()
initial_proj = cam.projection_matrix().detach().clone()

# Train a few steps
optimizer = optim.SGD(cam.parameters(), lr=0.1, momentum=0.9)
for idx in range(10):
    view_proj = cam.view_projection_matrix()
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(target_mat, view_proj)
    loss.backward()
    optimizer.step()
    print(f'Iteration {idx}:')
    print(f'Loss: {loss.item()}')
    print(f'Extrinsics: {cam.extrinsics.parameters()}')
    print(f'Intrinsics: {cam.intrinsics.parameters()}')

# Projection matrix grads are much smaller as they're scaled by the view-frustum dimensions..
print(f'View matrix before: {initial_view}')
print(f'View matrix after: {cam.view_matrix()}')
print(f'Projection matrix before: {initial_proj}')
print(f'Projection matrix after: {cam.projection_matrix()}')

print('Did the camera change?')
print(not torch.allclose(cam, target))
