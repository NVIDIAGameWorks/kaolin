# ==============================================================================================================
# The following snippet demonstrates how to use the quaternion representation transforms
# ==============================================================================================================

import torch
from kaolin.math import quat

device = 'cuda'

# generate a batch of 2 identity quaternions
identity = quat.quat_identity([2], device=device)

# manually create a batch of 2 quaternions
quats = torch.tensor([[2, 3, 4, 1], [0, -4, -2, -10]], device=device)
print(quats)

# multiply the two quaternions
quats_mul = quat.quat_mul(identity, quats)
print(quats_mul)

# convert to 3x3 rotation matrix representation
rot33 = quat.rot33_from_quat(quats)
print(rot33)

# convert to angle-axis representation
angle, axis = quat.angle_axis_from_quat(quats)
print(angle, axis)

# convert the 3x3 rotation matrix to the matching angle-axis representation
angle2, axis2 = quat.angle_axis_from_rot33(rot33)
print(angle2, axis2)
print(torch.allclose(angle, angle2))
print(torch.allclose(axis, axis2))

# convert to 4x4 rotation matrix
rot44 = quat.rot44_from_quat(quats)
print(rot44)

# NOTE: there's plenty more conversions among these representations!

# compose a Euclidean transform matrix of the quaternion rotation and a translation
euclidean = quat.euclidean_from_rotation_translation(r=quats, t=torch.tensor([[1, 2, 3]]))
print(euclidean)

print(quat.euclidean_rotation_matrix(euclidean))  # get the rotation component
print(quat.euclidean_translation_vector(euclidean))  # or the translation component

# rotate a 3d point by the rotation represented by a tensor
point = torch.tensor([[1, 2, 3]], device=device)
quat_rot = quat.quat_rotate(quats, point)  # apply a batch of 2 rotations, represented by the two quats
print(quat_rot)
