# ==============================================================================================================
# The following snippet demonstrates how to initialize instances of kaolin's pinhole / ortho cameras
# explicitly.
# Also review `camera_init_simple` which greatly simplifies the construction methods shown here.
# ==============================================================================================================

import math
import torch
from kaolin.render.camera import Camera, CameraExtrinsics, PinholeIntrinsics, OrthographicIntrinsics

#################################################################
#   Camera 1: from eye, at, up and focal length (Perspective)   #
#################################################################
# Build the camera extrinsics object from lookat
eye = torch.tensor([0.0, 0.0, -1.0], device='cuda') # Camera positioned here in world coords
at = torch.tensor([0.0, 0.0, 0.0], device='cuda')   # Camera observing this world point
up = torch.tensor([0.0, 1.0, 0.0], device='cuda')   # Camera up direction vector
extrinsics = CameraExtrinsics.from_lookat(eye, at, up)

# Build a pinhole camera's intrinsics. This time we use focal length (other useful args: focal_y, x0, y0)
intrinsics = PinholeIntrinsics.from_focal(width=800, height=600, focal_x=1.0, device='cuda')

# Combine extrinsics and intrinsics to obtain the full camera object
camera_1 = Camera(extrinsics=extrinsics, intrinsics=intrinsics)
print('--- Camera 1 ---')
print(camera_1)

########################################################################
#   Camera 2: from camera position, orientation and fov (Perspective)  #
########################################################################
# Build the camera extrinsics object from lookat
cam_pos = torch.tensor([0.0, 0.0, -1.0], device='cuda')
cam_dir = torch.tensor([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]], device='cuda')  # 3x3 orientation within the world
extrinsics = CameraExtrinsics.from_camera_pose(cam_pos=cam_pos, cam_dir=cam_dir)

# Use pinhole camera intrinsics, construct using field-of-view (other useful args: camera_fov_direction, x0, y0)
intrinsics = PinholeIntrinsics.from_fov(width=800, height=600, fov=math.radians(45.0), device='cuda')
camera_2 = Camera(extrinsics=extrinsics, intrinsics=intrinsics)

print('--- Camera 2 ---')
print(camera_2)

####################################################################
#   Camera 3: camera view matrix, (Orthographic)                   #
####################################################################
# Build the camera extrinsics object from lookat
world2cam = torch.tensor([[1.0, 0.0, 0.0, 0.5],
                          [0.0, 1.0, 0.0, 0.5],
                          [0.0, 0.0, 1.0, 0.5],
                          [0.0, 0.0, 0.0, 1.0]], device='cuda')  # 3x3 orientation within the world
extrinsics = CameraExtrinsics.from_view_matrix(view_matrix=world2cam)

# Use pinhole camera intrinsics, construct using field-of-view (other useful args: camera_fov_direction, x0, y0)
intrinsics = OrthographicIntrinsics.from_frustum(width=800, height=600, near=-800, far=800,
                                                 fov_distance=1.0, device='cuda')
camera_3 = Camera(extrinsics=extrinsics, intrinsics=intrinsics)

print('--- Camera 3 ---')
print(camera_3)


##########################################################
#   Camera 4: Combining cameras                          #
##########################################################
# Must be of the same intrinsics type, and non params fields such as width, height, near, far
# (currently we don't perform validation)
camera_4 = Camera.cat((camera_1, camera_2))

print('--- Camera 4 ---')
print(camera_4)


##########################################################
#   Camera 5: constructing a batch of cameras together   #
##########################################################

# Extrinsics are created using batched tensors. The intrinsics are automatically broadcasted.
camera_5 = Camera.from_args(
    eye=torch.tensor([[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]),
    at=torch.tensor([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]]),
    up=torch.tensor([[0.0, 1.0, 0.0], [4.0, 4.0, 4.0]]),
    width=800, height=600, focal_x=300.0
)

print('--- Camera 5 ---')
print(camera_5)