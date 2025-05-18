import numpy as np
from kaolin.render.camera import Camera, camera_path_generator, loop_camera_path_generator


key_camera_1 = Camera.from_args(
    eye=[4.0, 4.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
    fov=np.deg2rad(45), width=800, height=800
)
key_camera_2 = Camera.from_args(
    eye=[8.0, 4.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
    fov=np.deg2rad(45), width=800, height=800
)
key_camera_3 = Camera.from_args(
    eye=[4.0, 8.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
    fov=np.deg2rad(45), width=800, height=800
)
key_camera_4 = Camera.from_args(
    eye=[4.0, 4.0, 8.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
    fov=np.deg2rad(45), width=800, height=800
)

print('============================================================================')
print('Camera trajectories interpolate additional camera motion between keyframes.')
print('i.e. adding frames_between_cameras=2 between each keyframe camera results in a trajectory of:')
print('X..X..X..X \n where:\n  X is the keyframe cameras \n  . are interpoalated cameras.')


print('============================================================================\n')
print('A path that moves exactly through the keyframe cameras:')
cam_path = camera_path_generator(
    trajectory=[key_camera_1, key_camera_2, key_camera_3, key_camera_4],
    frames_between_cameras=5,
    interpolation="catmull_rom"
)
for cam_idx, cam in enumerate(cam_path):
    cam_pos = cam.cam_pos().squeeze().cpu().numpy()
    print(f"Camera #{cam_idx} position: {cam_pos}")


print('============================================================================\n')
print('\nA cyclic path that moves smoothly and near, but not exactly through the keyframe cameras:')
cam_path = loop_camera_path_generator(
    trajectory=[key_camera_1, key_camera_2, key_camera_3, key_camera_4],
    frames_between_cameras=5,
    interpolation="polynomial",
    repeat=2    # or set to None for an infinite loop
)
for cam_idx, cam in enumerate(cam_path):
    cam_pos = cam.cam_pos().squeeze().cpu().numpy()
    print(f"Camera #{cam_idx} position: {cam_pos}")
