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


import torch
from torch.utils.data import Dataset
from kaolin.ops.spc import scan_octrees, generate_points, morton_to_points
from kaolin.render.camera import Camera
from kaolin.render.spc import unbatched_raytrace, mark_pack_boundaries
from kaolin.render.camera import generate_rays


class RayTracedSPCDataset(Dataset):
    """
    A collection of ray intersections from predefined viewpoints with a structured point cloud (octree).
    Useful for, i.e., carving voxels with ray tracing.

    The return value of `__getitem__` will be:
        image (torch.FloatTensor) containing an rgb image of SPC object from input viewpoint, of size (camera.height, camera.width, 3),
        depthmap (torch.FloatTensor) containing distant from viewpoint to intersection for each pixel ray of size (camera.height*camera.width, 1),
        Cam (torch.FloatTensor) containing world to pixel space matrix of size (4,4),
        In (torch.FloatTensor) containing camera intrinsic matrix of size (4,4),
        max_depth (float) value of maximum depth in depth map,
        mip_levels (int) number of mip levels to construct
        TRUE_DEPTH (bool) indicates if depth map is actual depth, or instead z-buffer depth. Default is True,
        start_level (int) level to start carving algorithm
        Points (torch.ShortTensor) Initial set of points to begin carving. Default is dense octree of level=start_level
    """

    def __init__(self, viewpoints, gs_octree, res=8):
        self.viewpoints = viewpoints
        self.gs_octree = gs_octree

        lengths = torch.tensor([len(gs_octree)], dtype=torch.int32)
        self.level, self.pyramid, self.exsum = scan_octrees(gs_octree, lengths)
        self.point_hierarchy = generate_points(gs_octree, self.pyramid, self.exsum)
        self.device = self.gs_octree.device

        # Constants
        self.carve_camera_fov = 0.644  # In radians
        self.max_depth = torch.finfo(torch.float32).max
        self.mip_levels = 6
        self.start_level = 4
        self.res = res

    def __len__(self):
        # When limiting the number of data
        return len(self.viewpoints)     

    def __getitem__(self, index):
        res = 2**self.res


        eye = self.viewpoints[index]
        up = eye.new_tensor([0.0, 0.0, 1.0])
        at = eye.new_tensor([0.0, 0.0, 0.0])
        # Avoid degenerate coordinate systems if up and forward axes of camera are parallel
        if torch.allclose(torch.cross(up, at - eye, dim=-1), torch.zeros_like(eye)):
            up = eye.new_tensor([0.0, 1.0, 0.0])

        camera = Camera.from_args(
            eye=eye, at=at, up=up,
            fov=self.carve_camera_fov,
            width=res, height=res,
            dtype=torch.float32,
            device=self.device
        )

        origins, dirs = generate_rays(camera)
        ridx, pidx, depths = unbatched_raytrace(self.gs_octree, self.point_hierarchy, self.pyramid[0],
                                                self.exsum, origins, dirs,
                                                self.level, return_depth=True, with_exit=False)

        is_any_ray_hit = (len(ridx) > 0)

        if not is_any_ray_hit:
            return None, None, None, None, None, None, None, None, None, is_any_ray_hit

        # get closest hits
        first_hits_mask = mark_pack_boundaries(ridx)
        first_hits_ray = ridx[first_hits_mask].long()
        first_depths = depths[first_hits_mask]

        # black background
        image = torch.zeros((camera.height*camera.width, 3), dtype=torch.float32, device=self.device)

        # write colors to image
        image[first_hits_ray,:] = 1.0
        image = image.reshape(camera.height, camera.width, 3)

        depthmap = torch.full((camera.height*camera.width, 1), self.max_depth, dtype=torch.float32, device=self.device)
        depthmap[first_hits_ray,:] = first_depths[:]
        depthmap = depthmap.reshape(camera.height, camera.width)

        cx = camera.intrinsics.cx
        cy = camera.intrinsics.cy
        fx = camera.intrinsics.focal_x
        fy = camera.intrinsics.focal_y

        # Kaolin cameras use computer-graphics conventions of perspective division by w coordinate.
        # In the following we assume computer-vision conventions of perspective division by z.
        # Hence the view-projection matrix is constructed differently.
        In = torch.tensor([
            [fx,0,0,0],
            [0,fy,0,0],
            [cx,cy,1,0],
            [0,0,0,1]])
        
        Ex = camera.extrinsics.view_matrix().squeeze(0).to(torch.float).cpu().T

        G = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                        [ 0.0, -1.0, 0.0, 0.0],
                        [ 0.0, 0.0, -1.0, 0.0],
                        [ 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

        Ex = torch.mm(Ex, G)
        Cam = torch.mm(Ex, In)

        # generate starting points
        points = morton_to_points(torch.arange(pow(8, self.start_level), device=self.device))
        return image, depthmap, Cam, In, self.max_depth, self.mip_levels, True, self.start_level, points, is_any_ray_hit



