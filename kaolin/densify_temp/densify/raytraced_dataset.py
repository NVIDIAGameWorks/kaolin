import torch
from torch.utils.data import Dataset
from kaolin.render.camera import Camera
from kaolin.ops.spc import scan_octrees, generate_points, morton_to_points
from kaolin.render.spc import unbatched_raytrace, mark_pack_boundaries

from .raygen import generate_pinhole_rays, generate_centered_pixel_coords

class raytraced_dataset(Dataset):
    def __init__(self, viewpoints, gs_octree):

        self.viewpoints = viewpoints
        self.gs_octree = gs_octree

        lengths = torch.tensor([len(gs_octree)], dtype=torch.int32)
        self.level, self.pyramid, self.exsum = scan_octrees(gs_octree, lengths)
        self.point_hierarchy = generate_points(gs_octree, self.pyramid, self.exsum)

    def __len__(self):
        # When limiting the number of data
        return len(self.viewpoints)     

    def __getitem__(self, index):
        res = 2**self.level

        camera = Camera.from_args(
            eye=self.viewpoints[index],
            at=torch.tensor([0.0, 0.0, 0.0]),
            up=torch.tensor([0.0, 0.0, 1.0]),
            fov=0.644,  # In radians
            width=res, height=res,
            dtype=torch.float32,
            device='cuda'
        )

        ray_grid = generate_centered_pixel_coords(camera.width, camera.height, 
                                                camera.width, camera.height, device='cuda')
        origins, dirs = generate_pinhole_rays(camera, ray_grid)

        ridx, pidx, depths = unbatched_raytrace(self.gs_octree, self.point_hierarchy, self.pyramid[0], self.exsum, origins, dirs, 
                                                self.level, return_depth=True, with_exit=False)


        # get closest hits
        first_hits_mask = mark_pack_boundaries(ridx)
        first_hits_point = pidx[first_hits_mask]
        first_hits_ray = ridx[first_hits_mask].long()
        first_depths = depths[first_hits_mask]

        # black background
        image = torch.zeros((camera.height*camera.width, 3), dtype=torch.float32, device='cuda')  

        # write colors to image
        image[first_hits_ray,:] = 1.0
        image = image.reshape(camera.height, camera.width, 3)

        maxdepth = 6.0
        depthmap = torch.full((camera.height*camera.width, 1), maxdepth, dtype=torch.float32, device='cuda') 
        depthmap[first_hits_ray,:] = first_depths[:]
        depthmap = depthmap.reshape(camera.height, camera.width)

        cx = camera.intrinsics.cx
        cy = camera.intrinsics.cy
        fx = camera.intrinsics.focal_x
        fy = camera.intrinsics.focal_y

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
        
        mip_levels = 6

        # generate starting points
        start_level = 4
        points = morton_to_points(torch.arange(pow(8, start_level), device='cuda'))
        return image, depthmap, Cam, In, maxdepth, mip_levels, True, start_level, points



