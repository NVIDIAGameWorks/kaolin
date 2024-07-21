import glob
import math
import copy
import os

import torch
from matplotlib import pyplot as plt
from tutorial_common import COMMON_DATA_DIR

import kaolin as kal
import nvdiffrast.torch as dr
from mediapy import write_image

glctx = dr.RasterizeCudaContext(device="cuda")

path = os.path.join(COMMON_DATA_DIR, 'meshes', 'avocado.gltf')

mesh = kal.io.gltf.import_mesh(path)

mesh = mesh.cuda()
mesh.vertices = kal.ops.pointcloud.center_points(
    mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)*2

print(mesh)
print(f'\nFirst material: \n {mesh.materials[0]}')

# Lighting parameters that are easy to control with sliders
azimuth = torch.zeros((1,), device='cuda')
elevation = torch.full((1,), math.pi / 3., device='cuda')
amplitude = torch.full((1, 3), 3., device='cuda')
sharpness = torch.full((1,), 5., device='cuda')

def current_lighting():
    """ Convert slider lighting parameters to paramater class used for rendering."""
    direction = kal.render.lighting.sg_direction_from_azimuth_elevation(azimuth, elevation)
    return kal.render.lighting.SgLightingParameters(
        amplitude=amplitude, sharpness=sharpness, direction=direction)

# Camera
# camera = kal.render.easy_render.default_camera(1024).cuda()
# camera = kal.render.camera.Camera.from_args(
#     eye=torch.tensor([0., 2., 0.], device='cuda'),
#     at=torch.tensor([0., 0., 0.], device='cuda'),
#     up=torch.tensor([0., 1., 0.], device='cuda'),
#     fov=math.pi * 30 / 180, width=1024, height=1024,
#     near=1e-2, far=1e2,
#     device='cuda'
# )

camera = kal.render.camera.Camera.from_args(
                eye=torch.tensor([0.0, 0.0, 1.0]),
                at=torch.tensor([0.0, 0.0, 0.0]),
                up=torch.tensor([0.0, 1.0, 0.0]),
                width=1024, height=1024,
                near=0.01, far=100,
                fov_distance=1.0,
                dtype=torch.float32,
                device='cuda'
            )

print(camera)


# face_idx = "face_idx"
# uvs = "uvs"
# albedo = "albedo"
# normals = "normals"
# roughness = "roughness"
# diffuse = "diffuse"
# specular = "specular"
# features = "features"
# render = "render"
# alpha = "alpha"

# Rendering
active_pass=kal.render.easy_render.RenderPass.render
def render(camera):
    """Render the mesh and its bundled materials.
    
    This is the main function provided to the interactive visualizer
    """
    render_res = kal.render.easy_render.render_mesh(camera, mesh, lighting=current_lighting(), nvdiffrast_context=glctx)
    img = render_res[active_pass]
    return {"img": (torch.clamp(img, 0., 1.)[0] * 255.).to(torch.uint8),
            "normals": render_res[kal.render.easy_render.RenderPass.normals][0]*0.5+0.5}
    
def lowres_render(camera):
    """Render with lower dimension.
    
    This function will be used as a "fast" rendering used when the mouse is moving to avoid slow down.
    """
    lowres_cam = copy.deepcopy(camera)
    lowres_cam.width = camera.width // 8
    lowres_cam.height = camera.height // 8
    return render(lowres_cam)

output = render(camera)
# plt.figure()

write_image('output_img.png', output['img'].cpu().numpy())

write_image('output_normals.png', output['normals'].cpu().numpy())