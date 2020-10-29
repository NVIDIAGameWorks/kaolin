"""
DIB-R: Forward pass rendering example
Uses DIB-R to render a mesh from a specified set of viewpoints.
"""

import argparse
import os

import imageio
import numpy as np
import torch
import tqdm

from kaolin.rep import TriangleMesh
from kaolin.graphics.dib_renderer.renderer import Renderer as DIBRenderer
from kaolin.graphics.dib_renderer.utils.sphericalcoord import get_spherical_coords_x


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=" DIB-R Example")

    parser.add_argument(
        "--use-texture", action="store_true", help="Whether to render a textured mesh."
    )
    args = parser.parse_args()

    # Camera settings.
    camera_distance = (
        2.0  # Distance of the camera from the origin (i.e., center of the object)
    )
    elevation = 30.0  # Angle of elevation
    azimuth = 0.0  # Azimuth angle

    # Directory in which sample data is located.
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "sampledata")

    # Load mesh
    mesh = TriangleMesh.from_obj(os.path.join(DATA_DIR, "banana.obj"))
    vertices = mesh.vertices.cuda()
    faces = mesh.faces.long().cuda()

    # Expand such that batch size = 1
    vertices = vertices.unsqueeze(0)

    # "Center" the mesh
    vertices_max = vertices.max()
    vertices_min = vertices.min()
    vertices_middle = (vertices_max + vertices_min) / 2.0
    # Scale the mesh by a factor of 5, so that it occupies a significant part
    # of the image (works only if banana.obj is used).
    vertices = (vertices - vertices_middle) * 5.0

    # Generate vertex color
    if not args.use_texture:
        vert_min = torch.min(vertices, dim=1, keepdims=True)[0]
        vert_max = torch.max(vertices, dim=1, keepdims=True)[0]
        colors = (vertices - vert_min) / (vert_max - vert_min)

    # Generate texture mapping
    if args.use_texture:
        uv = get_spherical_coords_x(vertices[0].cpu().numpy())
        uv = torch.from_numpy(uv).unsqueeze(0).cuda()

    # Load texture
    if args.use_texture:
        # Load image
        texture = imageio.imread(os.path.join(DATA_DIR, "texture.png"))
        texture = torch.from_numpy(texture).cuda()
        # Convert from [0, 255] to [0, 1]
        texture = texture.float() / 255.0
        # Convert to NxCxHxW layout
        texture = texture.permute(2, 0, 1).unsqueeze(0)

    # Initialize renderer
    if args.use_texture:
        renderer_mode = "Lambertian"
    else:
        renderer_mode = "VertexColor"

    renderer = DIBRenderer(256, 256, mode=renderer_mode)

    # Loop and render images
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    loop.set_description("Drawing")

    savename = (
        "dibr_rendered_vertexcolor.gif" if not args.use_texture else "dibr_rendered_texture.gif"
    )
    writer = imageio.get_writer(savename, mode="I")
    for azimuth in loop:
        renderer.set_look_at_parameters(
            [90 - azimuth], [elevation], [camera_distance]
        )

        if args.use_texture:
            predictions, _, _ = renderer(
                points=[vertices, faces.long()], uv_bxpx2=uv, texture_bx3xthxtw=texture
            )

        else:
            predictions, _, _ = renderer(
                points=[vertices, faces.long()], colors_bxpx3=colors
            )

        image = predictions.detach().cpu().numpy()[0]
        writer.append_data((image * 255).astype(np.uint8))

    writer.close()
