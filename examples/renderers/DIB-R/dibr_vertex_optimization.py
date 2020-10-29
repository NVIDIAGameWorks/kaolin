"""
DIB-R: Vertex optimization example.
Uses DIB-R to optimize the vertices a sphere mesh to match a rendered banana image.
"""

import argparse
import os

import imageio
import numpy as np
import torch
from tqdm import tqdm, trange

from kaolin.rep import TriangleMesh
from kaolin.graphics.dib_renderer.renderer import Renderer as DIBRenderer

# Example script that uses SoftRas to deform a sphere mesh to aproximate
# the image of a banana.


class Model(torch.nn.Module):
    """Wrap vertices into an nn.Module, for optimization. """

    def __init__(self, vertices):
        super(Model, self).__init__()
        self.update = torch.nn.Parameter(torch.rand(vertices.shape) * 0.001)
        self.verts = vertices

    def forward(self):
        return self.update + self.verts


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Number of iterations to run optimization for.",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualization steps."
    )
    args = parser.parse_args()

    # Device to store tensors on. Must be a CUDA device.
    device = "cuda:0"

    # Initialize the soft rasterizer.
    renderer = DIBRenderer(256, 256, mode="VertexColor")

    # Camera settings.
    camera_distance = (
        2.0  # Distance of the camera from the origin (i.e., center of the object)
    )
    elevation = 30.0  # Angle of elevation
    azimuth = 0.0  # Azimuth angle

    # Directory in which sample data is located.
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "sampledata")

    # Read in the input mesh. TODO: Add filepath as argument.
    mesh = TriangleMesh.from_obj(os.path.join(DATA_DIR, "sphere.obj"))

    # Output filename to write out a rendered .gif to, showing the progress of optimization.
    progressfile = "vertex_optimization_progress.gif"
    # Output filename to write out a rendered .gif file to, rendering the optimized mesh.
    outfile = "vertex_optimization_output.gif"

    # Extract the vertices, faces, and texture the mesh (currently color with white).
    vertices = mesh.vertices
    faces = mesh.faces
    vertices = vertices[None, :, :].cuda()
    faces = faces[None, :, :].cuda()
    # Initialize all faces to yellow (to color the banana)!
    textures = torch.stack(
        (
            torch.ones(1, vertices.shape[-2], dtype=torch.float32, device=device),
            torch.ones(1, vertices.shape[-2], dtype=torch.float32, device=device),
            torch.zeros(1, vertices.shape[-2], dtype=torch.float32, device=device),
        ),
        dim=-1,
    )

    img_target = torch.from_numpy(
        imageio.imread(os.path.join(DATA_DIR, "banana.png")).astype(np.float32) / 255
    ).cuda()
    img_target = img_target[None, ...]  # .permute(0, 3, 1, 2)

    # Create a 'model' (an nn.Module) that wraps around the vertices, making it 'optimizable'.
    # TODO: Replace with a torch optimizer that takes vertices as a 'params' argument.
    # Deform the vertices slightly.
    model = Model(vertices.clone()).cuda()
    renderer.set_look_at_parameters([90 - azimuth], [elevation], [camera_distance])
    optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
    mseloss = torch.nn.MSELoss()

    # Perform vertex optimization.
    if not args.no_viz:
        writer = imageio.get_writer(progressfile, mode="I")
    for i in trange(args.iters):
        optimizer.zero_grad()
        new_vertices = model()
        img_pred, alpha, _ = renderer.forward(
            points=[new_vertices, faces[0].long()], colors_bxpx3=textures
        )
        rgba = torch.cat((img_pred, alpha), dim=-1)
        loss = mseloss(rgba, img_target)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            # TODO: Add functionality to write to gif output file.
            tqdm.write(f"Loss: {loss.item():.5}")
            if not args.no_viz:
                img = img_pred[0].detach().cpu().numpy()
                writer.append_data((255 * img).astype(np.uint8))
    if not args.no_viz:
        writer.close()

        # Write optimized mesh to output file.
        writer = imageio.get_writer(outfile, mode="I")
        for azimuth in trange(0, 360, 6):
            renderer.set_look_at_parameters(
                [90 - azimuth], [elevation], [camera_distance]
            )
            img_pred, alpha, _ = renderer.forward(
                points=[new_vertices, faces[0].long()], colors_bxpx3=textures
            )
            img = img_pred[0].detach().cpu().numpy()
            writer.append_data((255 * img).astype(np.uint8))
        writer.close()
