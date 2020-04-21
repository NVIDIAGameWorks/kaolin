# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#
#
# Soft Rasterizer (SoftRas)
#
# Copyright (c) 2017 Hiroharu Kato
# Copyright (c) 2018 Nikos Kolotouros
# Copyright (c) 2019 Shichen Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import imageio
import torch

import kaolin

# Example script that uses SoftRas to render an image, given a mesh input


class Model(nn.Module):
    """Wrap textures into an nn.Module, for optimization. """

    def __init__(self, textures):
        super(Model, self).__init__()
        self.textures = textures

    def forward(self):
        return torch.sigmoid(self.textures)


if __name__ == "__main__":

    # Initialize the soft rasterizer.
    renderer = kaolin.graphics.SoftRenderer(camera_mode="look_at", device="cuda:0")

    # Read in the input mesh. TODO: Add filepath as argument.
    mesh = kaolin.rep.TriangleMesh.from_obj("tests/model.obj")

    # Extract the vertices, faces, and texture the mesh (currently color with white).
    vertices = mesh.vertices.float()
    faces = mesh.faces.long()
    face_textures = faces.clone()
    vertices = vertices[None, :, :].cuda()
    faces = faces[None, :, :].cuda()
    face_textures = face_textures[None, :, :].cuda()
    textures = torch.ones(1, faces.shape[1], 2, 3, dtype=torch.float32).cuda()

    # TODO: Normalize vertices (Use kaolin functionality to do this)
    vertices = vertices - 0.5 * (vertices.max() - vertices.min())
    # Scale the mesh to a "reasonable" size (TODO: avoid this "magic" number)
    vertices = 5 * vertices

    # TODO: Fix path. Add obj to kaolin.
    mesh_target = kaolin.rep.TriangleMesh.from_obj("data/banana.obj")
    # TODO: Fix path
    img_target = torch.from_numpy(
        imageio.imread("data/target.png").astype(np.float32).mean(-1) / 255
    )[None, ::].cuda()

    # Create a 'model' (an nn.Module) that wraps around the vertices, making it 'optimizable'.
    # TODO: Replace with a torch optimizer that takes vertices as a 'params' argument.
    # Deform the vertices slightly.
    model = Model(textures).cuda()
    # renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
    optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.5, 0.99))
    mseloss = torch.nn.MSELoss()

    # Perform vertex optimization.
    for i in tqdm.tqdm(2000):
        optimizer.zero_grad()
        textures = model()
        rgb, _, _ = renderer(vertices, faces, new_textures)
        loss = mseloss(rgb, img_target[None, :, :])
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            # TODO: Add functionality to write to gif output file.
            tqdm.write("Loss: " + str(loss.item()) + "\n")
