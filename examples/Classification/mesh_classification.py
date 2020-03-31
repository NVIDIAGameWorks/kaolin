# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

from tqdm import tqdm, trange

import kaolin as kal
from kaolin.datasets import SHREC16
from kaolin.models import MeshCNNClassifier


torch.manual_seed(1234)

category1 = "ants"
category2 = "cat"

model = MeshCNNClassifier(5, 2, [16, 32, 32], [1140, 780, 580], 100, 0, 750)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lossfn = torch.nn.functional.nll_loss
dataset_train = SHREC16(
    "/home/jatavalk/data/shrec_16", categories=[category1, category2], mode="train"
)
dataset_test = SHREC16(
    "/home/jatavalk/data/shrec_16", categories=[category1, category2], mode="test"
)

epochs = 10


def prepare_mesh(obj_path):
    mesh = kal.rep.TriangleMesh.from_obj(obj_path)
    mesh.vertex_mask = torch.ones(mesh.vertices.shape[0]).to(mesh.vertices)
    _, face_areas = kal.models.meshcnn.compute_face_normals_and_areas(mesh)
    (
        edge2key,
        edges,
        vv,
        vv_count,
        ve,
        ve_count,
        vf,
        vf_count,
        ff,
        ff_count,
        ee,
        ee_count,
        ef,
        ef_count,
    ) = kal.rep.Mesh.compute_adjacency_info(mesh.vertices, mesh.faces)

    mesh.edges = edges
    mesh.ve = ve
    mesh.ve_count = ve_count
    mesh.ee = ee
    mesh.ee_count = ee_count

    kal.models.meshcnn.build_gemm_representation(mesh, face_areas)
    edge_points = kal.models.meshcnn.get_edge_points_vectorized(mesh)
    kal.models.meshcnn.set_edge_lengths(mesh, edge_points)
    kal.models.meshcnn.extract_meshcnn_features(mesh, edge_points)
    mesh.edges_count = mesh.edges.shape[-2]
    mesh.pool_count = 0

    mesh.edges = mesh.edges.numpy()
    mesh.faces = mesh.faces.numpy()
    mesh.ve = list(mesh.ve.numpy())
    mesh.ve = [
        list([ve[j] for j in range(mesh.ve_count[i])]) for i, ve in enumerate(mesh.ve)
    ]

    return mesh


model.train()
for e in trange(epochs):

    # Shuffle the dataset_train
    randperm = torch.randperm(len(dataset_train))

    # Train
    for idx in range(len(dataset_train)):

        i = randperm[idx]
        item = dataset_train[i]

        mesh = prepare_mesh(item["attributes"]["name"])

        # Some SHREC meshes are ill-behaved (roughly 8% of the dataset_train). We ignore them.
        x = None
        try:
            x = model(mesh.features.unsqueeze(0), [mesh])
        except Exception:
            pass
            # print("Skipping mesh:", idx)

        label = 0 if item["attributes"]["class"] == category1 else 1
        label = torch.tensor([label], dtype=torch.long, device=mesh.vertices.device)
        if x is not None:
            loss = lossfn(x, label)
            tqdm.write(f"Loss: {loss.item():.6}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

# Test
correct = 0
total = 0

model.eval()

for i in range(len(dataset_test)):

    item = dataset_test[i]
    mesh = prepare_mesh(item["attributes"]["name"])
    x = None
    try:
        x = model(mesh.features.unsqueeze(0), [mesh])
    except Exception:
        pass
    label = 0 if item["attributes"]["class"] == category1 else 1
    label = torch.tensor([label], dtype=torch.long, device=mesh.vertices.device)
    if x is not None:
        loss = lossfn(x, label)
        pred = x.argmax()
        if pred == label:
            correct += 1
        total += 1
print("Accuracy:", correct * 100 / total, "(%)")
