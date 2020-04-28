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
# MIT License

# Copyright (c) 2019 Rana Hanocka

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from heapq import heappop, heapify
from threading import Thread
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "MeshCNNClassifier",
    "compute_face_normals_and_areas",
    "extract_meshcnn_features",
]


def compute_face_normals_for_mesh(mesh):
    r"""Compute face normals for an input kaolin.rep.TriangleMesh object.

    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.

    Returns:
        face_normals (torch.Tensor): Tensor containing face normals for
            each triangle in the mesh (shape: :math:`(M, 3)`), where :math:`N`
            is the number of faces (triangles) in the mesh.
    """
    face_normals = torch.cross(
        mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]],
        mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 1]],
    )
    face_normals = face_normals / face_normals.norm(p=2, dim=-1)[..., None]
    return face_normals


def compute_face_normals_and_areas(mesh):
    r"""Compute face normals and areas for an input kaolin.rep.TriangleMesh object.

    Args:
       mesh (kaolin.rep.TriangleMesh): A triangle mesh object.

    Returns:
        face_normals (torch.Tensor): Tensor containing face normals for
            each triangle in the mesh (shape: :math:`(M, 3)`), where :math:`M`
            is the number of faces (triangles) in the mesh.
        face_areas (torch.Tensor): Tensor containing areas for each triangle
            in the mesh (shape: :math:`(M, 1)`), where :math:`M` is the number
            of faces (triangles) in the mesh.
    """
    face_normals = torch.cross(
        mesh.vertices[mesh.faces[:, 1]] - mesh.vertices[mesh.faces[:, 0]],
        mesh.vertices[mesh.faces[:, 2]] - mesh.vertices[mesh.faces[:, 1]],
    )
    face_normal_lengths = face_normals.norm(p=2, dim=-1)
    face_normals = face_normals / face_normal_lengths[..., None]
    # Recall: area of a triangle defined by vectors a and b is 0.5 * norm(cross(a, b))
    face_areas = 0.5 * face_normal_lengths
    return face_normals, face_areas


def is_two_manifold(mesh):
    """Returns whether the current mesh is 2-manifold. Assumes that adjacency info
    for the mesh is enabled.

    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object (assumes adjacency
            info is enabled).
    """
    return (mesh.ef.shape[-1] == 2) and (mesh.ef.min() >= 0)


def build_gemm_representation(mesh, face_areas):
    r"""Build a GeMM-suitable representation for the current mesh.

    The GeMM representation contains the following attributes:
        gemm_edges: tensor of four 1-ring neighbours per edge (E, 4)
        sides: tensor of indices (in the range [0, 3]) indicating the index of an edge
            in the gemm_edges entry of the 4 neighbouring edges.
        Eg. edge i => gemm_edges[gemm_edges[i], sides[i]] = [i, i, i, i]

    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh that is 2-manifold.
        face_areas (torch.Tensor): Areas of each triangle in the mesh
            (shape: :math:`(F)`, where :math:`F` is the number of faces).

    """
    # Retain first four neighbours for each edge (Needed esp if using newer
    # adjacency computation code).
    mesh.gemm_edges = mesh.ee[..., :4]
    # Compute the "sides" tensor
    mesh.sides = torch.zeros_like(mesh.gemm_edges)

    # TODO: Vectorize this!
    for i in range(mesh.gemm_edges.shape[-2]):
        for j in range(mesh.gemm_edges.shape[-1]):
            nbr = mesh.gemm_edges[i, j]
            ind = torch.nonzero(mesh.gemm_edges[nbr] == i)
            mesh.sides[i, j] = ind

    # Average area of all faces neighbouring an edge (normalized by the overall area of
    # the mesh). Weirdly, MeshCNN, computes averages by dividing by 3 (as opposed to
    # dividing by 2), and hence, we adopt their convention.
    mesh.edge_areas = face_areas[mesh.ef].sum(dim=-1) / (3 * face_areas.sum())


def get_edge_points_vectorized(mesh):
    r"""Get the edge points (a, b, c, d, e) as defined in Fig. 4 of the MeshCNN
    paper: https://arxiv.org/pdf/1809.05910.pdf.

    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.

    Returns:
        (torch.Tensor): Tensor containing "edge points" of the mesh, as per
            MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E` is the
            total number of edges in the mesh.
    """

    a = mesh.edges
    b = mesh.edges[mesh.gemm_edges[:, 0]]
    c = mesh.edges[mesh.gemm_edges[:, 1]]
    d = mesh.edges[mesh.gemm_edges[:, 2]]
    e = mesh.edges[mesh.gemm_edges[:, 3]]

    v1 = torch.zeros(a.shape[0]).bool().to(a.device)
    v2 = torch.zeros_like(v1)
    v3 = torch.zeros_like(v1)

    a_in_b = (a[:, 1] == b[:, 0]) + (a[:, 1] == b[:, 1])
    not_a_in_b = ~a_in_b
    a_in_b = a_in_b.long()
    not_a_in_b = not_a_in_b.long()
    b_in_c = ((b[:, 1] == c[:, 0]) + (b[:, 1] == c[:, 1])).long()
    d_in_e = ((d[:, 1] == e[:, 0]) + (d[:, 1] == e[:, 1])).long()

    arange = torch.arange(mesh.edges.shape[0]).to(a.device)

    return torch.stack(
        (
            a[arange, a_in_b],
            a[arange, not_a_in_b],
            b[arange, b_in_c],
            d[arange, d_in_e],
        ),
        dim=-1,
    )


def set_edge_lengths(mesh, edge_points):
    r"""Set edge lengths for each of the edge points. 

    Args:
       mesh (kaolin.rep.TriangleMesh): A triangle mesh object.
       edge_points (torch.Tensor): Tensor containing "edge points" of the mesh,
            as per MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E`
            is the total number of edges in the mesh.
    """
    mesh.edge_lengths = (
        mesh.vertices[edge_points[:, 0]] - mesh.vertices[edge_points[:, 1]]
    ).norm(p=2, dim=1)


def compute_normals_from_gemm(mesh, edge_points, side, eps=1e-1):
    r"""Compute vertex normals from the GeMM representation.

    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.
        edge_points (torch.Tensor): Tensor containing "edge points" of the mesh,
            as per MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E`
            is the total number of edges in the mesh.
        side (int): Side of the edge used in computing normals (0, 1, 2, or 3
            following MeshCNN convention).
        eps (float): A small number, for numerical stability (default: 1e-1,
            following MeshCNN implementation).

    Returns:
        (torch.Tensor): Face normals for each vertex on the chosen side of the
            edge (shape: :math:`(E, 2)`, where :math:`E` is the number of edges
            in the mesh).
    """
    a = (
        mesh.vertices[edge_points[:, side // 2 + 2]]
        - mesh.vertices[edge_points[:, side // 2]]
    )
    b = (
        mesh.vertices[edge_points[:, 1 - side // 2]]
        - mesh.vertices[edge_points[:, side // 2 + 2]]
    )
    normals = torch.cross(a, b)
    return normals / (normals.norm(p=2, dim=-1)[:, None] + eps)


def compute_dihedral_angles(mesh, edge_points):
    r"""Compute dihedral angle features for each edge. 

    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.
        edge_points (torch.Tensor): Tensor containing "edge points" of the mesh,
            as per MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E`
            is the total number of edges in the mesh.

    Returns:
        (torch.Tensor): Dihedral angle features for each edge in the mesh
            (shape: :math:`(E, 4)`, where :math:`E` is the number of edges
            in the mesh).
    """
    a = compute_normals_from_gemm(mesh, edge_points, 0)
    b = compute_normals_from_gemm(mesh, edge_points, 3)
    dot = (a * b).sum(dim=-1).clamp(-1, 1)
    return math.pi - torch.acos(dot)


def compute_opposite_angles(mesh, edge_points, side, eps=1e-1):
    r"""Compute opposite angle features for each edge.

    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.
        edge_points (torch.Tensor): Tensor containing "edge points" of the mesh,
            as per MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E`
            is the total number of edges in the mesh.
        side (int): Side of the edge used in computing normals (0, 1, 2, or 3
            following MeshCNN convention).
        eps (float): A small number, for numerical stability (default: 1e-1,
            following MeshCNN implementation).

    Returns:
        (torch.Tensor): Opposite angle features on the chosen side of the
            edge (shape: :math:`(E, 2)`, where :math:`E` is the number of edges
            in the mesh).
    """
    a = (
        mesh.vertices[edge_points[:, side // 2]]
        - mesh.vertices[edge_points[:, side // 2 + 2]]
    )
    b = (
        mesh.vertices[edge_points[:, 1 - side // 2]]
        - mesh.vertices[edge_points[:, side // 2 + 2]]
    )
    a = a / (a.norm(p=2, dim=-1)[:, None] + eps)
    b = b / (b.norm(p=2, dim=-1)[:, None] + eps)
    dot = (a * b).sum(dim=-1).clamp(-1, 1)
    return torch.acos(dot)


def compute_symmetric_opposite_angles(mesh, edge_points):
    r"""Compute symmetric opposite angle features for each edge.

    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.
        edge_points (torch.Tensor): Tensor containing "edge points" of the mesh,
            as per MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E`
            is the total number of edges in the mesh.

    Returns:
        (torch.Tensor): Symmetric opposite angle features for each edge in the mesh
            (shape: :math:`(E, 4)`, where :math:`E` is the number of edges
            in the mesh).
    """
    a = compute_opposite_angles(mesh, edge_points, 0)
    b = compute_opposite_angles(mesh, edge_points, 3)
    angles = torch.stack((a, b), dim=0)
    val, _ = torch.sort(angles, dim=0)
    return val


def compute_edgelength_ratios(mesh, edge_points, side, eps=1e-1):
    r"""Compute edge-length ratio features for each edge.

    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.
        edge_points (torch.Tensor): Tensor containing "edge points" of the mesh,
            as per MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E`
            is the total number of edges in the mesh.
        side (int): Side of the edge used in computing normals (0, 1, 2, or 3
            following MeshCNN convention).
        eps (float): A small number, for numerical stability (default: 1e-1,
            following MeshCNN implementation).

    Returns:
        (torch.Tensor): Edge-length ratio features on the chosen side of the
            edge (shape: :math:`(E, 2)`, where :math:`E` is the number of edges
            in the mesh).
    """
    edge_lengths = (
        mesh.vertices[edge_points[:, side // 2]]
        - mesh.vertices[edge_points[:, 1 - side // 2]]
    ).norm(p=2, dim=-1)
    o = mesh.vertices[edge_points[:, side // 2 + 2]]
    a = mesh.vertices[edge_points[:, side // 2]]
    b = mesh.vertices[edge_points[:, 1 - side // 2]]
    ab = b - a
    projection_length = (ab * (o - a)).sum(dim=-1) / (ab.norm(p=2, dim=-1) + eps)
    closest_point = a + (projection_length / edge_lengths)[:, None] * ab
    d = (o - closest_point).norm(p=2, dim=-1)
    return d / edge_lengths


def compute_symmetric_edgelength_ratios(mesh, edge_points):
    r"""Compute symmetric edge-length ratio features for each edge.

    Args:
        mesh (kaolin.rep.TriangleMesh): A triangle mesh object.
        edge_points (torch.Tensor): Tensor containing "edge points" of the mesh,
            as per MeshCNN convention (shape: :math:`(E, 4)`), where :math:`E`
            is the total number of edges in the mesh.

    Returns:
        (torch.Tensor): Symmetric edge-length ratio features for each edge in the mesh
            (shape: :math:`(E, 4)`, where :math:`E` is the number of edges
            in the mesh).
    """
    ratios_a = compute_edgelength_ratios(mesh, edge_points, 0)
    ratios_b = compute_edgelength_ratios(mesh, edge_points, 3)
    ratios = torch.stack((ratios_a, ratios_b), dim=0)
    val, _ = torch.sort(ratios, dim=0)
    return val


def extract_meshcnn_features(mesh, edge_points):
    r"""Extract the various features used by MeshCNN.

    Args:
        mesh (kaolin.rep.TriangleMesh): Input (2-manifold) triangle mesh.
        edge_points (torch.Tensor): Computed edge points from the input
            triangle mesh (following MeshCNN convention).
    """
    dihedral_angles = compute_dihedral_angles(mesh, edge_points).unsqueeze(0)
    symmetric_opposite_angles = compute_symmetric_opposite_angles(mesh, edge_points)
    symmetric_edgelength_ratios = compute_symmetric_edgelength_ratios(mesh, edge_points)
    mesh.features = torch.cat(
        (dihedral_angles, symmetric_opposite_angles, symmetric_edgelength_ratios), dim=0
    )


class MeshCNNConv(torch.nn.Module):
    r"""Implements the MeshCNN convolution operator. Recall that convolution is performed on the 1-ring
    neighbours of each (non-manifold) edge in the mesh.

    Args:
        in_channels (int): number of channels (features) in the input.
        out_channels (int): number of channels (features) in the output.
        kernel_size (int): kernel size of the filter.
        bias (bool, Optional): whether or not to use a bias term (default: True).

    .. note::

    If you use this code, please cite the original paper in addition to Kaolin.

    .. code-block::

        @article{meshcnn,
          title={MeshCNN: A Network with an Edge},
          author={Hanocka, Rana and Hertz, Amir and Fish, Noa and Giryes, Raja and Fleishman, Shachar and Cohen-Or, Daniel},
          journal={ACM Transactions on Graphics (TOG)},
          volume={38},
          number={4},
          pages = {90:1--90:12},
          year={2019},
          publisher={ACM}
        }
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: Optional[bool] = True,
    ):
        super(MeshCNNConv, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            bias=bias,
        )
        self.kernel_size = kernel_size

    def __call__(self, edge_features: torch.Tensor, meshes: Iterable):
        r"""Calls forward when invoked.

        Args:
            edge_features (torch.Tensor): input features of the mesh (shape: :math:`(B, F, E)`), where :math:`B` is
                the batchsize, :math:`F` is the number of features per edge, and :math:`E` is the number of edges
                in the input mesh.
            meshes (list[kaolin.rep.TriangleMesh]): list of TriangleMesh objects. Length of the list must be equal
                to the batchsize :math:`B` of `edge_features`.
        """
        return self.forward(edge_features, meshes)

    def forward(self, x, meshes):
        r"""Implements forward pass of the MeshCNN convolution operator.

        Args:
            x (torch.Tensor): input features of the mesh (shape: :math:`(B, F, E)`), where :math:`B` is
                the batchsize, :math:`F` is the number of features per edge, and :math:`E` is the number of edges
                in the input mesh.
            meshes (list[kaolin.rep.TriangleMesh]): list of TriangleMesh objects. Length of the list must be equal
                to the batchsize :math:`B` of `x`.
        """
        x = x.squeeze(-1)
        G = torch.cat([self.pad_gemm(i, x.shape[2], x.device) for i in meshes], 0)
        # MeshCNN "trick": Build a "neighbourhood map" and apply 2D convolution.
        G = self.create_gemm(x, G)
        return self.conv(G)

    def flatten_gemm_inds(self, Gi: torch.Tensor):
        r"""Flattens the indices of the gemm representation.
        """
        B, NE, NN = Gi.shape
        NE += 1
        batch_n = torch.floor(
            torch.arange(B * NE, device=Gi.device, dtype=torch.float) / NE
        ).view(B, NE)
        add_fac = batch_n * NE
        add_fac = add_fac.view(B, NE, 1)
        add_fac = add_fac.repeat(1, 1, NN)
        Gi = Gi.float() + add_fac[:, 1:, :]
        return Gi

    def create_gemm(self, x: torch.Tensor, Gi: torch.Tensor):
        r"""Gathers edge features (x) from within the 1-ring neighbours (Gi) and applies symmetric pooling for order
        invariance. Returns a "neighbourhood map" that we can use 2D convolution on.

        Args:
            x (torch.Tensor):
            Gi (torch.Tensor):
        """
        Gishape = Gi.shape
        # Zero-pad the first row of every sample in the batch.
        # TODO: Can replace by torch.nn.functional.pad()
        padding = torch.zeros(
            (x.shape[0], x.shape[1], 1),
            requires_grad=True,
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat((padding, x), dim=2)
        Gi = Gi + 1

        # Flatten indices
        Gi_flat = self.flatten_gemm_inds(Gi)
        Gi_flat = Gi_flat.view(-1).long()

        outdims = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(outdims[0] * outdims[2], outdims[1])

        f = torch.index_select(x, dim=0, index=Gi_flat)
        f = f.view(Gishape[0], Gishape[1], Gishape[2], -1)
        f = f.permute(0, 3, 1, 2)

        # Perform "symmetrization" (ops defined in paper) for the convolution to be "equivariant".
        x1 = f[:, :, :, 1] + f[:, :, :, 3]
        x2 = f[:, :, :, 2] + f[:, :, :, 4]
        x3 = (f[:, :, :, 1] - f[:, :, :, 3]).abs()
        x4 = (f[:, :, :, 2] - f[:, :, :, 4]).abs()
        return torch.stack((f[:, :, :, 0], x1, x2, x3, x4), dim=3)

    def pad_gemm(self, mesh, desired_size: int, device: torch.device):
        r"""Extracts the 1-ring neighbours (four per edge), adds the edge itself to the list, and pads to
        `desired_size`.

        Args:
            mesh (kaolin.rep.TriangleMesh): Mesh to convolve over.
            desired_size (int): Desired size to pad to.

        """
        # padded_gemm = torch.tensor(mesh.gemm_edges, device=device, dtype=torch.float, requires_grad=True)
        padded_gemm = mesh.gemm_edges.clone().float()
        padded_gemm.requires_grad = True
        # TODO: Revisit when batching is implemented, to update `mesh.edges.shape[1]`.
        num_edges = mesh.edges.shape[-2]
        padded_gemm = torch.cat(
            (
                torch.arange(num_edges, device=device, dtype=torch.float).unsqueeze(-1),
                padded_gemm,
            ),
            dim=1,
        )
        padded_gemm = F.pad(
            padded_gemm, (0, 0, 0, desired_size - num_edges), "constant", 0
        )
        return padded_gemm.unsqueeze(0)


class MeshCNNUnpool(torch.nn.Module):
    r"""Implements the MeshCNN unpooling operator.

    Args:
        unroll_target (int): number of target edges to unroll to.

    .. note::

    If you use this code, please cite the original paper in addition to Kaolin.

    .. code-block::

        @article{meshcnn,
          title={MeshCNN: A Network with an Edge},
          author={Hanocka, Rana and Hertz, Amir and Fish, Noa and Giryes, Raja and Fleishman, Shachar and Cohen-Or, Daniel},
          journal={ACM Transactions on Graphics (TOG)},
          volume={38},
          number={4},
          pages = {90:1--90:12},
          year={2019},
          publisher={ACM}
        }
    """

    def __init__(self, unroll_target: int):
        super(MeshCNNUnpool, self).__init__()
        self.unroll_target = unroll_target

    def __call__(self, features: torch.Tensor, meshes):
        return self.forward(features, meshes)

    def pad_groups(self, group: torch.Tensor, unroll_start: int):
        start, end = group.shape
        padding_rows = unroll_start - start
        padding_cols = self.unroll_target - end
        if padding_rows != 0 or padding_cols != 0:
            padding = torch.nn.ConstantPad2d((0, padding_cols, 0, padding_rows), 0)
            group = padding(group)
        return group

    def pad_occurrences(self, occurrences: torch.Tensor):
        padding = self.unroll_target - occurrences.shape[0]
        if padding != 0:
            padding = torch.nn.ConstantPad1d((0, padding), 1)
            occurrences = padding(occurrences)
        return occurrences

    def forward(self, features: torch.Tensor, meshes):
        r"""Implements forward pass of the MeshCNN convolution operator.

        Args:
            x (torch.Tensor): input features of the mesh (shape: :math:`(B, F, E)`), where :math:`B` is
                the batchsize, :math:`F` is the number of features per edge, and :math:`E` is the number of edges
                in the input mesh.
            meshes (list[kaolin.rep.TriangleMesh]): list of TriangleMesh objects. Length of the list must be equal
                to the batchsize :math:`B` of `x`.

        Returns:
            (torch.Tensor): output features, at the target unpooled size
                (shape: :math:`(B, F, \text{self.unroll_target})`)
        """
        B, F, E = features.shape
        groups = [self.pad_groups(mesh.get_groups(), E) for mesh in meshes]
        unroll_mat = torch.cat(groups, dim=0).view(B, E, -1)
        occurrences = [self.pad_occurrences(mesh.get_occurrences()) for mesh in meshes]
        occurrences = torch.cat(occurrences, dim=0).view(B, 1, -1)
        occurrences = occurrences.expand(unroll_mat.shape)
        unroll_mat = unroll_mat / occurrences
        urnoll_mat = unroll_mat.to(features)
        for mesh in meshes:
            mesh.unroll_gemm()
        return torch.matmul(features, unroll_mat)


class MeshUnion:
    r"""Implements the MeshCNN "union" operator.

    Args:
        num_edges (int): number of edges to attach (i.e., perform "union").
        device (torch.device): device on which tensors reside.

    .. note::

    If you use this code, please cite the original paper in addition to Kaolin.

    .. code-block::

        @article{meshcnn,
          title={MeshCNN: A Network with an Edge},
          author={Hanocka, Rana and Hertz, Amir and Fish, Noa and Giryes, Raja and Fleishman, Shachar and Cohen-Or, Daniel},
          journal={ACM Transactions on Graphics (TOG)},
          volume={38},
          number={4},
          pages = {90:1--90:12},
          year={2019},
          publisher={ACM}
        }
    """

    def __init__(self, num_edges: int, device: torch.device):
        self.groups = torch.eye(num_edges, device=device)
        self.rebuild_features = self.rebuild_features_average

    def union(self, source, target):
        self.groups[target, :] = self.groups[target, :] + self.groups[source, :]

    def remove_group(self, index):
        return

    def get_group(self, edge_key):
        return self.groups[edge_key, :]

    def get_occurrences(self):
        return torch.sum(self.groups, 0)

    def get_groups(self, mask):
        self.groups = self.groups.clamp(0, 1)
        return self.groups[mask, :]

    def rebuild_features_average(self, features, mask, target_edges):
        self.prepare_groups(features, mask)
        faces_edges = torch.matmul(features.squeeze(-1), self.groups)
        occurrences = torch.sum(self.groups, 0).expand(faces_edges.shape)
        faces_edges = faces_edges / occurrences
        padding = target_edges - faces_edges.shape[1]
        if padding > 0:
            padding = torch.nn.ConstantPad2d((0, padding, 0, 0), 0)
            faces_edges = padding(faces_edges)
        return faces_edges

    def prepare_groups(self, features, mask):
        mask = torch.from_numpy(mask)
        self.groups = self.groups[mask, :].clamp(0, 1).transpose_(1, 0)
        padding = features.shape[1] - self.groups.shape[0]
        if padding > 0:
            padding = torch.nn.ConstantPad2d((0, 0, 0, padding), 0)
            self.groups = padding(self.groups)


class MeshPool(torch.nn.Module):
    r"""Implements the MeshCNN pooling operator.

    Args:
        target (int): number of target edges to pool to.
        multi_thread (bool): Optionally run multi-threaded (default: False).

    .. note::

    If you use this code, please cite the original paper in addition to Kaolin.

    .. code-block::

        @article{meshcnn,
          title={MeshCNN: A Network with an Edge},
          author={Hanocka, Rana and Hertz, Amir and Fish, Noa and Giryes, Raja and Fleishman, Shachar and Cohen-Or, Daniel},
          journal={ACM Transactions on Graphics (TOG)},
          volume={38},
          number={4},
          pages = {90:1--90:12},
          year={2019},
          publisher={ACM}
        }
    """

    def __init__(self, target, multi_thread=False):
        super(MeshPool, self).__init__()
        self.__out_target = target
        self.__multi_thread = multi_thread
        self.__fe = None
        self.__updated_fe = None
        self.__meshes = None
        self.__merge_edges = [-1, -1]

    def __call__(self, fe, meshes):
        return self.forward(fe, meshes)

    def forward(self, fe, meshes):
        r"""Pool edges from the mesh and update features.

        Args:
            fe (torch.Tensor): Face-edge neighbourhood tensor.
            meshes (Iterable[kaolin.rep.TriangleMesh]): List of meshes to pool.

        Returns:
            out_features (torch.Tensor): Updated mesh features.

        """

        self.__updated_fe = [[] for _ in range(len(meshes))]
        pool_threads = []
        self.__fe = fe
        self.__meshes = meshes
        # iterate over batch
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:
                pool_threads.append(Thread(target=self.__pool_main, args=(mesh_index,)))
                pool_threads[-1].start()
            else:
                self.__pool_main(mesh_index)
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join()
        out_features = torch.cat(self.__updated_fe).view(
            len(meshes), -1, self.__out_target
        )
        return out_features

    def __pool_main(self, mesh_index):
        mesh = self.__meshes[mesh_index]
        queue = self.__build_queue(
            self.__fe[mesh_index, :, : mesh.edges_count], mesh.edges_count
        )
        # recycle = []
        # last_queue_len = len(queue)
        last_count = mesh.edges_count + 1
        mask = np.ones(mesh.edges_count, dtype=np.bool)
        edge_groups = MeshUnion(mesh.edges_count, self.__fe.device)
        while mesh.edges_count > self.__out_target:
            value, edge_id = heappop(queue)
            edge_id = int(edge_id)
            if mask[edge_id]:
                self.__pool_edge(mesh, edge_id, mask, edge_groups)
        self.clean(mesh, mask, edge_groups)
        fe = edge_groups.rebuild_features(
            self.__fe[mesh_index], mask, self.__out_target
        )
        self.__updated_fe[mesh_index] = fe

    def __pool_edge(self, mesh, edge_id, mask, edge_groups):
        if self.has_boundaries(mesh, edge_id):
            return False
        elif (
            self.__clean_side(mesh, edge_id, mask, edge_groups, 0)
            and self.__clean_side(mesh, edge_id, mask, edge_groups, 2)
            and self.__is_one_ring_valid(mesh, edge_id)
        ):
            self.__merge_edges[0] = self.__pool_side(
                mesh, edge_id, mask, edge_groups, 0
            )
            self.__merge_edges[1] = self.__pool_side(
                mesh, edge_id, mask, edge_groups, 2
            )
            self.merge_vertices(mesh, edge_id)
            mask[edge_id] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_id)
            mesh.edges_count -= 1
            return True
        else:
            return False

    def merge_vertices(self, mesh, edgeidx):
        self.remove_edge(mesh, edgeidx)
        edge = mesh.edges[edgeidx]
        v_a = mesh.vertices[edge[0]]
        v_b = mesh.vertices[edge[1]]
        v_a.__iadd__(v_b)
        v_a.__itruediv__(2)
        mesh.vertex_mask[edge[1]] = False
        mask = mesh.edges == edge[1]
        mesh.ve[edge[0]].extend(mesh.ve[edge[1]])
        mesh.edges[mask] = edge[0]

    def __clean_side(self, mesh, edge_id, mask, edge_groups, side):
        if mesh.edges_count <= self.__out_target:
            return False
        invalid_edges = MeshPool.__get_invalids(mesh, edge_id, edge_groups, side)
        while len(invalid_edges) != 0 and mesh.edges_count > self.__out_target:
            self.__remove_triplete(mesh, mask, edge_groups, invalid_edges)
            if mesh.edges_count <= self.__out_target:
                return False
            if self.has_boundaries(mesh, edge_id):
                return False
            invalid_edges = self.__get_invalids(mesh, edge_id, edge_groups, side)
        return True

    def clean(self, mesh, edges_mask, groups):
        edges_mask = edges_mask.astype(bool)
        torch_mask = torch.from_numpy(edges_mask.copy())
        mesh.gemm_edges = mesh.gemm_edges[edges_mask]
        mesh.edges = mesh.edges[edges_mask]
        mesh.sides = mesh.sides[edges_mask]
        new_ve = []
        edges_mask = np.concatenate([edges_mask, [False]])
        new_indices = np.zeros(edges_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        new_indices[edges_mask] = np.arange(0, np.ma.where(edges_mask)[0].shape[0])
        mesh.gemm_edges[:, :] = (
            torch.from_numpy(new_indices[mesh.gemm_edges[:, :]])
            .to(mesh.vertices.device)
            .long()
        )
        for v_index, ve in enumerate(mesh.ve):
            update_ve = []
            # if mesh.v_mask[v_index]:
            for e in ve:
                update_ve.append(new_indices[e])
            new_ve.append(update_ve)
        mesh.ve = new_ve
        mesh.pool_count += 1

    @staticmethod
    def has_boundaries(mesh, edge_id):
        for edge in mesh.gemm_edges[edge_id]:
            if edge == -1 or -1 in mesh.gemm_edges[edge]:
                return True
        return False

    @staticmethod
    def __is_one_ring_valid(mesh, edge_id):
        v_a = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 0]]].reshape(-1))
        v_b = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 1]]].reshape(-1))
        shared = v_a & v_b - set(mesh.edges[edge_id])
        return len(shared) == 2

    def __pool_side(self, mesh, edge_id, mask, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, _, other_side_b, _, other_keys_b = info
        self.__redirect_edges(
            mesh,
            key_a,
            side_a - side_a % 2,
            other_keys_b[0],
            mesh.sides[key_b, other_side_b],
        )
        self.__redirect_edges(
            mesh,
            key_a,
            side_a - side_a % 2 + 1,
            other_keys_b[1],
            mesh.sides[key_b, other_side_b + 1],
        )
        MeshPool.__union_groups(mesh, edge_groups, key_b, key_a)
        MeshPool.__union_groups(mesh, edge_groups, edge_id, key_a)
        mask[key_b] = False
        MeshPool.__remove_group(mesh, edge_groups, key_b)
        self.remove_edge(mesh, key_b)
        mesh.edges_count -= 1
        return key_a

    def remove_edge(self, mesh, edgeidx):
        vs = mesh.edges[edgeidx]
        for v in vs:
            mesh.ve[v].remove(edgeidx)

    @staticmethod
    def __get_invalids(mesh, edge_id, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        (
            key_a,
            key_b,
            side_a,
            side_b,
            other_side_a,
            other_side_b,
            other_keys_a,
            other_keys_b,
        ) = info
        shared_items = MeshPool.__get_shared_items(other_keys_a, other_keys_b)
        if len(shared_items) == 0:
            return []
        else:
            assert len(shared_items) == 2
            middle_edge = other_keys_a[shared_items[0]]
            update_key_a = other_keys_a[1 - shared_items[0]]
            update_key_b = other_keys_b[1 - shared_items[1]]
            update_side_a = mesh.sides[key_a, other_side_a + 1 - shared_items[0]]
            update_side_b = mesh.sides[key_b, other_side_b + 1 - shared_items[1]]
            MeshPool.__redirect_edges(mesh, edge_id, side, update_key_a, update_side_a)
            MeshPool.__redirect_edges(
                mesh, edge_id, side + 1, update_key_b, update_side_b
            )
            MeshPool.__redirect_edges(
                mesh,
                update_key_a,
                MeshPool.__get_other_side(update_side_a),
                update_key_b,
                MeshPool.__get_other_side(update_side_b),
            )
            MeshPool.__union_groups(mesh, edge_groups, key_a, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_b, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_a, update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, key_b, update_key_b)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_b)
            return [key_a, key_b, middle_edge]

    @staticmethod
    def __redirect_edges(mesh, edge_a_key, side_a, edge_b_key, side_b):
        mesh.gemm_edges[edge_a_key, side_a] = edge_b_key
        mesh.gemm_edges[edge_b_key, side_b] = edge_a_key
        mesh.sides[edge_a_key, side_a] = side_b
        mesh.sides[edge_b_key, side_b] = side_a

    @staticmethod
    def __get_shared_items(list_a, list_b):
        shared_items = []
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if list_a[i] == list_b[j]:
                    shared_items.extend([i, j])
        return shared_items

    @staticmethod
    def __get_other_side(side):
        return side + 1 - 2 * (side % 2)

    @staticmethod
    def __get_face_info(mesh, edge_id, side):
        key_a = mesh.gemm_edges[edge_id, side]
        key_b = mesh.gemm_edges[edge_id, side + 1]
        side_a = mesh.sides[edge_id, side]
        side_b = mesh.sides[edge_id, side + 1]
        other_side_a = (side_a - (side_a % 2) + 2) % 4
        other_side_b = (side_b - (side_b % 2) + 2) % 4
        other_keys_a = [
            mesh.gemm_edges[key_a, other_side_a],
            mesh.gemm_edges[key_a, other_side_a + 1],
        ]
        other_keys_b = [
            mesh.gemm_edges[key_b, other_side_b],
            mesh.gemm_edges[key_b, other_side_b + 1],
        ]
        return (
            key_a,
            key_b,
            side_a,
            side_b,
            other_side_a,
            other_side_b,
            other_keys_a,
            other_keys_b,
        )

    @staticmethod
    def __remove_triplete(mesh, mask, edge_groups, invalid_edges):
        vertex = set(mesh.edges[invalid_edges[0]])
        for edge_key in invalid_edges:
            vertex &= set(mesh.edges[edge_key])
            mask[edge_key] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_key)
        mesh.edges_count -= 3
        vertex = list(vertex)
        assert len(vertex) == 1
        mesh.vertex_mask[vertex[0]] = False
        # mesh.remove_vertex(vertex[0])

    def __build_queue(self, features, edges_count):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        edge_ids = torch.arange(
            edges_count, device=squared_magnitude.device, dtype=torch.float32
        ).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, edge_ids), dim=-1).tolist()
        heapify(heap)
        return heap

    @staticmethod
    def __union_groups(mesh, edge_groups, source, target):
        edge_groups.union(source, target)
        # mesh.union_groups(source, target)

    @staticmethod
    def __remove_group(mesh, edge_groups, index):
        edge_groups.remove_group(index)
        # mesh.remove_group(index)


class MResConv(torch.nn.Module):
    r"""Implements a residual block of MeshCNNConv layers.

    Args:
        in_channels (int): number of channels (features) in the input.
        out_channels (int): number of channels (features) in the output.
        skip (Optional, int): number of skip connected layers to add (default: 1).
        kernel_size (Optional, int): kernel size of the (2D) conv filter. (default: 5).

    .. note::

    If you use this code, please cite the original paper in addition to Kaolin.

    .. code-block::

        @article{meshcnn,
          title={MeshCNN: A Network with an Edge},
          author={Hanocka, Rana and Hertz, Amir and Fish, Noa and Giryes, Raja and Fleishman, Shachar and Cohen-Or, Daniel},
          journal={ACM Transactions on Graphics (TOG)},
          volume={38},
          number={4},
          pages = {90:1--90:12},
          year={2019},
          publisher={ACM}
        }
    """

    def __init__(self, in_channels, out_channels, skip=1, kernel_size=5):
        super(MResConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip = 1
        self.conv0 = MeshCNNConv(
            self.in_channels, self.out_channels, kernel_size=kernel_size, bias=False
        )
        for i in range(self.skip):
            setattr(self, f"bn{i + 1}", torch.nn.BatchNorm2d(self.out_channels))
            setattr(
                self,
                f"conv{i + 1}",
                MeshCNNConv(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=kernel_size,
                    bias=False,
                ),
            )

    def forward(self, x, mesh):
        x = self.conv0(x, mesh)
        x1 = x
        for i in range(self.skip):
            x = getattr(self, f"bn{i + 1}")(F.relu(x))
            x = getattr(self, f"conv{i + 1}")(x, mesh)
        x = x + x1
        return F.relu(x)


class MeshCNNClassifier(torch.nn.Module):
    r"""Implements a MeshCNN classifier.

    Args:
        in_channels (int): number of channels (features) in the input.
        out_channels (int): number of channels (features) in the output (usually equal
            to the number of classes).
        conv_layer_sizes (Iterable): List of sizes of residual MeshCNNConv blocks to
            be used.
        pool_sizes (Iterable): Target number of edges in the mesh after each pooling
            step.
        fc_size (int): Number of neurons in the penultimate fully-connected layer.
        num_res_blocks (int): Number of residual blocks to use in the classifier.
        num_edges_in (int): Number of edges in the input mesh.

    .. note::

    If you use this code, please cite the original paper in addition to Kaolin.

    .. code-block::

        @article{meshcnn,
          title={MeshCNN: A Network with an Edge},
          author={Hanocka, Rana and Hertz, Amir and Fish, Noa and Giryes, Raja and Fleishman, Shachar and Cohen-Or, Daniel},
          journal={ACM Transactions on Graphics (TOG)},
          volume={38},
          number={4},
          pages = {90:1--90:12},
          year={2019},
          publisher={ACM}
        }
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_layer_sizes,
        pool_sizes,
        fc_size,
        num_res_blocks,
        num_edges_in,
    ):
        super(MeshCNNClassifier, self).__init__()
        self.layer_sizes = [in_channels] + conv_layer_sizes
        self.edge_sizes = [num_edges_in] + pool_sizes

        for i, size in enumerate(self.layer_sizes[:-1]):
            setattr(
                self,
                f"conv{i}",
                MResConv(size, self.layer_sizes[i + 1], num_res_blocks),
            )
            setattr(self, f"pool{i}", MeshPool(self.edge_sizes[i + 1]))

        self.global_pooling = torch.nn.AvgPool1d(self.edge_sizes[-1])
        self.fc1 = torch.nn.Linear(self.layer_sizes[-1], fc_size)
        self.fc2 = torch.nn.Linear(fc_size, out_channels)

    def forward(self, x, mesh):
        for i in range(len(self.layer_sizes) - 1):
            x = F.relu(getattr(self, f"conv{i}")(x, mesh))
            x = getattr(self, f"pool{i}")(x, mesh)
        x = self.global_pooling(x)
        x = x.view(-1, self.layer_sizes[-1])
        x = F.relu(self.fc1(x))
        return self.fc2(x)
