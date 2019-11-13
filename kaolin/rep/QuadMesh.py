# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from abc import abstractmethod

import torch

from kaolin.helpers import _composedecorator
from kaolin.rep.Mesh import Mesh
import numpy as np


class QuadMesh(Mesh):
    """ Abstract class to represent 3D Quad meshes. """

    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor,
                 uvs: torch.Tensor, face_textures: torch.Tensor,
                 textures: torch.Tensor, edges: torch.Tensor, edge2key: dict, vv: torch.Tensor,
                 vv_count: torch.Tensor, vf: torch.Tensor, vf_count: torch.Tensor,
                 ve: torch.Tensor, ve_count: torch.Tensor, ff: torch.Tensor,
                 ff_count: torch.Tensor, ef: torch.Tensor, ef_count: torch.Tensor,
                 ee: torch.Tensor, ee_count: torch.Tensor):

        # Vertices of the mesh
        self.vertices = vertices
        # Faces of the mesh
        self.faces = faces
        # uv coordinates of each vertex
        self.uvs = uvs
        # uv indecies for each face
        self.face_textures = face_textures
        # texture for each face
        self.textures = textures
        # Edges of the mesh
        self.edges = edges
        # Dictionary that maps an edge (tuple) to an edge idx
        self.edge2key = edge2key
        # Vertex-Vertex neighborhood tensor (for each vertex, contains
        # indices of the vertices neighboring it)
        self.vv = vv
        # Number of vertices neighbouring each vertex
        self.vv_count = vv_count
        # Vertex-Face neighborhood tensor
        self.vf = vf
        # Number of faces neighbouring each vertex
        self.vf_count = vf_count
        # Vertex-Edge neighborhood tensor
        self.ve = ve
        # Number of edges neighboring each vertex
        self.ve_count = ve_count
        # Face-Face neighborhood tensor
        self.ff = ff
        # Number of faces neighbouring each face
        self.ff_count = ff_count
        # Edge-Face neighbourhood tensor
        self.ef = ef
        # Number of edges neighbouring each face
        self.ef_count = ef_count
        # Edge-Edge neighbourhood tensor
        self.ee = ee
        # Number of edges neighbouring each edge
        self.ee_count = ee_count
        # adjacency matrix for verts
        self.adj = None

        # Initialize device on which tensors reside.
        self.device = self.vertices.device

    def save_mesh(self, filename):
        r""" Save a mesh to a wavefront .obj file format

            Args:
                filename (str) : target filename
                verts (FloatTensor) : vertices of the mesh
                faces (LongTensor) : list of vertex indexes for each face

            Example:
                >>> verts, faces = load_obj('object.obj')
                >>> verts = verts * 20
                >>> save_mesh('larger_object.obj', verts, faces)
        """

        with open(filename, 'w') as f:

            # write verts
            for vert in self.vertices:
                f.write('v %f %f %f\n' % tuple(vert))
            # write faces
            for face in self.faces:
                f.write('f %d %d %d %d\n' % tuple(face + 1))

    def sample(self, num_samples: int):
        r"""Uniformly samples the surface of a mesh.

            Args:
                num_samples (int): number of points to sample.

            Returns:
                points (torch.Tensor): uniformly sampled points
                face_choices (torch.Tensor): the face idexes which each point
                    corresponds to.

            Example:
                >>> points, chosen_faces = mesh.sample(10)
                >>> points
                tensor([[ 0.0293,  0.2179,  0.2168],
                        [ 0.2003, -0.3367,  0.2187],
                        [ 0.2152, -0.0943,  0.1907],
                        [-0.1852,  0.1686, -0.0522],
                        [-0.2167,  0.3171,  0.0737],
                        [ 0.2219, -0.0289,  0.1531],
                        [ 0.2217, -0.0115,  0.1247],
                        [-0.1400,  0.0364, -0.1618],
                        [ 0.0658, -0.0310, -0.2198],
                        [ 0.1926, -0.1867, -0.2153]])
                >>> chosen_faces
                tensor([ 953,   38,    6, 3480,  563,  393,  395, 3309,  373,  271])

        """

        if self.vertices.is_cuda:
            dist_uni = torch.distributions.Uniform(
                torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
        else:
            dist_uni = torch.distributions.Uniform(
                torch.tensor([0.0]), torch.tensor([1.0]))
        tri_faces_1 = torch.cat((self.faces[:, :2], self.faces[:, 3:]), dim=1)
        tri_faces_2 = torch.cat((self.faces[:, :1], self.faces[:, 2:]), dim=1)
        tri_faces = torch.cat((tri_faces_1, tri_faces_2))

        # calculate area of each face
        x1, x2, x3 = torch.split(torch.index_select(
            self.vertices, 0, tri_faces[:, 0]) - torch.index_select(self.vertices, 0, tri_faces[:, 1]), 1, dim=1)
        y1, y2, y3 = torch.split(torch.index_select(
            self.vertices, 0, tri_faces[:, 1]) - torch.index_select(self.vertices, 0, tri_faces[:, 2]), 1, dim=1)
        a = (x2 * y3 - x3 * y2)**2
        b = (x3 * y1 - x1 * y3)**2
        c = (x1 * y2 - x2 * y1)**2
        Areas = torch.sqrt(a + b + c) / 2
        # percentage of each face w.r.t. full surface area
        Areas = Areas / torch.sum(Areas)

        # define descrete distribution w.r.t. face area ratios caluclated
        cat_dist = torch.distributions.Categorical(Areas.view(-1))
        face_choices = cat_dist.sample([num_samples])

        # from each face sample a point
        select_faces = tri_faces[face_choices]
        xs = torch.index_select(self.vertices, 0, select_faces[:, 0])
        ys = torch.index_select(self.vertices, 0, select_faces[:, 1])
        zs = torch.index_select(self.vertices, 0, select_faces[:, 2])
        u = torch.sqrt(dist_uni.sample([num_samples]))
        v = dist_uni.sample([num_samples])
        points = (1 - u) * xs + (u * (1 - v)) * ys + u * v * zs

        # redefining face choices to match quad faces
        face_choices[face_choices >= self.faces.shape[
            0]] -= self.faces.shape[0]

        return points, face_choices

    @classmethod
    def compute_vertex_normals(self):
        raise NotImplementedError

    def compute_edge_lengths(self):
        raise NotImplementedError

    def compute_face_areas(self):
        raise NotImplementedError

    def compute_interior_angles_per_edge(self):
        raise NotImplementedError

    def compute_dihedral_angles_per_edge(self):
        raise NotImplementedError

    def load_tensors(filename: (str), enable_adjacency: bool = False):
        r"""Loads the tensor information of the mesh from a saved numpy array.

        Args:
            filename: the file name to load the file from.

        Example:
            >>> mesh = QuadMesh.load_tensors('mesh.npy')

        """
        data = np.load(filename)

        vertices = torch.FloatTensor(data['vertices'])
        faces = torch.LongTensor(data['faces'].astype(int))

        return QuadMesh.from_tensors(vertices, faces)

    def compute_adjacency_matrix_full(self):
        r""" calcualtes a binary adjacency matrix for a mesh

            Returns:
                (torch.Tensor) : binary adjacency matrix

            Example:
                >>> mesh = QuadMesh.from_obj('model.obj')
                >>> adj_info = mesh.compute_adjacency_matrix_full()
                >>> neighborhood_sum = torch.mm( adj_info, mesh.vertices)
        """

        adj = torch.zeros((self.vertices.shape[0], self.vertices.shape[0])).to(
            self.vertices.device)
        v1 = self.faces[:, 0]
        v2 = self.faces[:, 1]
        v3 = self.faces[:, 2]
        v4 = self.faces[:, 2]

        adj[(v1, v1)] = 1
        adj[(v2, v2)] = 1
        adj[(v3, v3)] = 1
        adj[(v4, v4)] = 1

        adj[(v1, v2)] = 1
        adj[(v2, v1)] = 1
        adj[(v1, v4)] = 1
        adj[(v4, v1)] = 1

        adj[(v3, v2)] = 1
        adj[(v2, v3)] = 1
        adj[(v3, v4)] = 1
        adj[(v4, v3)] = 1

        return adj

    def compute_adjacency_matrix_sparse(self):
        r""" Calcualtes a sparse adjacency matrix for a mess

            Returns:
                (torch.sparse.Tensor) : sparse adjacency matrix

            Example:
                >>> mesh = Mesh.from_obj('model.obj')
                >>> adj_info = mesh.compute_adjacency_matrix_sparse()
                >>> neighborhood_sum = torch.sparse.mm(adj_info, mesh.vertices)

        """

        if self.adj is None:

            v1 = self.faces[:, 0].view(-1, 1)
            v2 = self.faces[:, 1].view(-1, 1)
            v3 = self.faces[:, 2].view(-1, 1)
            v4 = self.faces[:, 2].view(-1, 1)

            vert_len = self.vertices.shape[0]
            identity_indices = torch.arange(vert_len).view(-1, 1).to(v1.device)
            identity = torch.cat(
                (identity_indices, identity_indices), dim=1).to(v1.device)
            identity = torch.cat((identity, identity))

            i_1 = torch.cat((v1, v2), dim=1)
            i_2 = torch.cat((v1, v4), dim=1)

            i_3 = torch.cat((v2, v1), dim=1)
            i_4 = torch.cat((v2, v3), dim=1)

            i_5 = torch.cat((v3, v2), dim=1)
            i_6 = torch.cat((v3, v4), dim=1)

            i_7 = torch.cat((v4, v3), dim=1)
            i_8 = torch.cat((v4, v1), dim=1)

            indices = torch.cat(
                (identity, i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8), dim=0).t()
            values = torch.ones(indices.shape[1]).to(indices.device) * .5
            self.adj = torch.sparse.FloatTensor(
                indices, values, torch.Size([vert_len, vert_len]))
        return self.adj.clone()
