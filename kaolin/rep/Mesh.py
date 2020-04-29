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

from typing import Optional

from abc import abstractmethod
import os
from PIL import Image

import torch
import numpy as np

from kaolin.helpers import _assert_tensor
from kaolin.helpers import _composedecorator

import kaolin.cuda.load_textures as load_textures_cuda
import kaolin as kal


class Mesh():
    """ Abstract class to represent 3D polygon meshes. """

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

    @classmethod
    def from_tensors(cls, vertices: torch.Tensor, faces: torch.Tensor,
                     uvs: torch.Tensor = None,
                     face_textures: torch.Tensor = None,
                     textures: torch.Tensor = None, enable_adjacency=False):
        r"""Returns mesh with supplied tensor information.

        Args:
            vertices (torch.Tensor): mesh vertices.
            faces (torch.Tensor): mesh faces.
            uvs (torch.Tensor): uv coordinates for the vertices in mesh.
            face_textures (torch.Tensor): uv number for each face's vertices.
            textures (torch.Tensor):  texture info for each face.
            enable_adjacency (torch.Tensor): adjacency information is computed
        """
        vertices = vertices.clone()
        faces = faces.clone()
        if enable_adjacency:
            edge2key, edges, vv, vv_count, ve, ve_count, vf, vf_count, ff, \
                ff_count, ee, ee_count, ef, ef_count = \
                cls.compute_adjacency_info(vertices, faces)
            return cls(vertices, faces, uvs, face_textures, textures, edges,
                       edge2key, vv, vv_count, vf, vf_count, ve, ve_count,
                       ff, ff_count, ef, ef_count, ee, ee_count)
        else:
            return cls(vertices, faces, uvs, face_textures, textures,
                       None, None, None, None, None, None, None, None,
                       None, None, None, None, None, None)

    @_composedecorator(classmethod, abstractmethod)
    def from_obj(self, filename: str, with_vt: bool = False,
                 enable_adjacency: bool = False, texture_res=4):
        r"""Loads object in .obj wavefront format.

        Args:
            filename (str) : location of file.
            with_vt (bool): objects loaded with textures specified by vertex
                textures.
            enable_adjacency (bool): adjacency information is computed.
            texture_res (int): resolution of loaded face colors.

        Note: the with_vt parameter requires cuda.

        Example:
            >>> mesh = Mesh.from_obj('model.obj')
            >>> mesh.vertices.shape
            torch.Size([482, 3])
            >>> mesh.faces.shape
            torch.Size([960, 3])

        """
        # run through obj file and extract obj info
        vertices = []
        faces = []
        face_textures = []
        uvs = []
        with open(filename, 'r') as mesh:
            for line in mesh:
                data = line.split()
                if len(data) == 0:
                    continue
                if data[0] == 'v':
                    vertices.append(data[1:])
                elif data[0] == 'vt':
                    uvs.append(data[1:3])
                elif data[0] == 'f':
                    if '//' in data[1]:
                        data = [da.split('//') for da in data]
                        faces.append([int(d[0]) for d in data[1:]])
                        face_textures.append([int(d[1]) for d in data[1:]])
                    elif '/' in data[1]:
                        data = [da.split('/') for da in data]
                        faces.append([int(d[0]) for d in data[1:]])
                        face_textures.append([int(d[1]) for d in data[1:]])
                    else:
                        faces.append([int(d) for d in data[1:]])
                        continue
        vertices = torch.FloatTensor([float(el) for sublist in vertices for el in sublist]).view(-1, 3)
        faces = torch.LongTensor(faces) - 1

        # compute texture info
        textures = None
        if with_vt:
            with open(filename, 'r') as f:
                textures = None
                for line in f:
                    if line.startswith('mtllib'):
                        filename_mtl = os.path.join(
                            os.path.dirname(filename), line.split()[1])
                        textures = self.load_textures(
                            filename, filename_mtl, texture_res)

                f.close()

        if len(uvs) > 0:
            uvs = torch.FloatTensor([float(el) for sublist in uvs for el in sublist]).view(-1, 2)
        else:
            uvs = None
        if len(face_textures) > 0:
            face_textures = torch.LongTensor(face_textures) - 1
        else:
            face_textures = None

        if enable_adjacency:
            edge2key, edges, vv, vv_count, ve, ve_count, vf, vf_count, ff, ff_count, \
                ee, ee_count, ef, ef_count = self.compute_adjacency_info(
                    vertices, faces)
        else:
            edge2key, edges, vv, vv_count, ve, ve_count, vf, vf_count, ff, \
                ff_count, ee, ee_count, ef, ef_count = None, None, None, \
                None, None, None, None, None, None, None, None, None, None, \
                None

        output = self(vertices, faces, uvs, face_textures, textures, edges,
                    edge2key, vv, vv_count, vf, vf_count, ve, ve_count, ff, ff_count,
                    ef, ef_count, ee, ee_count)
        return output

    @classmethod
    def from_off(self, filename: str,
                 enable_adjacency: Optional[bool] = False):
        r"""Loads a mesh from a .off file.

        Args:
            filename (str): Path to the .off file.
            enable_adjacency (str): Whether or not to compute adjacency info.

        Returns:
            (kaolin.rep.Mesh): Mesh object.

        """
        vertices = []
        faces = []
        num_vertices = 0
        num_faces = 0
        num_edges = 0
        # Flag to store the number of vertices, faces, and edges that have
        # been read.
        read_vertices = 0
        read_faces = 0
        read_edgs = 0
        # Flag to indicate whether or not metadata (number of vertices,
        # number of faces, (optionally) number of edges) has been read.
        # For .off files, metadata is the first valid line of each file
        # (neglecting the "OFF" header).
        metadata_read = False
        with open(filename, 'r') as infile:
            for line in infile.readlines():
                # Ignore comments
                if line.startswith('#'):
                    continue
                if line.startswith('OFF'):
                    continue
                data = line.strip().split()
                data = [da for da in data if len(da) > 0]
                # Ignore blank lines
                if len(data) == 0:
                    continue
                if metadata_read is False:
                    num_vertices = int(data[0])
                    num_faces = int(data[1])
                    if len(data) == 3:
                        num_edges = int(data[2])
                    metadata_read = True
                    continue
                if read_vertices < num_vertices:
                    vertices.append([float(d) for d in data])
                    read_vertices += 1
                    continue
                if read_faces < num_faces:
                    numedges = int(data[0])
                    faces.append([int(d) for d in data[1:1+numedges]])
                    read_faces += 1
                    continue
                if read_edges < num_edges:
                    edges.append([int(d) for d in data[1:]])
                    read_edges += 1
                    continue
        vertices = torch.FloatTensor(np.array(vertices, dtype=np.float32))
        faces = torch.LongTensor(np.array(faces, dtype=np.int64))

        if enable_adjacency:
            edge2key, edges, vv, vv_count, ve, ve_count, vf, vf_count, ff, ff_count, \
                ee, ee_count, ef, ef_count = self.compute_adjacency_info(
                    vertices, faces)
        else:
            edge2key, edges, vv, vv_count, ve, ve_count, vf, vf_count, ff, \
                ff_count, ee, ee_count, ef, ef_count = None, None, None, \
                None, None, None, None, None, None, None, None, None, None, \
                None

        return self(vertices, faces, None, None, None, edges,
                    edge2key, vv, vv_count, vf, vf_count, ve, ve_count, ff, ff_count,
                    ef, ef_count, ee, ee_count)

    @staticmethod
    def _cuda_helper(tensor):
        if tensor is not None:
            return tensor.cuda()

    @staticmethod
    def _cpu_helper(tensor):
        if tensor is not None:
            return tensor.cpu()

    @staticmethod
    def _to_helper(tensor, device):
        if tensor is not None:
            return tensor.to(device)

    def cuda(self):
        r""""Maps all tensors of the current class to CUDA. """

        self.vertices = self._cuda_helper(self.vertices)

        self.faces = self._cuda_helper(self.faces)
        self.uvs = self._cuda_helper(self.uvs)
        self.face_textures = self._cuda_helper(self.face_textures)
        self.textures = self._cuda_helper(self.textures)
        self.edges = self._cuda_helper(self.edges)
        self.vv = self._cuda_helper(self.vv)
        self.vv_count = self._cuda_helper(self.vv_count)
        self.vf = self._cuda_helper(self.vf)
        self.vf_count = self._cuda_helper(self.vf_count)
        self.ve = self._cuda_helper(self.ve)
        self.ve_count = self._cuda_helper(self.ve_count)
        self.ff = self._cuda_helper(self.ff)
        self.ff_count = self._cuda_helper(self.ff_count)
        self.ef = self._cuda_helper(self.ef)
        self.ef_count = self._cuda_helper(self.ef_count)
        self.ee = self._cuda_helper(self.ee)
        self.ee_count = self._cuda_helper(self.ee_count)

        self.device = self.vertices.device

    def cpu(self):
        r""""Maps all tensors of the current class to CPU. """

        self.vertices = self._cpu_helper(self.vertices)

        self.faces = self._cpu_helper(self.faces)
        self.uvs = self._cpu_helper(self.uvs)
        self.face_textures = self._cpu_helper(self.face_textures)
        self.textures = self._cpu_helper(self.textures)
        self.edges = self._cpu_helper(self.edges)
        self.vv = self._cpu_helper(self.vv)
        self.vv_count = self._cpu_helper(self.vv_count)
        self.vf = self._cpu_helper(self.vf)
        self.vf_count = self._cpu_helper(self.vf_count)
        self.ve = self._cpu_helper(self.ve)
        self.ve_count = self._cpu_helper(self.ve_count)
        self.ff = self._cpu_helper(self.ff)
        self.ff_count = self._cpu_helper(self.ff_count)
        self.ef = self._cpu_helper(self.ef)
        self.ef_count = self._cpu_helper(self.ef_count)
        self.ee = self._cpu_helper(self.ee)
        self.ee_count = self._cpu_helper(self.ee_count)

        self.device = self.vertices.device

    def to(self, device):
        r"""Maps all tensors of the current class to the specified device. """

        self.vertices = self._to_helper(self.vertices, device)

        self.faces = self._to_helper(self.faces, device)
        self.uvs = self._to_helper(self.uvs, device)
        self.face_textures = self._to_helper(self.face_textures, device)
        self.textures = self._to_helper(self.textures, device)
        self.edges = self._to_helper(self.edges, device)
        self.vv = self._to_helper(self.vv, device)
        self.vv_count = self._to_helper(self.vv_count, device)
        self.vf = self._to_helper(self.vf, device)
        self.vf_count = self._to_helper(self.vf_count, device)
        self.ve = self._to_helper(self.ve, device)
        self.ve_count = self._to_helper(self.ve_count, device)
        self.ff = self._to_helper(self.ff, device)
        self.ff_count = self._to_helper(self.ff_count, device)
        self.ef = self._to_helper(self.ef, device)
        self.ef_count = self._to_helper(self.ef_count, device)
        self.ee = self._to_helper(self.ee, device)
        self.ee_count = self._to_helper(self.ee_count, device)

        self.device = self.vertices.device


    @staticmethod
    def load_mtl(filename_mtl: str):
        r""" Returns all colours and texture files found in an mtl files.

        Args:
                filename_mtl (str) : mtl file name

        """
        texture_filenames = {}
        colors = {}
        material_name = ''
        with open(filename_mtl) as f:
            for line in f.readlines():
                if len(line.split()) != 0:
                    if line.split()[0] == 'newmtl':
                        material_name = line.split()[1]
                    if line.split()[0] == 'map_Kd':
                        texture_filenames[material_name] = line.split()[1]
                    if line.split()[0] == 'Kd':
                        colors[material_name] = np.array(
                            list(map(float, line.split()[1:4])))
        return colors, texture_filenames

    @classmethod
    def load_textures(self, filename_obj: str, filename_mtl: str,
                      texture_res: int):
        r""" Returns texture for a given obj file, where texture is
        defined using vertex texture uvs.

        Args:
            filename_obj (str) : obj file name
            filename_mtl (str) : mtl file name
            texture_res  (int) : texture resolution for each face


        Returns:
           textures (torch.Tensor) : texture values for each face

        """
        assert torch.cuda.is_available()
        vertices = []
        with open(filename_obj) as f:
            lines = f.readlines()
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'vt':
                vertices.append([float(v) for v in line.split()[1:3]])
        vertices = np.vstack(vertices).astype(np.float32)

        # load faces for textures
        faces = []
        material_names = []
        material_name = ''
        for line in lines:
            if len(line.split()) == 0:
                continue
            if line.split()[0] == 'f':
                vs = line.split()[1:]
                nv = len(vs)
                if '/' in vs[0] and '//' not in vs[0]:
                    v0 = int(vs[0].split('/')[1])
                else:
                    v0 = 0
                for i in range(nv - 2):
                    if '/' in vs[i + 1] and '//' not in vs[i + 1]:
                        v1 = int(vs[i + 1].split('/')[1])
                    else:
                        v1 = 0
                    if '/' in vs[i + 2] and '//' not in vs[i + 2]:
                        v2 = int(vs[i + 2].split('/')[1])
                    else:
                        v2 = 0
                    faces.append((v0, v1, v2))
                    material_names.append(material_name)
            if line.split()[0] == 'usemtl':
                material_name = line.split()[1]
        faces = np.vstack(faces).astype(np.int32) - 1
        faces = vertices[faces]
        faces = torch.from_numpy(faces).cuda()
        faces[1 < faces] = faces[1 < faces] % 1

        colors, texture_filenames = self.load_mtl(filename_mtl)
        textures = torch.ones(
            faces.shape[0], texture_res**2, 3, dtype=torch.float32)
        textures = textures.cuda()

        for material_name, color in list(colors.items()):
            color = torch.from_numpy(color).cuda()
            for i, material_name_f in enumerate(material_names):
                if material_name == material_name_f:
                    textures[i, :, :] = color[None, :]

        for material_name, filename_texture in list(texture_filenames.items()):
            filename_texture = os.path.join(
                os.path.dirname(filename_obj), filename_texture)
            image = np.array(Image.open(filename_texture)
                             ).astype(np.float32) / 255.

            # texture image may have one channel (grey color)
            if len(image.shape) == 2:
                image = np.stack((image,) * 3, -1)
            # or has extral alpha channel shoule ignore for now
            if image.shape[2] == 4:
                image = image[:, :, :3]

            # pytorch does not support negative slicing for the moment
            image = image[::-1, :, :]
            image = torch.from_numpy(image.copy()).cuda()
            is_update = (np.array(material_names)
                         == material_name).astype(np.int32)
            is_update = torch.from_numpy(is_update).cuda()
            textures = load_textures_cuda.load_textures(
                image, faces, textures, is_update)
        return textures

    @staticmethod
    def get_edges_from_face(f: torch.Tensor):
        """Returns a list of edges forming the current face.

        Args:
            f: Face (quadruplet of indices into 'vertices').
            vertices (torch.Tensor): Vertices (3D points).

        Returns:
            edge_inds (list): List of tuples (a, b) for each edge (a, b) in
                faces.
        """
        _assert_tensor(f)
        n = f.numel()
        edges = []
        for i in range(n):
            if f[i] < f[(i + 1) % n]:
                edges.append((f[i].item(), f[(i + 1) % n].item()))
            else:
                edges.append((f[(i + 1) % n].item(), f[i].item()))
        return edges

    @staticmethod
    def get_edge_order(a: int, b: int):
        """ Returns (a, b) or (b, a), depending on which is smaller.
        (Smaller element first, for unique keys)

        Args:
            a (int): Index of first vertex in edge.
            b (int): Index of second vertex in edge.

        """
        return (a, b) if a < b else (b, a)

    @staticmethod
    def has_common_vertex(e1: torch.Tensor, e2: torch.Tensor):
        r"""Returns True if the vertices e1, e2 share a common vertex,
        False otherwise.

        Args:
            e1 (torch.Tensor): First edge (shape: :math:`2`).
            e2 (torch.Tensor): Second edge (shape: :math: `2`).

        Returns:
            (bool): Whether or not e1 and e2 share a common vertex.

        """
        return (e1[0] in e2) or (e1[1] in e2)

    @staticmethod
    def get_common_vertex(e1: torch.Tensor, e2: torch.Tensor):
        r"""Returns the common vertex in edges e1 and e2 (if any).

        Args:
            e1 (torch.Tensor): First edge (shape: :math:`2`).
            e2 (torch.Tensor): Second edge (shape: :math:`2`).

        Returns:
            common_vertex (torch.LongTensor): Index of common vertex
                    (shape: :math:`1`).
            first_nbr (torch.LongTensor): Index of one neighbouring
                    vertex of the common vertex (shape: :math:`1`).
            second_nbr (torch.LongTensor): Index of the other neighbouring
                    vertex of the common vertex (shape: :math:`1`).

        """
        if e1[0] == e2[0]:
            return e1[0], e1[1], e2[1]
        if e1[0] == e2[1]:
            return e1[0], e1[1], e2[0]
        if e1[1] == e2[0]:
            return e1[1], e1[0], e2[1]
        if e1[1] == e2[1]:
            return e1[1], e1[0], e2[0]
        return None, None, None

    @staticmethod
    def list_of_lists_to_matrix(
            list_of_lists: list, sublist_lengths: torch.Tensor, matrix: torch.Tensor):
        r"""Takes a list of lists (each sub-list of variable size), and maps it
        to a matrix. Decorated by numba, for efficiency sake.

        Args:
            list_of_lists (list): A list containing 'sub-'lists (Note: the sub-list
                    cannont contain lists; needs to contain numbers).
            sublist_lengths (torch.Tensor): Array containing lengths of each sublist.
            matrix (torch.Tensor): Matrix in which to `mould` the list
                    (Note: the matrix must contain as many columns as required to
                    encapsulate the largest sub-list of `list_of_lists`).

        """
        for i in range(matrix.shape[0]):
            l = sublist_lengths[i]
            if l > 0:
                matrix[i, 0:l] = list_of_lists[i]
        return matrix

    @staticmethod
    def compute_adjacency_info(vertices: torch.Tensor, faces: torch.Tensor):
        """Build data structures to help speed up connectivity queries. Assumes
        a homogeneous mesh, i.e., each face has the same number of vertices.

        The outputs have the following format: AA, AA_count
        AA_count: [count_0, ..., count_n]
        with AA:
        [[aa_{0,0}, ..., aa_{0,count_0} (, -1, ..., -1)],
         [aa_{1,0}, ..., aa_{1,count_1} (, -1, ..., -1)],
                    ...
         [aa_{n,0}, ..., aa_{n,count_n} (, -1, ..., -1)]]
        """

        device = vertices.device
        facesize = faces.shape[1]
        nb_vertices = vertices.shape[0]
        nb_faces = faces.shape[0]
        edges = torch.cat([faces[:,i:i+2] for i in range(facesize - 1)] +
                          [faces[:,[-1,0]]], dim=0)
        # Sort the vertex of edges in increasing order
        edges = torch.sort(edges, dim=1)[0]
        # id of corresponding face in edges
        face_ids = torch.arange(nb_faces, device=device, dtype=torch.long).repeat(facesize)
        # remove multiple occurences and sort by the first vertex
        # the edge key / id is fixed from now as the first axis position
        # edges_ids will give the key of the edges on the original vector
        edges, edges_ids = torch.unique(edges, sorted=True, return_inverse=True, dim=0)
        nb_edges = edges.shape[0]

        # EDGE2EDGES
        _edges_ids = edges_ids.reshape(facesize, nb_faces)
        edges2edges = torch.cat([
            torch.stack([_edges_ids[1:], _edges_ids[:-1]], dim=-1).reshape(-1, 2),
            torch.stack([_edges_ids[-1:], _edges_ids[:1]], dim=-1).reshape(-1, 2)
        ], dim=0)

        double_edges2edges = torch.cat([edges2edges, torch.flip(edges2edges, dims=(1,))], dim=0)
        double_edges2edges = torch.cat(
            [double_edges2edges, torch.arange(double_edges2edges.shape[0], device=device, dtype=torch.long).reshape(-1, 1)], dim=1)
        double_edges2edges = torch.unique(double_edges2edges, sorted=True, dim=0)[:,:2]
        idx_first = torch.where(
            torch.nn.functional.pad(double_edges2edges[1:,0] != double_edges2edges[:-1,0],
                                    (1, 0), value=1))[0]
        nb_edges_per_edge = idx_first[1:] - idx_first[:-1]
        offsets = torch.zeros(double_edges2edges.shape[0], device=device, dtype=torch.long)
        offsets[idx_first[1:]] = nb_edges_per_edge
        sub_idx = (torch.arange(double_edges2edges.shape[0], device=device,dtype=torch.long) -
                   torch.cumsum(offsets, dim=0))
        nb_edges_per_edge = torch.cat([nb_edges_per_edge,
                                       double_edges2edges.shape[0] - idx_first[-1:]],
                                      dim=0)
        max_sub_idx = torch.max(nb_edges_per_edge)
        ee = torch.full((nb_edges, max_sub_idx), device=device, dtype=torch.long, fill_value=-1)
        ee[double_edges2edges[:,0], sub_idx] = double_edges2edges[:,1]

        # EDGE2FACE
        sorted_edges_ids, order_edges_ids = torch.sort(edges_ids)
        sorted_faces_ids = face_ids[order_edges_ids]
        # indices of first occurences of each key
        idx_first = torch.where(
            torch.nn.functional.pad(sorted_edges_ids[1:] != sorted_edges_ids[:-1],
                                    (1,0), value=1))[0]
        nb_faces_per_edge = idx_first[1:] - idx_first[:-1]
        # compute sub_idx (2nd axis indices to store the faces)
        offsets = torch.zeros(sorted_edges_ids.shape[0], device=device, dtype=torch.long)
        offsets[idx_first[1:]] = nb_faces_per_edge
        sub_idx = (torch.arange(sorted_edges_ids.shape[0], device=device, dtype=torch.long) -
                   torch.cumsum(offsets, dim=0))
        # TODO(cfujitsang): potential way to compute sub_idx differently
        #                   to test with bigger model
        #sub_idx = torch.ones(sorted_edges_ids.shape[0], device=device, dtype=torch.long)
        #sub_idx[0] = 0
        #sub_idx[idx_first[1:]] = 1 - nb_faces_per_edge
        #sub_idx = torch.cumsum(sub_idx, dim=0)
        nb_faces_per_edge = torch.cat([nb_faces_per_edge,
                                       sorted_edges_ids.shape[0] - idx_first[-1:]],
                                      dim=0)
        max_sub_idx = torch.max(nb_faces_per_edge)
        ef = torch.full((nb_edges, max_sub_idx), device=device, dtype=torch.long, fill_value=-1)
        ef[sorted_edges_ids, sub_idx] = sorted_faces_ids
        # FACE2FACES
        nb_faces_per_face = torch.stack([nb_faces_per_edge[edges_ids[i*nb_faces:(i+1)*nb_faces]]
                                         for i in range(facesize)], dim=1).sum(dim=1) - facesize
        ff = torch.cat([ef[edges_ids[i*nb_faces:(i+1)*nb_faces]] for i in range(facesize)], dim=1)
        # remove self occurences
        ff[ff == torch.arange(nb_faces, device=device, dtype=torch.long).view(-1,1)] = -1
        ff = torch.sort(ff, dim=-1, descending=True)[0]
        to_del = (ff[:,1:] == ff[:,:-1]) & (ff[:,1:] != -1)
        ff[:,1:][to_del] = -1
        nb_faces_per_face = nb_faces_per_face - torch.sum(to_del, dim=1)
        max_sub_idx = torch.max(nb_faces_per_face)
        ff = torch.sort(ff, dim=-1, descending=True)[0][:,:max_sub_idx]

        # VERTEX2VERTICES and VERTEX2EDGES
        npy_edges = edges.cpu().numpy()
        edge2key = {tuple(npy_edges[i]): i for i in range(nb_edges)}
        #_edges and double_edges 2nd axis correspond to the triplet:
        # [left vertex, right vertex, edge key]
        _edges = torch.cat([edges, torch.arange(nb_edges, device=device).view(-1, 1)],
                           dim=1)
        double_edges = torch.cat([_edges, _edges[:,[1,0,2]]], dim=0)
        double_edges = torch.unique(double_edges, sorted=True, dim=0)
        # TODO(cfujitsang): potential improvment, to test with bigger model:
        #double_edges0, order_double_edges = torch.sort(double_edges[0])
        nb_double_edges = double_edges.shape[0]
        # indices of first occurences of each key
        idx_first = torch.where(
            torch.nn.functional.pad(double_edges[1:,0] != double_edges[:-1,0],
                                    (1,0), value=1))[0]
        nb_edges_per_vertex = idx_first[1:] - idx_first[:-1]
        # compute sub_idx (2nd axis indices to store the edges)
        offsets = torch.zeros(nb_double_edges, device=device, dtype=torch.long)
        offsets[idx_first[1:]] = nb_edges_per_vertex
        sub_idx = (torch.arange(nb_double_edges, device=device, dtype=torch.long) -
                   torch.cumsum(offsets, dim=0))
        nb_edges_per_vertex = torch.cat([nb_edges_per_vertex,
                                         nb_double_edges - idx_first[-1:]], dim=0)
        max_sub_idx = torch.max(nb_edges_per_vertex)
        vv = torch.full((nb_vertices, max_sub_idx), device=device, dtype=torch.long, fill_value=-1)
        vv[double_edges[:,0], sub_idx] = double_edges[:,1]
        ve = torch.full((nb_vertices, max_sub_idx), device=device, dtype=torch.long, fill_value=-1)
        ve[double_edges[:,0], sub_idx] = double_edges[:,2]

        # VERTEX2FACES
        vertex_ordered, order_vertex = torch.sort(faces.view(-1))
        face_ids_in_vertex_order = order_vertex / facesize
        # indices of first occurences of each id
        idx_first = torch.where(
            torch.nn.functional.pad(vertex_ordered[1:] != vertex_ordered[:-1], (1,0), value=1))[0]
        nb_faces_per_vertex = idx_first[1:] - idx_first[:-1]
        # compute sub_idx (2nd axis indices to store the faces)
        offsets = torch.zeros(vertex_ordered.shape[0], device=device, dtype=torch.long)
        offsets[idx_first[1:]] = nb_faces_per_vertex
        sub_idx = (torch.arange(vertex_ordered.shape[0], device=device, dtype=torch.long) -
                   torch.cumsum(offsets, dim=0))
        # TODO(cfujitsang): it seems that nb_faces_per_vertex == nb_edges_per_vertex ?
        nb_faces_per_vertex = torch.cat([nb_faces_per_vertex,
                                         vertex_ordered.shape[0] - idx_first[-1:]], dim=0)
        max_sub_idx = torch.max(nb_faces_per_vertex)
        vf = torch.full((nb_vertices, max_sub_idx), device=device, dtype=torch.long, fill_value=-1)
        vf[vertex_ordered, sub_idx] = face_ids_in_vertex_order

        return edge2key, edges, vv, nb_edges_per_vertex, ve, nb_edges_per_vertex, vf, \
            nb_faces_per_vertex, ff, nb_faces_per_face, ee, nb_edges_per_edge, ef, nb_faces_per_edge


    @staticmethod
    def old_compute_adjacency_info(vertices: torch.Tensor, faces: torch.Tensor):
        """Build data structures to help speed up connectivity queries. Assumes
        a homogeneous mesh, i.e., each face has the same number of vertices.

        """
        
        device = vertices.device

        facesize = faces.shape[1]

        # Dictionary to hash each edge
        edge2key = dict()
        # List of edges
        edges = []
        # List of neighboring vertices to each vertex
        vertex_vertex_nbd = [set() for _ in vertices]
        # List of neighboring edges to each vertex
        vertex_edge_nbd = [set() for _ in vertices]
        # List of neighboring faces to each vertex
        vertex_face_nbd = [set() for _ in vertices]
        # List of neighboring edges to each edge
        edge_edge_nbd = []
        # List of neighboring faces to each edge
        edge_face_nbd = []
        # List of neighboring faces to each face
        face_face_nbd = [set() for _ in faces]
        # Counter for edges
        num_edges = 0

        for fid, f in enumerate(faces):

            # Get a list of edges in the current face
            face_edges = Mesh.get_edges_from_face(f)
            # Run a pass through the edges, and add any new
            # edges found, to the list of edges. Also, initialize
            # corresponding neighborhood info.
            for idx, edge in enumerate(face_edges):
                if edge not in edge2key:
                    edge2key[edge] = num_edges
                    edges.append(list(edge))
                    edge_edge_nbd.append([])
                    edge_face_nbd.append([fid])
                    vertex_edge_nbd[edge[0]].add(num_edges)
                    vertex_edge_nbd[edge[1]].add(num_edges)
                    num_edges += 1
            # Now, run another pass through the edges, this time to
            # compute adjacency info.
            for idx, edge in enumerate(face_edges):
                k = edge2key[edge]
                for j in range(1, facesize):
                    q = edge2key[face_edges[(idx + j) % facesize]]
                    common_vtx, first_nbr, second_nbr = Mesh.get_common_vertex(
                        edges[k], edges[q])
                    edge_edge_nbd[k].append(q)
                    if common_vtx:
                        vertex_vertex_nbd[common_vtx].add(first_nbr)
                        vertex_vertex_nbd[common_vtx].add(second_nbr)
                        vertex_vertex_nbd[first_nbr].add(common_vtx)
                        vertex_vertex_nbd[second_nbr].add(common_vtx)

                # q = edge2key[face_edges[(idx+1)%facesize]]
                # r = edge2key[face_edges[(idx+2)%facesize]]
                # s = edge2key[face_edges[(idx+3)%facesize]]
                # if Mesh.has_common_vertex(edges[k], edges[q]):
                #     edge_edge_nbd[k].append(q)
                # if Mesh.has_common_vertex(edges[k], edges[r]):
                #     edge_edge_nbd[k].append(r)
                # if Mesh.has_common_vertex(edges[k], edges[s]):
                #     edge_edge_nbd[k].append(s)
                if fid not in edge_face_nbd[k]:
                    edge_face_nbd[k].append(fid)
                vertex_edge_nbd[edge[0]].add(k)
                vertex_edge_nbd[edge[1]].add(k)
                vertex_face_nbd[edge[0]].add(fid)
                vertex_face_nbd[edge[1]].add(fid)
        # Compute face-face adjacency info
        for fid, f in enumerate(faces):
            face_edges = Mesh.get_edges_from_face(f)
            for idx, edge in enumerate(face_edges):
                k = edge2key[edge]
                for nbr in edge_face_nbd[k]:
                    if nbr == fid:
                        continue
                    face_face_nbd[fid].add(nbr)

        # Helper variables
        N = vertices.shape[0]
        M = len(edges)
        P = faces.shape[0]

        # Convert sets to lists in vertex_edge_nbd, vertex_face_nbd, and
        # face_face_nbd
        vertex_vertex_nbd = [torch.Tensor(list(l)).long().to(device)
                             for l in vertex_vertex_nbd]
        vertex_edge_nbd = [torch.Tensor(list(l)).long().to(device)
                           for l in vertex_edge_nbd]
        vertex_face_nbd = [torch.Tensor(list(l)).long().to(device)
                           for l in vertex_face_nbd]
        face_face_nbd = [torch.Tensor(list(l)).long().to(device)
                         for l in face_face_nbd]
        edge_edge_nbd = [torch.Tensor(l).long().to(device)
                         for l in edge_edge_nbd]
        edge_face_nbd = [torch.Tensor(l).long().to(device)
                         for l in edge_face_nbd]

        # Map vertex_vertex_nbd to a matrix
        vv_count = torch.Tensor([len(l) for l in vertex_vertex_nbd]).long()
        vv_max = max(vv_count)
        vv = -torch.ones((N, vv_max)).long().to(device)
        vv = Mesh.list_of_lists_to_matrix(vertex_vertex_nbd, vv_count, vv)

        # Map vertex_edge_nbd to a matrix
        ve_count = torch.Tensor([len(l) for l in vertex_edge_nbd]).long()
        ve_max = max(ve_count)
        ve = -torch.ones((N, ve_max)).long().to(device)
        ve = Mesh.list_of_lists_to_matrix(vertex_edge_nbd, ve_count, ve)

        # Map vertex_face_nbd to a matrix
        vf_count = torch.Tensor([len(l) for l in vertex_face_nbd]).long()
        vf_max = max(vf_count)
        vf = -torch.ones((N, vf_max)).long().to(device)
        vf = Mesh.list_of_lists_to_matrix(vertex_face_nbd, vf_count, vf)

        # Map edge_edge_nbd to a matrix
        ee_count = torch.Tensor([len(l) for l in edge_edge_nbd]).long()
        ee_max = max(ee_count)
        ee = -torch.ones((M, ee_max)).long().to(device)
        ee = Mesh.list_of_lists_to_matrix(edge_edge_nbd, ee_count, ee)

        # Map edge_face_nbd to a matrix
        ef_count = torch.Tensor([len(l) for l in edge_face_nbd]).long()
        ef_max = max(ef_count)
        ef = -torch.ones((M, ef_max)).long().to(device)
        ef = Mesh.list_of_lists_to_matrix(edge_face_nbd, ef_count, ef)

        # Map face_face_nbd to a matrix
        ff_count = torch.Tensor([len(l) for l in face_face_nbd]).long()
        ff_max = max(ff_count)
        ff = -torch.ones((P, ff_max)).long().to(device)
        ff = Mesh.list_of_lists_to_matrix(face_face_nbd, ff_count, ff)

        # Convert to numpy arrays
        edges = torch.Tensor(edges).long().to(device)

        return edge2key, edges, vv, vv_count, ve, ve_count, vf, vf_count, \
            ff, ff_count, ee, ee_count, ef, ef_count

    def laplacian_smoothing(self, iterations: int = 1):
        r""" Applies laplacian smoothing to the mesh.

            Args:
                iterations (int) : number of iterations to run the algorithm for.

            Example:
                >>> mesh = Mesh.from_obj('model.obj')
                >>> mesh.compute_laplacian().abs().mean()
                tensor(0.0010)
                >>> mesh.laplacian_smoothing(iterations=3)
                >>> mesh.compute_laplacian().abs().mean()
                tensor(9.9956e-05)
    """

        adj_sparse = self.compute_adjacency_matrix_sparse()

        neighbor_num = torch.sparse.sum(
            adj_sparse, dim=1).to_dense().view(-1, 1)

        for _ in range(iterations):
            neighbor_sum = torch.sparse.mm(adj_sparse, self.vertices)
            self.vertices = neighbor_sum / neighbor_num

    def compute_laplacian(self):
        r"""Calcualtes the laplcaian of the graph, meaning the average
                difference between a vertex and its neighbors.

            Returns:
                (FloatTensor) : laplacian of the mesh.

            Example:
                >>> mesh = Mesh.from_obj('model.obj')
                >>> lap = mesh.compute_laplacian()

        """

        adj_sparse = self.compute_adjacency_matrix_sparse()

        neighbor_sum = torch.sparse.mm(
            adj_sparse, self.vertices) - self.vertices
        neighbor_num = torch.sparse.sum(
            adj_sparse, dim=1).to_dense().view(-1, 1) - 1
        neighbor_num[neighbor_num == 0] = 1
        neighbor_num = (1. / neighbor_num).view(-1, 1)

        neighbor_sum = neighbor_sum * neighbor_num
        lap = self.vertices - neighbor_sum
        return lap

    def show(self):
        r""" Visuailizes the mesh.

            Example:
                >>> mesh = Mesh.from_obj('model.obj')
                >>> mesh.show()

        """

        kal.visualize.show_mesh(self)

    def save_tensors(self, filename: (str)):
        r"""Saves the tensor information of the mesh in a numpy .npz format.

        Args:
            filename: the file name to save the file under

        Example:
            >>> mesh = Mesh.from_obj('model.obj')
            >>> mesh.save_tensors()

        """
        np.savez(filename, vertices=self.vertices.data.cpu().numpy(),
                 faces=self.faces.data.cpu().numpy())

    @staticmethod
    def normalize_zerosafe(matrix: torch.Tensor):
        """Normalizes each row of a matrix in a 'division by zero'-safe way.

        Args:
            matrix (torch.tensor): Matrix where each row contains a vector
                to be normalized.

        """

        assert matrix.dim() == 2, 'Need matrix to contain exactly 2 dimensions'
        magnitude = torch.sqrt(torch.sum(torch.pow(matrix, 2), dim=1))
        valid_inds = magnitude > 0
        matrix[valid_inds] = torch.div(matrix[valid_inds], magnitude[
                                       valid_inds].unsqueeze(1))
        return matrix

    def sample(self, num_points):
        raise NotImplementedError

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

    def __getstate__(self):
        outputs = {'vertices': self.vertices,
                   'faces': self.faces}
        if self.uvs is not None:
            outputs['uvs'] = self.uvs
        if self.face_textures is not None:
            outputs['face_textures'] = self.face_textures
        if self.textures is not None:
            outputs['textures'] = self.textures
        return outputs

    def __setstate__(self, args):
        self.vertices = args['vertices']
        self.faces = args['faces']
        if 'uvs' in args:
            self.uvs = args['uvs']
        if 'face_textures' in args:
            self.face_textures = args['face_textures']
        if 'textures' in args:
            self.textures = args['textures']
