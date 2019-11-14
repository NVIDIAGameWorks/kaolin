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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import soft_renderer.functional as srf


class Mesh(object):
    '''
    A simple class for creating and manipulating trimesh objects
    '''
    def __init__(self, vertices, faces, textures=None, texture_res=1, texture_type='surface'):
        '''
        vertices, faces and textures(if not None) are expected to be Tensor objects
        '''
        self._vertices = vertices
        self._faces = faces

        if isinstance(self._vertices, np.ndarray):
            self._vertices = torch.from_numpy(self._vertices).float().cuda()
        if isinstance(self._faces, np.ndarray):
            self._faces = torch.from_numpy(self._faces).int().cuda()
        if self._vertices.ndimension() == 2:
            self._vertices = self._vertices[None, :, :]
        if self._faces.ndimension() == 2:
            self._faces = self._faces[None, :, :]

        self.device = self._vertices.device
        self.texture_type = texture_type

        self.batch_size = self._vertices.shape[0]
        self.num_vertices = self._vertices.shape[1]
        self.num_faces = self._faces.shape[1]

        self._face_vertices = None
        self._face_vertices_update = True
        self._surface_normals = None
        self._surface_normals_update = True
        self._vertex_normals = None
        self._vertex_normals_update = True

        self._fill_back = False

        # create textures
        if textures is None:
            if texture_type == 'surface':
                self._textures = torch.ones(self.batch_size, self.num_faces, texture_res**2, 3, 
                                            dtype=torch.float32).to(self.device)
                self.texture_res = texture_res
            elif texture_type == 'vertex':
                self._textures = torch.ones(self.batch_size, self.num_vertices, 3, 
                                            dtype=torch.float32).to(self.device)
                self.texture_res = 1
        else:
            if isinstance(textures, np.ndarray):
                textures = torch.from_numpy(textures).float().cuda()
            if textures.ndimension() == 3 and texture_type == 'surface':
                textures = textures[None, :, :, :]
            if textures.ndimension() == 2 and texture_type == 'vertex':
                textures = textures[None, :, :]
            self._textures = textures
            self.texture_res = int(np.sqrt(self._textures.shape[2]))

        self._origin_vertices = self._vertices
        self._origin_faces = self._faces
        self._origin_textures = self._textures

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, faces):
        # need check tensor
        self._faces = faces
        self.num_faces = self._faces.shape[1]
        self._face_vertices_update = True
        self._surface_normals_update = True
        self._vertex_normals_update = True

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        # need check tensor
        self._vertices = vertices
        self.num_vertices = self._vertices.shape[1]
        self._face_vertices_update = True
        self._surface_normals_update = True
        self._vertex_normals_update = True

    @property
    def textures(self):
        return self._textures

    @textures.setter
    def textures(self, textures):
        # need check tensor
        self._textures = textures

    @property
    def face_vertices(self):
        if self._face_vertices_update:
            self._face_vertices = srf.face_vertices(self.vertices, self.faces)
            self._face_vertices_update = False
        return self._face_vertices

    @property
    def surface_normals(self):
        if self._surface_normals_update:
            v10 = self.face_vertices[:, :, 0] - self.face_vertices[:, :, 1]
            v12 = self.face_vertices[:, :, 2] - self.face_vertices[:, :, 1]
            self._surface_normals = F.normalize(torch.cross(v12, v10), p=2, dim=2, eps=1e-6)
            self._surface_normals_update = False
        return self._surface_normals

    @property
    def vertex_normals(self):
        if self._vertex_normals_update:
            self._vertex_normals = srf.vertex_normals(self.vertices, self.faces)
            self._vertex_normals_update = False
        return self._vertex_normals

    @property
    def face_textures(self):
        if self.texture_type in ['surface']:
            return self.textures
        elif self.texture_type in ['vertex']:
            return srf.face_vertices(self.textures, self.faces)
        else:
            raise ValueError('texture type not applicable')

    def fill_back_(self):
        if not self._fill_back:
            self.faces = torch.cat((self.faces, self.faces[:, :, [2, 1, 0]]), dim=1)
            self.textures = torch.cat((self.textures, self.textures), dim=1)
            self._fill_back = True

    def reset_(self):
        self.vertices = self._origin_vertices
        self.faces = self._origin_faces
        self.textures = self._origin_textures
        self._fill_back = False
    
    @classmethod
    def from_obj(cls, filename_obj, normalization=False, load_texture=False, texture_res=1, texture_type='surface'):
        '''
        Create a Mesh object from a .obj file
        '''
        if load_texture:
            vertices, faces, textures = srf.load_obj(filename_obj,
                                                     normalization=normalization,
                                                     texture_res=texture_res,
                                                     load_texture=True,
                                                     texture_type=texture_type)
        else:
            vertices, faces = srf.load_obj(filename_obj,
                                           normalization=normalization,
                                           texture_res=texture_res,
                                           load_texture=False)
            textures = None
        return cls(vertices, faces, textures, texture_res, texture_type)

    def save_obj(self, filename_obj, save_texture=False, texture_res_out=16):
        if self.batch_size != 1:
            raise ValueError('Could not save when batch size >= 1')
        if save_texture:
            srf.save_obj(filename_obj, self.vertices[0], self.faces[0], 
                         textures=self.textures[0],
                         texture_res=texture_res_out, texture_type=self.texture_type)
        else:
            srf.save_obj(filename_obj, self.vertices[0], self.faces[0], textures=None)

    def voxelize(self, voxel_size=32):
        face_vertices_norm = self.face_vertices * voxel_size / (voxel_size - 1) + 0.5
        return srf.voxelization(face_vertices_norm, voxel_size, False)