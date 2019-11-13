# MIT License

# Copyright (c) 2017 Hiroharu Kato
# Copyright (c) 2018 Nikos Kolotouros
# A PyTorch implementation of Neural 3D Mesh Renderer (https://github.com/hiroharu-kato/neural_renderer)

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

import torch
import torch.nn as nn

import neural_renderer as nr

class Mesh(object):
    '''
    A simple class for creating and manipulating trimesh objects
    '''
    def __init__(self, vertices, faces, textures=None, texture_size=4):
        '''
        vertices, faces and textures(if not None) are expected to be Tensor objects
        '''
        self.vertices = vertices
        self.faces = faces
        self.num_vertices = self.vertices.shape[0]
        self.num_faces = self.faces.shape[0]

        # create textures
        if textures is None:
            shape = (self.num_faces, texture_size, texture_size, texture_size, 3)
            self.textures = nn.Parameter(0.05*torch.randn(*shape))
            self.texture_size = texture_size
        else:
            self.texture_size = textures.shape[0]

    @classmethod
    def fromobj(cls, filename_obj, normalization=True, load_texture=False, texture_size=4):
        '''
        Create a Mesh object from a .obj file
        '''
        if load_texture:
            vertices, faces, textures = nr.load_obj(filename_obj,
                                                    normalization=normalization,
                                                    texture_size=texture_size,
                                                    load_texture=True)
        else:
            vertices, faces = nr.load_obj(filename_obj,
                                          normalization=normalization,
                                          texture_size=texture_size,
                                          load_texture=False)
            textures = None
        return cls(vertices, faces, textures, texture_size)
