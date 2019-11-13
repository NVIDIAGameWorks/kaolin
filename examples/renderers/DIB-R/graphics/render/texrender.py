# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

from graphics.vshader.perpsective import PersepctiveProjection
from graphics.rasterize.rasterize import TriRender2D
from graphics.fshader.frag_tex import fragmentshader
from graphics.utils.utils import datanormalize


##################################################################
class TexRender(nn.Module):
    
    def __init__(self, height, width):
        super(TexRender, self).__init__()
        
        self.height = height
        self.width = width
    
    def forward(self, points, cameras, colors):
        
        ##############################################################
        # first, MVP projection in vertexshader
        points_bxpx3, faces_fx3 = points
        
        # camera_rot_bx3x3, camera_pos_bx3, camera_proj_3x1 = cameras
        
        points3d_bxfx9, points2d_bxfx6, normal_bxfx3 = \
        PersepctiveProjection(points_bxpx3, faces_fx3, cameras)
        
        ################################################################
        # normal
        
        # decide which faces are front and which faces are back
        normalz_bxfx1 = normal_bxfx3[:, :, 2:3]
        # normalz_bxfx1 = torch.abs(normalz_bxfx1)
        
        # normalize normal
        normal1_bxfx3 = datanormalize(normal_bxfx3, axis=2)
        
        ############################################################
        # second, rasterization
        uv_bxpx2, ft_fx3, texture_bx3xthxtw = colors
        c0 = uv_bxpx2[:, ft_fx3[:, 0], :]
        c1 = uv_bxpx2[:, ft_fx3[:, 1], :]
        c2 = uv_bxpx2[:, ft_fx3[:, 2], :]
        mask = torch.ones_like(c0[:, :, :1])
        uv_bxfx9 = torch.cat((c0, mask, c1, mask, c2, mask), dim=2)
  
        imfeat, improb_bxhxwx1 = TriRender2D(self.height, self.width)(points3d_bxfx9, points2d_bxfx6, normalz_bxfx1, uv_bxfx9)
    
        imtexcoords = imfeat[:, :, :, :2]
        hardmask = imfeat[:, :, :, 2:3]
        
        # fragrement shader
        imrender = fragmentshader(imtexcoords, texture_bx3xthxtw, hardmask)
        
        return imrender, improb_bxhxwx1, normal1_bxfx3

