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

from ..rasterizer import linear_rasterizer
from ..utils import datanormalize
from .fragment_shaders.frag_shtex import fragmentshader
from .vertex_shaders.perpsective import perspective_projection
import torch
import torch.nn as nn


##################################################################
class SHRender(nn.Module):

    def __init__(self, height, width):
        super(SHRender, self).__init__()

        self.height = height
        self.width = width

        # render with point normal or not
        self.smooth = False

    def set_smooth(self, pfmtx):
        self.smooth = True
        self.pfmtx = pfmtx

    def forward(self,
                points,
                cameras,
                uv_bxpx2,
                texture_bx3xthxtw,
                lightparam,
                ft_fx3=None):

        assert lightparam is not None, 'When using the Spherical Harmonics model, light parameters must be passed'

        ##############################################################
        # first, MVP projection in vertexshader
        points_bxpx3, faces_fx3 = points

        # use faces_fx3 as ft_fx3 if not given
        if ft_fx3 is None:
            ft_fx3 = faces_fx3

        # camera_rot_bx3x3, camera_pos_bx3, camera_proj_3x1 = cameras

        points3d_bxfx9, points2d_bxfx6, normal_bxfx3 = \
            perspective_projection(points_bxpx3, faces_fx3, cameras)

        ################################################################
        # normal

        # decide which faces are front and which faces are back
        normalz_bxfx1 = normal_bxfx3[:, :, 2:3]
        # normalz_bxfx1 = torch.abs(normalz_bxfx1)

        # normalize normal
        normal1_bxfx3 = datanormalize(normal_bxfx3, axis=2)

        ####################################################
        # smooth or not
        if self.smooth:
            normal_bxpx3 = torch.matmul(self.pfmtx, normal_bxfx3)
            n0 = normal_bxpx3[:, faces_fx3[:, 0], :]
            n1 = normal_bxpx3[:, faces_fx3[:, 1], :]
            n2 = normal_bxpx3[:, faces_fx3[:, 2], :]
            normal_bxfx9 = torch.cat((n0, n1, n2), dim=2)
        else:
            normal_bxfx9 = normal_bxfx3.repeat(1, 1, 3)

        #########################################################
        # second, rasterization
        fnum = normal1_bxfx3.shape[1]
        bnum = normal1_bxfx3.shape[0]

        c0 = uv_bxpx2[:, ft_fx3[:, 0], :]
        c1 = uv_bxpx2[:, ft_fx3[:, 1], :]
        c2 = uv_bxpx2[:, ft_fx3[:, 2], :]
        mask = torch.ones_like(c0[:, :, :1])
        uv_bxfx3x3 = torch.cat(
            (c0, mask, c1, mask, c2, mask), dim=2).view(bnum, fnum, 3, -1)

        # normal
        normal_bxfx3x3 = normal_bxfx9.view(bnum, fnum, 3, -1)
        feat = torch.cat((normal_bxfx3x3, uv_bxfx3x3), dim=3)
        feat = feat.view(bnum, fnum, -1)

        imfeat, improb_bxhxwx1 = linear_rasterizer(self.height, self.width,
                                                   points3d_bxfx9, points2d_bxfx6, normalz_bxfx1, feat)
        imnormal_bxhxwx3 = imfeat[:, :, :, :3]
        imtexcoords = imfeat[:, :, :, 3:5]
        hardmask = imfeat[:, :, :, 5:]

        ####################################################
        # fragrement shader
        # parallel light
        imnormal1_bxhxwx3 = datanormalize(imnormal_bxhxwx3, axis=3)
        imrender = fragmentshader(
            imnormal1_bxhxwx3, lightparam, imtexcoords, texture_bx3xthxtw, hardmask)

        return imrender, improb_bxhxwx1, normal1_bxfx3
