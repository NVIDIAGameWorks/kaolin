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
import torch.nn

from graphics.fshader.interpolation import texinterpolation


#####################################################
def fragmentshader(imnormal1_bxhxwx3, \
                   lightdirect1_bx3, \
                   eyedirect1_bxhxwx3, \
                   material_bx3x3, shininess_bx1, \
                   imtexcoord_bxhxwx2, texture_bx3xthxtw, \
                   improb_bxhxwx1, \
                   ):
    # parallel light
    lightdirect1_bx1x1x3 = lightdirect1_bx3.view(-1, 1, 1, 3)

    # lambertian
    cosTheta_bxhxwx1 = torch.sum(imnormal1_bxhxwx3 * lightdirect1_bx1x1x3, dim=3, keepdim=True)
    cosTheta_bxhxwx1 = torch.clamp(cosTheta_bxhxwx1, 0, 1)
    
    # specular
    reflect = -lightdirect1_bx1x1x3 + 2 * cosTheta_bxhxwx1 * imnormal1_bxhxwx3
    cosAlpha_bxhxwx1 = torch.sum(reflect * eyedirect1_bxhxwx3, dim=3, keepdim=True)
    cosAlpha_bxhxwx1 = torch.clamp(cosAlpha_bxhxwx1, 1e-5, 1)  # should not be 0 since nan error
    cosAlpha_bxhxwx1 = torch.pow(cosAlpha_bxhxwx1, shininess_bx1.view(-1, 1, 1, 1))  # shininess should be large than 0
    
    # simplified model
    # light color is [1, 1, 1]
    MatAmbColor_bx1x1x3 = material_bx3x3[:, 0:1, :].view(-1, 1, 1, 3);
    MatDifColor_bxhxwx3 = material_bx3x3[:, 1:2, :].view(-1, 1, 1, 3) * cosTheta_bxhxwx1
    MatSpeColor_bxhxwx3 = material_bx3x3[:, 2:3, :].view(-1, 1, 1, 3) * cosAlpha_bxhxwx1
    
    # tex color
    texcolor_bxhxwx3 = texinterpolation(imtexcoord_bxhxwx2, texture_bx3xthxtw)
    
    # ambient and diffuse rely on object color while specular doesn't
    color = (MatAmbColor_bx1x1x3 + MatDifColor_bxhxwx3) * texcolor_bxhxwx3 + MatSpeColor_bxhxwx3
    color = color * improb_bxhxwx1
    
    return torch.clamp(color, 0, 1)

