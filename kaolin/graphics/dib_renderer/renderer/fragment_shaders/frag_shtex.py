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


# 33
def fragmentshader(imnormal1_bxhxwx3,
                   lightparam_bx9,
                   imtexcoord_bxhxwx2,
                   texture_bx3xthxtw,
                   improb_bxhxwx1):

    # light effect
    x = imnormal1_bxhxwx3[:, :, :, 0:1]
    y = imnormal1_bxhxwx3[:, :, :, 1:2]
    z = imnormal1_bxhxwx3[:, :, :, 2:3]

    # spherical harmonic parameters
    band0 = 0.2820948 * torch.ones_like(x)
    band10 = -0.3257350 * y
    band11 = 0.3257350 * z
    band12 = -0.3257350 * x
    band20 = 0.2731371 * (x * y)
    band21 = -0.2731371 * (y * z)
    band22 = 0.1365686 * (z * z) - 0.0788479
    band23 = -0.1931371 * (x * z)
    band24 = 0.1365686 * (x * x - y * y)

    bands = torch.cat((band0,
                       band10, band11, band12,
                       band20, band21, band22, band23, band24), dim=3)
    coef = torch.sum(bands * lightparam_bx9.view(-1, 1, 1, 9), dim=3, keepdim=True)

    # tex color
    texcolor_bxhxwx3 = texinterpolation(imtexcoord_bxhxwx2, texture_bx3xthxtw)

    # merge
    color = coef * texcolor_bxhxwx3 * improb_bxhxwx1

    return torch.clamp(color, 0, 1)
