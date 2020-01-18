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


################################################
def texinterpolation(imtexcoord_bxhxwx2, texture_bx3xthxtw, filtering='nearest'):
    '''
    Note that opengl tex coord is different from pytorch coord
    ogl coord ranges from 0 to 1, y axis is from bottom to top and it supports circular mode(-0.1 is the same as 0.9)
    pytorch coord ranges from -1 to 1, y axis is from top to bottom and does not support circular 

    filtering is the same as the mode parameter for torch.nn.functional.grid_sample.
    '''

    # convert coord mode from ogl to pytorch
    imtexcoord_bxhxwx2 = torch.remainder(imtexcoord_bxhxwx2, 1.0)
    imtexcoord_bxhxwx2 = imtexcoord_bxhxwx2 * 2 - 1  # [0, 1] to [-1, 1]
    imtexcoord_bxhxwx2[:, :, :, 1] = -1.0 * imtexcoord_bxhxwx2[:, :, :, 1]  # reverse y

    # sample
    texcolor = torch.nn.functional.grid_sample(texture_bx3xthxtw,
                                               imtexcoord_bxhxwx2,
                                               mode=filtering)
    texcolor = texcolor.permute(0, 2, 3, 1)

    return texcolor
