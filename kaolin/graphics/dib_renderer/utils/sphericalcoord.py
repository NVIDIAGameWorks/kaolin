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

import numpy as np


##################################################################
# symmetric over z axis
def get_spherical_coords_z(X):
    # X is N x 3
    rad = np.linalg.norm(X, axis=1)
    # Inclination
    theta = np.arccos(X[:, 2] / rad)
    # Azimuth
    phi = np.arctan2(X[:, 1], X[:, 0])

    # Normalize both to be between [-1, 1]
    vv = (theta / np.pi) * 2 - 1
    uu = ((phi + np.pi) / (2 * np.pi)) * 2 - 1
    # Return N x 2
    return np.stack([uu, vv], 1)


# symmetric over x axis
def get_spherical_coords_x(X):
    # X is N x 3
    rad = np.linalg.norm(X, axis=1)
    # Inclination
    # y == 1
    # cos = 0
    # y == -1
    # cos = pi
    theta = np.arccos(X[:, 0] / rad)
    # Azimuth
    phi = np.arctan2(X[:, 2], X[:, 1])

    # Normalize both to be between [-1, 1]
    uu = (theta / np.pi) * 2 - 1
    vv = ((phi + np.pi) / (2 * np.pi)) * 2 - 1
    # Return N x 2
    return np.stack([uu, vv], 1)


# symmetric spherical projection
def get_symmetric_spherical_tex_coords(vertex_pos,
                                       symmetry_axis=1,
                                       up_axis=2,
                                       front_axis=0):
    # vertex_pos is N x 3
    length = np.linalg.norm(vertex_pos, axis=1)
    # Inclination
    theta = np.arccos(vertex_pos[:, front_axis] / length)
    # Azimuth
    phi = np.abs(np.arctan2(vertex_pos[:, symmetry_axis],
                            vertex_pos[:, up_axis]))

    # Normalize both to be between [-1, 1]
    uu = (theta / np.pi) * 2 - 1
    # vv = ((phi + np.pi) / (2 * np.pi)) * 2 - 1
    vv = (phi / np.pi) * 2 - 1
    # Return N x 2
    return np.stack([uu, vv], 1)


#########################################################################
if __name__ == '__main__':

    from utils.utils_mesh import loadobj, savemeshtes
    import cv2

    p, f = loadobj('2.obj')
    uv = get_spherical_coords_x(p)
    uv[:, 0] = -uv[:, 0]

    uv[:, 1] = -uv[:, 1]
    uv = (uv + 1) / 2
    savemeshtes(p, uv, f, './2_x.obj')

    tex = np.zeros(shape=(256, 512, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 200)
    fontScale = 5
    fontColor = (0, 255, 255)
    lineType = 2

    cv2.putText(tex, 'Hello World!',
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    cv2.imshow('', tex)
    cv2.waitKey()
    cv2.imwrite('2_x.png', np.transpose(tex, [1, 0, 2]))
