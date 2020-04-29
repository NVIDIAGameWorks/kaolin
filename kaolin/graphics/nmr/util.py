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

from __future__ import division

import numpy as np
import torch.nn.functional as F
import math
import torch


def get_points_from_angles(distance, elevation, azimuth, degrees=True):
    if isinstance(distance, float) or isinstance(distance, int):
        if degrees:
            elevation = math.radians(elevation)
            azimuth = math.radians(azimuth)
        return (
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
            -distance * math.cos(elevation) * math.cos(azimuth))
    else:
        if degrees:
            elevation = math.pi / 180. * elevation
            azimuth = math.pi / 180. * azimuth
    #
        return torch.stack([
            distance * torch.cos(elevation) * torch.sin(azimuth),
            distance * torch.sin(elevation),
            -distance * torch.cos(elevation) * torch.cos(azimuth)
        ]).transpose(1, 0)


def lighting(faces, textures, intensity_ambient=0.5, intensity_directional=0.5,
             color_ambient=(1, 1, 1), color_directional=(1, 1, 1), direction=(0, 1, 0)):

    bs, nf = faces.shape[:2]
    device = faces.device

    # arguments
    # make sure to convert all inputs to float tensors
    if isinstance(color_ambient, tuple) or isinstance(color_ambient, list):
        color_ambient = torch.tensor(
            color_ambient, dtype=torch.float32, device=device)
    elif isinstance(color_ambient, np.ndarray):
        color_ambient = torch.from_numpy(color_ambient).float().to(device)
    if isinstance(color_directional, tuple) or isinstance(color_directional, list):
        color_directional = torch.tensor(
            color_directional, dtype=torch.float32, device=device)
    elif isinstance(color_directional, np.ndarray):
        color_directional = torch.from_numpy(
            color_directional).float().to(device)
    if isinstance(direction, tuple) or isinstance(direction, list):
        direction = torch.tensor(direction, dtype=torch.float32, device=device)
    elif isinstance(direction, np.ndarray):
        direction = torch.from_numpy(direction).float().to(device)
    if color_ambient.ndimension() == 1:
        color_ambient = color_ambient[None, :]
    if color_directional.ndimension() == 1:
        color_directional = color_directional[None, :]
    if direction.ndimension() == 1:
        direction = direction[None, :]

    # create light
    light = torch.zeros(bs, nf, 3, dtype=torch.float32).to(device)

    # ambient light
    if intensity_ambient != 0:
        light += intensity_ambient * color_ambient[:, None, :]

    # directional light
    if intensity_directional != 0:
        faces = faces.reshape((bs * nf, 3, 3))
        v10 = faces[:, 0] - faces[:, 1]
        v12 = faces[:, 2] - faces[:, 1]
        # pytorch normalize divides by max(norm, eps) instead of (norm+eps) in chainer
        normals = F.normalize(torch.cross(v10, v12), eps=1e-5)
        normals = normals.reshape((bs, nf, 3))

        if direction.ndimension() == 2:
            direction = direction[:, None, :]
        cos = F.relu(torch.sum(normals * direction, dim=2))
        # may have to verify that the next line is correct
        light += intensity_directional * \
            (color_directional[:, None, :] * cos[:, :, None])

    # apply
    light = light[:, :, None, None, None, :]
    textures *= light
    return textures


def look(vertices, eye, direction=[0, 1, 0], up=None):
    """
    "Look" transformation of vertices.
    """
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')

    device = vertices.device

    if isinstance(direction, list) or isinstance(direction, tuple):
        direction = torch.tensor(direction, dtype=torch.float32, device=device)
    elif isinstance(direction, np.ndarray):
        direction = torch.from_numpy(direction).to(device)
    elif torch.is_tensor(direction):
        direction = direction.to(device)

    if isinstance(eye, list) or isinstance(eye, tuple):
        eye = torch.tensor(eye, dtype=torch.float32, device=device)
    elif isinstance(eye, np.ndarray):
        eye = torch.from_numpy(eye).to(device)
    elif torch.is_tensor(eye):
        eye = eye.to(device)

    if up is None:
        up = torch.cuda.FloatTensor([0, 1, 0])
    if eye.ndimension() == 1:
        eye = eye[None, :]
    if direction.ndimension() == 1:
        direction = direction[None, :]
    if up.ndimension() == 1:
        up = up[None, :]

    # create new axes
    z_axis = F.normalize(direction, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis), eps=1e-5)

    # create rotation matrix: [bs, 3, 3]
    r = torch.cat((x_axis[:, None, :], y_axis[:, None, :],
                   z_axis[:, None, :]), dim=1)

    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != eye.shape:
        eye = eye[:, None, :]
    vertices = vertices - eye
    vertices = torch.matmul(vertices, r.transpose(1, 2))

    return vertices


def look_at(vertices, eye, at=[0, 0, 0], up=[0, 1, 0]):
    """
    "Look at" transformation of vertices.
    """
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')

    device = vertices.device

    # if list or tuple convert to numpy array
    if isinstance(at, list) or isinstance(at, tuple):
        at = torch.tensor(at, dtype=torch.float32, device=device)
    # if numpy array convert to tensor
    elif isinstance(at, np.ndarray):
        at = torch.from_numpy(at).to(device)
    elif torch.is_tensor(at):
        at.to(device)

    if isinstance(up, list) or isinstance(up, tuple):
        up = torch.tensor(up, dtype=torch.float32, device=device)
    elif isinstance(up, np.ndarray):
        up = torch.from_numpy(up).to(device)
    elif torch.is_tensor(up):
        up.to(device)

    if isinstance(eye, list) or isinstance(eye, tuple):
        eye = torch.tensor(eye, dtype=torch.float32, device=device)
    elif isinstance(eye, np.ndarray):
        eye = torch.from_numpy(eye).to(device)
    elif torch.is_tensor(eye):
        eye = eye.to(device)

    batch_size = vertices.shape[0]
    if eye.ndimension() == 1:
        eye = eye[None, :].repeat(batch_size, 1)
    if at.ndimension() == 1:
        at = at[None, :].repeat(batch_size, 1)
    if up.ndimension() == 1:
        up = up[None, :].repeat(batch_size, 1)

    # create new axes
    # eps is chosen as 0.5 to match the chainer version
    z_axis = F.normalize(at - eye, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis), eps=1e-5)

    # create rotation matrix: [bs, 3, 3]
    r = torch.cat((x_axis[:, None, :], y_axis[:, None, :],
                   z_axis[:, None, :]), dim=1)

    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != eye.shape:
        eye = eye[:, None, :]
    vertices = vertices - eye
    vertices = torch.matmul(vertices, r.transpose(1, 2))

    return vertices


def perspective(vertices, angle=30.):
    '''
    Compute perspective distortion from a given angle
    '''
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')
    device = vertices.device
    angle = torch.tensor(angle / 180 * math.pi,
                         dtype=torch.float32, device=device)
    angle = angle[None]
    width = torch.tan(angle)
    width = width[:, None]
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] / z / width
    y = vertices[:, :, 1] / z / width
    vertices = torch.stack((x, y, z), dim=2)
    return vertices


def projection(vertices, K, R, t, dist_coeffs, orig_size, eps=1e-9):
    '''
    Calculate projective transformation of vertices given a projection matrix
    Input parameters:
    K: batch_size * 3 * 3 intrinsic camera matrix
    R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
    dist_coeffs: vector of distortion coefficients
    orig_size: original size of image captured by the camera
    Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
    pixels and z is the depth
    '''

    # instead of P*x we compute x'*P'
    vertices = torch.matmul(vertices, R.transpose(2, 1)) + t
    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
    x_ = x / (z + eps)
    y_ = y / (z + eps)

    # Get distortion coefficients from vector
    k1 = dist_coeffs[:, None, 0]
    k2 = dist_coeffs[:, None, 1]
    p1 = dist_coeffs[:, None, 2]
    p2 = dist_coeffs[:, None, 3]
    k3 = dist_coeffs[:, None, 4]

    # we use x_ for x' and x__ for x'' etc.
    r = torch.sqrt(x_ ** 2 + y_ ** 2)
    x__ = x_ * (1 + k1 * (r**2) + k2 * (r**4) + k3 * (r**6)) + \
        2 * p1 * x_ * y_ + p2 * (r**2 + 2 * x_**2)
    y__ = y_ * (1 + k1 * (r**2) + k2 * (r**4) + k3 * (r**6)) + \
        p1 * (r**2 + 2 * y_**2) + 2 * p2 * x_ * y_
    vertices = torch.stack([x__, y__, torch.ones_like(z)], dim=-1)
    vertices = torch.matmul(vertices, K.transpose(1, 2))
    u, v = vertices[:, :, 0], vertices[:, :, 1]
    v = orig_size - v
    # map u,v from [0, img_size] to [-1, 1] to use by the renderer
    u = 2 * (u - orig_size / 2.) / orig_size
    v = 2 * (v - orig_size / 2.) / orig_size
    vertices = torch.stack([u, v, z], dim=-1)
    return vertices


def vertices_to_faces(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3)
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    faces = faces.long()

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + \
        (torch.arange(bs, dtype=torch.long, device=device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    return vertices[faces]
