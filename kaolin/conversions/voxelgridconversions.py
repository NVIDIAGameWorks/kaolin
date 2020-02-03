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
# Multi-View Silhouette and Depth Decomposition for High Resolution 3D Object Representation
#
# Copyright (c) 2019 Edward Smith
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

from typing import Optional, Union, List

import torch
import torch.nn.functional as F
import numpy as np
import kaolin as kal
import trimesh
from scipy import ndimage

# from kaolin.transforms import voxelfunc
from kaolin.rep import VoxelGrid
from kaolin import helpers


def downsample(voxel: Union[torch.Tensor, VoxelGrid], scale: List[int],
               inplace: Optional[bool] = True):
    r"""Downsamples a voxelgrid, given a (down)scaling factor for each
    dimension.

    .. Note::
        The voxel output is not thresholded.

    Args:
        voxel (torch.Tensor): Voxel grid to be downsampled (shape: must
            be a tensor containing exactly 3 dimensions).
        scale (list): List of tensors to scale each dimension down by
            (length: 3).
        inplace (bool, optional): Bool to make the operation in-place.

    Returns:
        (torch.Tensor): Downsampled voxelgrid.

    Example:
        >>> x = torch.ones([32, 32, 32])
        >>> print (x.shape)
        torch.Size([32, 32, 32])
        >>> x = downsample(x, [2,2,2])
        >>> print (x.shape)
        torch.Size([16, 16, 16])
    """
    if isinstance(voxel, VoxelGrid):
        voxel = voxel.voxels
    voxel = confirm_def(voxel)

    if not inplace:
        voxel = voxel.clone()

    # Verify that all elements of `scale` are greater than or equal to 1 and
    # less than the voxel shape for the corresponding dimension.
    scale_filter = [1, 1]
    scale_factor = 1.
    for i in range(3):
        if scale[i] < 1:
            raise ValueError('Downsample ratio must be at least 1 along every'
                             ' dimension.')
        if scale[i] >= voxel.shape[i]:
            raise ValueError('Downsample ratio must be less than voxel shape'
                             ' along every dimension.')
        scale_filter.append(scale[i])
        scale_factor *= scale[i]
    conv_filter = torch.ones(scale_filter).to(voxel.device) / scale_factor

    voxel = F.conv3d(voxel.unsqueeze(0).unsqueeze(
        0), conv_filter, stride=scale, padding=0)
    voxel = voxel.squeeze(0).squeeze(0)

    return voxel


def upsample(voxel: torch.Tensor, dim: int):
    r"""Upsamples a voxel grid by a given scaling factor.

    .. Note::
        The voxel output is not thresholded.

    Args:
        voxel (torch.Tensor): Voxel grid to be upsampled (shape: must
            be a 3D tensor)
        dim (int): New dimensionality (number of voxels along each dimension
            in the resulting voxel grid).

    Returns:
        torch.Tensor: Upsampled voxel grid.

    Example:
        >>> x = torch.ones([32, 32, 32])
        >>> print (x.shape)
        torch.Size([32, 32, 32])
        >>> x = upsample(x, 64)
        >>> print (x.shape)
        torch.Size([64, 64, 64])
    """
    if isinstance(voxel, VoxelGrid):
        voxel = voxel.voxels
    voxel = confirm_def(voxel)

    cur_shape = voxel.shape
    assert (dim >= cur_shape[0]) and ((dim >= cur_shape[1]) and (
        dim >= cur_shape[2])), 'All dim values must be larger then current dim'

    new_positions = []
    old_positions = []

    # Defining position correspondences
    for i in range(3):
        shape_params = [1, 1, 1]
        shape_params[i] = dim
        new_pos = np.arange(dim).reshape(shape_params)

        for j in range(3):
            if i == j:
                continue
            new_pos = np.repeat(new_pos, dim, axis=j)
        new_pos = new_pos.reshape(-1)

        ratio = float(cur_shape[i]) / float(dim)
        old_pos = (new_pos * ratio).astype(int)

        new_positions.append(new_pos)
        old_positions.append(old_pos)

    scaled_voxel = torch.FloatTensor(np.zeros([dim, dim, dim])).to(
        voxel.device)
    if voxel.is_cuda:
        scaled_voxel = scaled_voxel.cuda()

    scaled_voxel[tuple(new_positions)] = voxel[tuple(old_positions)]

    return scaled_voxel


def fill(voxel: Union[torch.Tensor, VoxelGrid], thresh: float = .5):
    r""" Fills the internal structures in a voxel grid. Used to fill holds
    and 'solidify' objects.

    Args:
        voxel (torch.Tensor): Voxel grid to be filled.
        thresh (float): Threshold to use for binarization of the grid.

    Returns:
        torch.Tensor: filled voxel array
    """

    if isinstance(voxel, VoxelGrid):
        voxel = voxel.voxels
    voxel = confirm_def(voxel)
    voxel = threshold(voxel, thresh)
    voxel = voxel.clone()
    on = ndimage.binary_fill_holes(voxel.data.cpu())
    voxel[np.where(on)] = 1
    return voxel


def extract_odms(voxel: Union[torch.Tensor, VoxelGrid]):
    r"""Extracts an orthographic depth map from a voxel grid.

    Args:
        voxel (torch.Tensor): Voxel grid from which odms are extracted.

    Returns:
        (torch.Tensor): 6 ODMs from the 6 primary viewing angles.

    Example:
        >>> voxel = torch.ones([128,128,128])
        >>> voxel = extract_odms(voxel)
        >>> voxel.shape
        torch.Size([6, 128, 128])
    """
    if isinstance(voxel, VoxelGrid):
        voxel = voxel.voxels
    voxel = confirm_def(voxel)
    cuda = voxel.is_cuda
    voxel = extract_surface(voxel)
    voxel = voxel.data.cpu().numpy()

    dim = voxel.shape[-1]
    a, b, c = np.where(voxel == 1)
    big_list = [[[[dim, dim]
                  for j in range(dim)] for i in range(dim)] for k in range(3)]
    for i, j, k in zip(a, b, c):
        big_list[0][i][j][0] = (min(dim - k - 1, big_list[0][i][j][0]))
        big_list[0][i][j][1] = (min(k, big_list[0][i][j][1]))
        big_list[1][i][k][0] = (min(dim - j - 1, big_list[1][i][k][0]))
        big_list[1][i][k][1] = (min(j, big_list[1][i][k][1]))
        big_list[2][j][k][0] = (min(dim - i - 1, big_list[2][j][k][0]))
        big_list[2][j][k][1] = (min(i, big_list[2][j][k][1]))

    odms = np.zeros((6, dim, dim))
    big_list = np.array(big_list)
    for k in range(6):
        odms[k] = big_list[k // 2, :, :, k % 2]
    odms = torch.FloatTensor(np.array(odms))

    if cuda:
        odms = odms.cuda()

    return odms


def project_odms(odms: torch.Tensor,
                 voxel: torch.Tensor = None, votes: int = 1):
    r"""Projects orthographic depth map onto a voxel array.

    .. Note::
        If no voxel grid is provided, we poject onto a completely filled grid.

    Args:
        odms (torch.Tensor): ODMs which are to be projected.
        voxel (torch.Tensor): Voxel grid onto which ODMs are projected.

    Returns:
        (torch.Tensor): Updated voxel grid.

    Example:
        >>> odms = torch.rand([6,128,128])*128
        >>> odms = voxel.int()
        >>> voxel = project_odms(odms)
        >>> voxel.shape
        torch.Size([128, 128, 128])
    """
    cuda = odms.is_cuda
    dim = odms.shape[-1]
    subtractor = 1. / float(votes)

    if voxel is None:
        voxel = torch.ones((dim, dim, dim))
    else:
        for i in range(3):
            assert (voxel.shape[i] == odms.shape[-1]
                    ), 'Voxel and odm dimension size must be the same'
        if isinstance(voxel, VoxelGrid):
            voxel = voxel.voxels
        voxel = confirm_def(voxel)
        voxel = threshold(voxel, .5)
    voxel = voxel.data.cpu().numpy()
    odms = odms.data.cpu().numpy()

    for i in range(3):
        odms[2 * i] = dim - odms[2 * i]

    depths = np.where(odms <= dim)
    for x, y, z in zip(*depths):
        pos = int(odms[x, y, z])
        if x == 0:
            voxel[y, z, pos:dim] -= subtractor
        elif x == 1:
            voxel[y, z, 0:pos] -= subtractor
        elif x == 2:
            voxel[y, pos:dim, z] -= subtractor
        elif x == 3:
            voxel[y, 0:pos, z] -= subtractor
        elif x == 4:
            voxel[pos:dim, y, z] -= subtractor
        else:
            voxel[0:pos, y, z] -= subtractor

    on = np.where(voxel > 0)
    off = np.where(voxel <= 0)
    voxel[on] = 1
    voxel[off] = 0

    voxel = confirm_def(voxel)
    if cuda:
        voxel = voxel.cuda()
    return voxel


def confirm_def(voxgrid: torch.Tensor):
    r""" Checks that the definition of the voxelgrid is correct.

    Args:
        voxgrid (torch.Tensor): Passed voxelgrid.

    Return:
        (torch.Tensor): Voxel grid as torch.Tensor.

    """
    if isinstance(voxgrid, np.ndarray):
        voxgrid = torch.Tensor(voxgrid)
    helpers._assert_tensor(voxgrid)
    helpers._assert_dim_eq(voxgrid, 3)
    assert ((voxgrid.max() <= 1.) and (voxgrid.min() >= 0.)
            ), 'All values in passed voxel grid must be in range [0,1]'
    return voxgrid


def threshold(voxel: Union[torch.Tensor, VoxelGrid], thresh: float,
              inplace: Optional[bool] = True):
    r"""Binarizes the voxel array using a specified threshold.

    Args:
        voxel (torch.Tensor): Voxel array to be binarized.
        thresh (float): Threshold with which to binarize.
        inplace (bool, optional): Bool to make the operation in-place.

    Returns:
        (torch.Tensor): Thresholded voxel array.

    """
    if isinstance(voxel, VoxelGrid):
        voxel = voxel.voxels
    helpers._assert_tensor(voxel)
    if inplace:
        voxel[:] = voxel > thresh
    else:
        voxel = (voxel > thresh).type(voxel.dtype)
    return voxel


def extract_surface(voxel: Union[torch.Tensor, VoxelGrid], thresh: float = .5):
    r"""Removes any inernal structure(s) from a voxel array.

    Args:
        voxel (torch.Tensor): voxel array from which to extract surface
        thresh (float): threshold with which to binarize

    Returns:
        torch.Tensor: surface voxel array
    """

    if isinstance(voxel, VoxelGrid):
        voxel = voxel.voxels
    voxel = confirm_def(voxel)
    voxel = threshold(voxel, thresh)
    off_positions = voxel == 0

    conv_filter = torch.ones((1, 1, 3, 3, 3))
    surface_voxel = torch.zeros(voxel.shape)
    if voxel.is_cuda:
        conv_filter = conv_filter.cuda()
        surface_voxel = surface_voxel.cuda()

    local_occupancy = F.conv3d(voxel.unsqueeze(
        0).unsqueeze(0), conv_filter, padding=1)
    local_occupancy = local_occupancy.squeeze(0).squeeze(0)
    # only elements with exposed faces
    surface_positions = (local_occupancy < 27) * (local_occupancy > 0)
    surface_voxel[surface_positions] = 1
    surface_voxel[off_positions] = 0

    return surface_voxel


def voxelgrid_to_pointcloud(voxel: torch.Tensor, num_points: int,
                            thresh: float = .5, mode: str = 'full',
                            normalize: bool = True):
    r""" Converts  passed voxel to a pointcloud

    Args:
        voxel (torch.Tensor): voxel array
        num_points (int): number of points in converted point cloud
        thresh (float): threshold from which to make voxel binary
        mode (str):
            -'full': sample the whole voxel model
            -'surface': sample only the surface voxels
        normalize (bool): whether to scale the array to (-.5,.5)

    Returns:
       (torch.Tensor): converted pointcloud

    Example:
        >>> voxel = torch.ones([32,32,32])
        >>> points = voxelgrid_to_pointcloud(voxel, 10)
        >>> points
        tensor([[0.5674, 0.8994, 0.8606],
                [0.2669, 0.9445, 0.5501],
                [0.2252, 0.9674, 0.8198],
                [0.5127, 0.9347, 0.4470],
                [0.7981, 0.1645, 0.5405],
                [0.7384, 0.4255, 0.6084],
                [0.9881, 0.3629, 0.2747],
                [0.1690, 0.2880, 0.4849],
                [0.8844, 0.3866, 0.0557],
                [0.4829, 0.0413, 0.6700]])
        >>> points.shape
        torch.Size([10, 3])
    """

    assert (mode in ['full', 'surface'])
    voxel = confirm_def(voxel)
    voxel = threshold(voxel, thresh=thresh)

    if mode == 'surface':
        voxel = extract_surface(voxel)

    voxel_positions = (voxel == 1).nonzero().float()

    index_list = list(range(voxel_positions.shape[0]))
    select_index = np.random.choice(index_list, size=num_points)
    point_positions = voxel_positions[select_index]

    point_displacement = torch.rand(
        point_positions.shape).to(
        point_positions.device)
    point_positions += point_displacement

    if normalize:
        shape = torch.FloatTensor(
            np.array(
                voxel.shape)).to(
            point_positions.device)
        point_positions /= shape
        point_positions = point_positions - .5

    return point_positions


def voxelgrid_to_trianglemesh(voxel: torch.Tensor, thresh: int = .5,
                              mode: str = 'marching_cubes',
                              normalize: bool = True):
    r""" Converts  passed voxel to a mesh

    Args:
        voxel (torch.Tensor): voxel array
        thresh (float): threshold from which to make voxel binary
        mode (str):
            -'exact': exect mesh conversion
            -'marching_cubes': marching cubes is applied to passed voxel
        normalize (bool): whether to scale the array to (-.5,.5)

    Returns:
        (torch.Tensor): computed mesh properties

    Example:
        >>> voxel = torch.ones([32,32,32])
        >>> verts, faces = voxelgrid_to_trianglemesh(voxel)
        >>> [verts.shape, faces.shape]
        [torch.Size([6144, 3]), torch.Size([12284, 3])]

    """
    assert (mode in ['exact', 'marching_cubes'])
    voxel = confirm_def(voxel)
    voxel = threshold(voxel, thresh=thresh)
    voxel_np = np.array((voxel.cpu() > thresh)).astype(bool)
    trimesh_voxel = trimesh.voxel.VoxelGrid(voxel_np)

    if mode == 'exact':
        trimesh_voxel = trimesh_voxel.as_boxes()
    elif mode == 'marching_cubes':
        trimesh_voxel = trimesh_voxel.marching_cubes

    verts = torch.FloatTensor(trimesh_voxel.vertices)
    faces = torch.LongTensor(trimesh_voxel.faces)
    shape = torch.FloatTensor(np.array(voxel.shape))
    if voxel.is_cuda:
        verts = verts.cuda()
        faces = faces.cuda()
        shape = shape.cuda()
    if normalize:

        verts /= shape
        verts = verts - .5

    return verts, faces


def voxelgrid_to_quadmesh(voxel: torch.Tensor, thresh: str = .5,
                          normalize: bool = True):
    r""" Converts passed voxel to quad mesh

    Args:
        voxel (torch.Tensor): voxel array
        thresh (float): threshold from which to make voxel binary
        normalize (bool): whether to scale the array to (-.5,.5)

    Returns:
        (torch.Tensor): converted mesh properties

    Example:
        >>> voxel = torch.ones([32,32,32])
        >>> verts, faces = voxelgrid_to_quadmesh(voxel)
        >>> [verts.shape, faces.shape]
        [torch.Size([6144, 3]), torch.Size([6142, 4])]

    """
    voxel = confirm_def(voxel)
    voxel = threshold(voxel, thresh=thresh)
    dim = voxel.shape[0]
    new_voxel = np.zeros((dim + 2, dim + 2, dim + 2))
    new_voxel[1:dim + 1, 1:dim + 1, 1:dim + 1] = voxel.cpu()

    voxel = new_voxel

    vert_dict = {}
    verts = []
    faces = []
    curr_vert_num = 1
    a, b, c = np.where(voxel == 1)
    for i, j, k in zip(a, b, c):

        # top
        if voxel[i, j, k + 1] != 1:
            vert_dict, verts, faces, curr_vert_num = _add_face(
                0, vert_dict, verts, faces, [i, j, k], curr_vert_num)
        # bottom
        if voxel[i, j, k - 1] != 1:
            vert_dict, verts, faces, curr_vert_num = _add_face(
                1, vert_dict, verts, faces, [i, j, k], curr_vert_num)
        # left
        if voxel[i - 1, j, k] != 1:
            vert_dict, verts, faces, curr_vert_num = _add_face(
                2, vert_dict, verts, faces, [i, j, k], curr_vert_num)
        # right
        if voxel[i + 1, j, k] != 1:
            vert_dict, verts, faces, curr_vert_num = _add_face(
                3, vert_dict, verts, faces, [i, j, k], curr_vert_num)
        # front
        if voxel[i, j - 1, k] != 1:
            vert_dict, verts, faces, curr_vert_num = _add_face(
                4, vert_dict, verts, faces, [i, j, k], curr_vert_num)
        # back
        if voxel[i, j + 1, k] != 1:
            vert_dict, verts, faces, curr_vert_num = _add_face(
                5, vert_dict, verts, faces, [i, j, k], curr_vert_num)
    verts = torch.FloatTensor(np.array(verts))
    faces = torch.LongTensor(np.array(faces) - 1)
    if normalize:
        shape = torch.FloatTensor(np.array(voxel.shape))
        verts /= shape
        verts = verts - .5

    return verts, faces


def _add_face(vert_set_index: int, vert_dict: dict, verts: torch.Tensor,
              faces: torch.Tensor, location: list, curr_vert_num: int):
    r""" Adds a face to the set of observed faces and verticies
    """
    top_verts = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    bottom_verts = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]])
    left_verts = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]])
    right_verts = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]])
    front_verts = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]])
    back_verts = np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]])
    vert_sets = [top_verts, bottom_verts, left_verts,
                 right_verts, front_verts, back_verts]

    vert_set = vert_sets[vert_set_index]
    face = np.array([0, 1, 2, 3]) + curr_vert_num
    update = 4
    for e, vs in enumerate(vert_set):
        new_position = vs + np.array(location)
        if str(new_position) in vert_dict:
            index = vert_dict[str(new_position)]
            face[e] = index
            for idx in range(e + 1, 4):
                face[idx] -= 1
            update -= 1
        else:
            vert_dict[str(new_position)] = len(verts) + 1
            verts.append(list(new_position))
    faces.append(face)
    curr_vert_num += update
    return vert_dict, verts, faces, curr_vert_num


def voxelgrid_to_sdf(voxel: torch.Tensor, thresh: float = .5,
                     normalize: bool = True):
    r""" Converts passed voxel to a signed distance function

    Args:
        voxel (torch.Tensor): voxel array
        thresh (float): threshold from which to make voxel binary
        normalize (bool): whether to scale the array to (0,1)

    Returns:
        a signed distance function

    Example:
        >>> voxel = torch.ones([32,32,32])
        >>> sdf = voxelgrid_to_sdf(voxel)
        >>> distances = sdf(torch.rand(100,3))
    """

    voxel = confirm_def(voxel)
    voxel = threshold(voxel, thresh=thresh)
    on_points = (voxel > .5).nonzero().float()

    if normalize:
        on_points = on_points / float(voxel.shape[0])
        on_points -= .5

    distance_fn = kal.metrics.point.directed_distance

    def eval_query(query):
        distances = distance_fn(query, on_points, mean=False)
        if normalize:
            query = ((query + .5) * (voxel.shape[0] - 1))

        query = np.floor(query.data.cpu().numpy())
        query_positions = [query[:, 0], query[:, 1], query[:, 2]]
        values = voxel[query_positions]
        distances[values == 1] = 0

        return distances

    return eval_query
