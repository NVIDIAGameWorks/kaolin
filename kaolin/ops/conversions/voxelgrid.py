# Copyright (c) 2019,20-22 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

import torch
import torch.nn.functional as F

from kaolin import _C

__all__ = ['voxelgrids_to_cubic_meshes', 'voxelgrids_to_trianglemeshes']

verts_template = torch.tensor(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0]
    ],
    dtype=torch.float
)

faces_template = torch.tensor(
    [
        [0, 2, 1, 3],
        [0, 1, 4, 5],
        [0, 4, 2, 6]
    ],
    dtype=torch.int64
)
faces_3x4x3 = verts_template[faces_template]
for i in range(3):
    faces_3x4x3[i, :, (i - 1) % 3] -= 1
    faces_3x4x3[i, :, (i + 1) % 3] -= 1

quad_face = torch.LongTensor([[0, 1, 3, 2]])
kernel = torch.zeros((1, 1, 2, 2, 2))
kernel[..., 0, 0, 0] = -1
kernel[..., 1, 0, 0] = 1
kernels = torch.cat([kernel, kernel.transpose(
    2, 3), kernel.transpose(2, 4)], 0)  # (3,1,2,2,2)

def voxelgrids_to_cubic_meshes(voxelgrids, is_trimesh=True):
    r"""Convert voxelgrids to meshes by replacing each occupied voxel with a cuboid mesh (unit cube). 
    Each cube has 8 vertices and 6 (for quadmesh) or 12 faces 
    (for triangular mesh). Internal faces are ignored. 
    If `is_trimesh==True`, this function performs the same operation
    as "Cubify" defined in the ICCV 2019 paper "Mesh R-CNN": 
    https://arxiv.org/abs/1906.02739.

    Args:
        voxelgrids (torch.Tensor): binary voxel array, of shape
                                   :math:`(\text{batch_size}, \text{X}, \text{Y}, \text{Z})`.
        is_trimesh (optional, bool): if True, the outputs are triangular meshes.
                                     Otherwise quadmeshes are returned. Default: True.

    Returns:
        (list[torch.Tensor], list[torch.LongTensor]):

            - The list of vertices for each mesh.
            - The list of faces for each mesh.

    Example:
        >>> voxelgrids = torch.ones((1, 1, 1, 1))
        >>> verts, faces = voxelgrids_to_cubic_meshes(voxelgrids)
        >>> verts[0]
        tensor([[0., 0., 0.],
                [0., 0., 1.],
                [0., 1., 0.],
                [0., 1., 1.],
                [1., 0., 0.],
                [1., 0., 1.],
                [1., 1., 0.],
                [1., 1., 1.]])
        >>> faces[0]
        tensor([[0, 1, 2],
                [5, 4, 7],
                [0, 4, 1],
                [6, 2, 7],
                [0, 2, 4],
                [3, 1, 7],
                [3, 2, 1],
                [6, 7, 4],
                [5, 1, 4],
                [3, 7, 2],
                [6, 4, 2],
                [5, 7, 1]])
    """
    device = voxelgrids.device
    voxelgrids = voxelgrids.unsqueeze(1)
    batch_size = voxelgrids.shape[0]

    face = quad_face.to(device)

    if device == 'cpu':
        k = kernels.to(device).half()
        voxelgrids = voxelgrids.half()
    else:
        k = kernels.to(device).float()
        voxelgrids = voxelgrids.float()

    conv_results = torch.nn.functional.conv3d(
        voxelgrids, k, padding=1).round()  # (B, 3, r, r, r)

    indices = torch.nonzero(conv_results.transpose(
        0, 1), as_tuple=True)  # (N, 5)
    dim, batch, loc = indices[0], indices[1], torch.stack(
        indices[2:], -1)  # (N,) , (N, ), (N, 3)
    invert = conv_results.transpose(0, 1)[indices] == -1
    _, counts = torch.unique(dim, sorted=True, return_counts=True)

    faces_loc = (torch.repeat_interleave(faces_3x4x3.to(device), counts, dim=0) +
                 loc.unsqueeze(1).float())  # (N, 4, 3)

    faces_batch = []
    verts_batch = []

    for b in range(batch_size):
        verts = faces_loc[torch.nonzero(batch == b)].view(-1, 3)
        if verts.shape[0] == 0:
            faces_batch.append(torch.zeros((0, 3 if is_trimesh else 4), device=device, dtype=torch.long))
            verts_batch.append(torch.zeros((0, 3), device=device))
            continue
        invert_batch = torch.repeat_interleave(
            invert[batch == b], face.shape[0], dim=0)
        N = verts.shape[0] // 4

        shift = torch.arange(N, device=device).unsqueeze(1) * 4  # (N,1)
        faces = (face.unsqueeze(0) + shift.unsqueeze(1)
                 ).view(-1, face.shape[-1])  # (N, 4) or (2N, 3)
        faces[invert_batch] = torch.flip(faces[invert_batch], [-1])

        if is_trimesh:
            faces = torch.cat(
                [faces[:, [0, 3, 1]], faces[:, [2, 1, 3]]], dim=0)

        verts, v = torch.unique(
            verts, return_inverse=True, dim=0)
        faces = v[faces.reshape(-1)].reshape((-1, 3 if is_trimesh else 4))
        faces_batch.append(faces)
        verts_batch.append(verts)

    return verts_batch, faces_batch

class MarchingCubesLorensenCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voxelgrid, iso_value):
        vertices, faces = _C.ops.conversions.unbatched_mcube_forward_cuda(voxelgrid, iso_value)
        return vertices, faces

    @staticmethod
    def backward(ctx, gradoutput):
        # TODO: do a custom backward pass.
        return None, None

def voxelgrids_to_trianglemeshes(voxelgrids, iso_value=0.5):
    r"""Converts voxelgrids to triangle meshes using marching cube algorithm.
    Please refer to: *Lorensen, William E.; Cline, Harvey E.* in
    `Marching cubes, A high resolution 3D surface construction algorithm`_

    Args:
        voxelgrids (torch.Tensor):
            Exact batched voxel array with shape
            :math:`(\text{batch_size}, \text{X}, \text{Y}, \text{Z})`.
        iso_value (optional, float):
            Value in the range :math:`[0, 1]` used to determine whether a voxel is inside the
            surface or not. Isovalue is also used to interpolate 
            newly created triangle vertices. Defaults to 0.5

    Returns:
        (list[torch.FloatTensor], list[torch.LongTensor]):

            - The list of vertices of each mesh.
            - The list of faces of each mesh.

    Example:
        >>> voxelgrid = torch.tensor([[[[1, 0], 
        ...                             [0, 0]], 
        ...                            [[0, 0], 
        ...                             [0, 0]]]], device='cuda', dtype=torch.uint8)
        >>> vertices, faces = voxelgrids_to_trianglemeshes(voxelgrid)
        >>> vertices[0]
        tensor([[1.0000, 1.0000, 0.5000],
                [1.0000, 0.5000, 1.0000],
                [0.5000, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.5000],
                [1.0000, 1.5000, 1.0000],
                [1.5000, 1.0000, 1.0000]], device='cuda:0')
        >>> faces[0]
        tensor([[0, 1, 2],
                [3, 2, 1],
                [4, 0, 2],
                [4, 2, 3],
                [0, 5, 1],
                [5, 3, 1],
                [4, 5, 0],
                [5, 4, 3]], device='cuda:0')

    .. _Marching cubes, A high resolution 3D surface construction algorithm:
        https://www.researchgate.net/publication/202232897_Marching_Cubes_A_High_Resolution_3D_Surface_Construction_Algorithm
    """
    # TODO: There is a bug in pytorch 1.7 and cuda 11.0, which for certain cuda operations, the value won't be written
    # to the tensor. However, this but does not exist in pytorch1.6 and cuda 10.0. Need to look into it in the future.
    voxelgrid_type = voxelgrids.dtype
    voxelgrid_device = voxelgrids.device

    batch_size = voxelgrids.shape[0]

    if not voxelgrids.is_cuda:
        raise NotImplementedError("voxelgrids_to_trianglemeshes does not support CPU.")

    # TODO: support half and double.
    voxelgrids = voxelgrids.float()
    # Pad the voxelgrid with 0 in all three dimensions
    voxelgrids = F.pad(voxelgrids, (1, 1, 1, 1, 1, 1), 'constant', 0)

    vertices_list = []
    faces_list = []
    for i in range(batch_size):
        curr_voxelgrid = voxelgrids[i]

        if torch.all(curr_voxelgrid == 0):  # Don't bother if the voxelgrid is all zeros
            vertices_list.append(torch.zeros((0, 3), dtype=torch.float, device=voxelgrid_device))
            faces_list.append(torch.zeros((0, 3), dtype=torch.long, device=voxelgrid_device))
            continue

        vertices, faces = MarchingCubesLorensenCuda.apply(curr_voxelgrid, iso_value)
        faces = faces.long()

        vertices_list.append(vertices)
        faces_list.append(faces)

    return vertices_list, faces_list
