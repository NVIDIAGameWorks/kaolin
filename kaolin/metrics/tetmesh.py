# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
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
from kaolin.ops.mesh.tetmesh import _validate_tet_vertices


def tetrahedron_volume(tet_vertices):
    r"""Compute the volume of tetrahedrons.

    Args:
        tet_vertices (torch.Tensor):
            Batched tetrahedrons, of shape
            :math:`(\text{batch_size}, \text{num_tetrahedrons}, 4, 3)`.
    Returns:
        (torch.Tensor):
            volume of each tetrahedron in each mesh, of shape
            :math:`(\text{batch_size}, \text{num_tetrahedrons})`.

    Example:
        >>> tet_vertices = torch.tensor([[[[0.5000, 0.5000, 0.4500],
        ...                                [0.4500, 0.5000, 0.5000],
        ...                                [0.4750, 0.4500, 0.4500],
        ...                                [0.5000, 0.5000, 0.5000]]]])
        >>> tetrahedron_volume(tet_vertices)
        tensor([[-2.0833e-05]])
    """
    _validate_tet_vertices(tet_vertices)

    # split the tensor
    A, B, C, D = [split.squeeze(2) for split in
                  torch.split(tet_vertices, split_size_or_sections=1, dim=2)]

    # compute the volume of each tetrahedron directly by using V = |(a - d) * ((b - d) x (c - d))| / 6
    volumes = torch.div(
        ((A - D) * torch.cross(input=(B - D), other=(C - D), dim=2)).sum(dim=2), 6)

    return volumes

def equivolume(tet_vertices, tetrahedrons_mean=None, pow=4):
    r"""Compute the EquiVolume loss as devised by *Gao et al.* in `Learning Deformable Tetrahedral Meshes for 3D
    Reconstruction <https://nv-tlabs.github.io/DefTet/>`_ NeurIPS 2020.
    See `supplementary material <https://nv-tlabs.github.io/DefTet/files/supplement.pdf>`_ for the definition of the loss function.

    Args:
        tet_vertices (torch.Tensor):
            Batched tetrahedrons, of shape
            :math:`(\text{batch_size}, \text{num_tetrahedrons}, 4, 3)`.
        tetrahedrons_mean (torch.Tensor):
            Mean volume of all tetrahedrons in a grid,
            of shape :math:`(\text{batch_size})` or :math:`(1,)` (broadcasting).
            Default: Compute ``torch.mean(tet_vertices, dim=-1)``.
        pow (int):
            Power for the equivolume loss.
            Increasing power puts more emphasis on the larger tetrahedron deformation.
            Default: 4.

    Returns:
        (torch.Tensor):
            EquiVolume loss for each mesh, of shape :math:`(\text{batch_size})`.

    Example:
        >>> tet_vertices = torch.tensor([[[[0.5000, 0.5000, 0.7500],
        ...                                [0.4500, 0.8000, 0.6000],
        ...                                [0.4750, 0.4500, 0.2500],
        ...                                [0.5000, 0.3000, 0.3000]],
        ...                               [[0.4750, 0.4500, 0.2500],
        ...                                [0.5000, 0.9000, 0.3000],
        ...                                [0.4500, 0.4000, 0.9000],
        ...                                [0.4500, 0.4500, 0.7000]]],
        ...                              [[[0.7000, 0.3000, 0.4500],
        ...                                [0.4800, 0.2000, 0.3000],
        ...                                [0.9000, 0.4500, 0.4500],
        ...                                [0.2000, 0.5000, 0.1000]],
        ...                               [[0.3750, 0.4500, 0.2500],
        ...                                [0.9000, 0.8000, 0.7000],
        ...                                [0.6000, 0.9000, 0.3000],
        ...                                [0.5500, 0.3500, 0.9000]]]])
        >>> equivolume(tet_vertices, pow=4)
        tensor([[2.2961e-10],
                [7.7704e-10]])
    """
    _validate_tet_vertices(tet_vertices)

    # compute the volume of each tetrahedron
    volumes = tetrahedron_volume(tet_vertices)

    if tetrahedrons_mean is None:
        # finding the mean volume of all tetrahedrons in the tetrahedron grid
        tetrahedrons_mean = torch.mean(volumes, dim=-1)
    tetrahedrons_mean = tetrahedrons_mean.reshape(1, -1)
    # compute EquiVolume loss
    equivolume_loss = torch.mean(torch.pow(
        torch.abs(volumes - tetrahedrons_mean), exponent=pow),
        dim=-1, keepdim=True)

    return equivolume_loss


def amips(tet_vertices, inverse_offset_matrix):
    r"""Compute the AMIPS (Advanced MIPS) loss as devised by *Fu et al.* in
    `Computing Locally Injective Mappings by Advanced MIPS. \
    <https://www.microsoft.com/en-us/research/publication/computing-locally-injective-mappings-advanced-mips/>`_
    ACM Transactions on Graphics (TOG) - Proceedings of ACM SIGGRAPH 2015.

    The Jacobian can be derived as: :math:`J = (g(x) - g(x_0)) / (x - x_0)`

    Only components where the determinant of the Jacobian is positive, are included in the calculation of AMIPS.
    This is because the AMIPS Loss is only defined for tetrahedrons whose determinant of the Jacobian is positive.

    Args:
        tet_vertices (torch.Tensor):
            Batched tetrahedrons, of shape
            :math:`(\text{batch_size}, \text{num_tetrahedrons}, 4, 3)`.
        inverse_offset_matrix (torch.LongTensor):
            The inverse of the offset matrix is of shape
            :math:`(\text{batch_size}, \text{num_tetrahedrons}, 3, 3)`.
            Refer to :func:`kaolin.ops.mesh.tetmesh.inverse_vertices_offset`.
    Returns:
        (torch.Tensor):
            AMIPS loss for each mesh, of shape :math:`(\text{batch_size})`.

    Example:
        >>> tet_vertices = torch.tensor([[[[1.7000, 2.3000, 4.4500],
        ...                                [3.4800, 0.2000, 5.3000],
        ...                                [4.9000, 9.4500, 6.4500],
        ...                                [6.2000, 8.5000, 7.1000]],
        ...                               [[-1.3750, 1.4500, 3.2500],
        ...                                [4.9000, 1.8000, 2.7000],
        ...                                [3.6000, 1.9000, 2.3000],
        ...                                [1.5500, 1.3500, 2.9000]]],
        ...                              [[[1.7000, 2.3000, 4.4500],
        ...                                [3.4800, 0.2000, 5.3000],
        ...                                [4.9000, 9.4500, 6.4500],
        ...                                [6.2000, 8.5000, 7.1000]],
        ...                               [[-1.3750, 1.4500, 3.2500],
        ...                                [4.9000, 1.8000, 2.7000],
        ...                                [3.6000, 1.9000, 2.3000],
        ...                                [1.5500, 1.3500, 2.9000]]]])
        >>> inverse_offset_matrix = torch.tensor([[[[ -1.1561, -1.1512, -1.9049],
        ...                                         [1.5138,  1.0108,  3.4302],
        ...                                         [1.6538, 1.0346,  4.2223]],
        ...                                        [[ 2.9020,  -1.0995, -1.8744],
        ...                                         [ 1.1554,  1.1519, 1.7780],
        ...                                         [-0.0766, 1.6350,  1.1064]]],
        ...                                        [[[-0.9969,  1.4321, -0.3075],
        ...                                         [-1.3414,  1.5795, -1.6571],
        ...                                         [-0.1775, -0.4349,  1.1772]],
        ...                                        [[-1.1077, -1.2441,  1.8037],
        ...                                         [-0.5722, 0.1755, -2.4364],
        ...                                         [-0.5263,  1.5765,  1.5607]]]])
        >>> amips(tet_vertices, inverse_offset_matrix)
        tensor([[13042.3408],
                [ 2376.2517]])
    """
    _validate_tet_vertices(tet_vertices)

    # split the tensor
    A, B, C, D = torch.split(tet_vertices, split_size_or_sections=1, dim=2)

    # compute the offset matrix of the tetrahedrons w.r.t. vertex A.
    offset_matrix = torch.cat([B - A, C - A, D - A], dim=2)

    # compute the Jacobian for each tetrahedron - the Jacobian represents the unique 3D deformation that transforms the
    # tetrahedron t into a regular tetrahedron.
    jacobian = torch.matmul(offset_matrix, inverse_offset_matrix)

    # compute determinant of Jacobian
    j_det = torch.det(jacobian)

    # compute the trace of J * J.T
    jacobian_squared = torch.matmul(jacobian, torch.transpose(jacobian, -2, -1))
    trace = torch.diagonal(jacobian_squared, dim1=-2, dim2=-1).sum(-1)

    # compute the determinant of the Jacobian to the 2/3
    EPS = 1e-10
    denominator = torch.pow(torch.pow(j_det, 2) + EPS, 1 / 3)

    # compute amips energy for positive tetrahedrons whose determinant of their Jacobian is positive
    amips_energy = torch.mean(torch.div(trace, denominator) * (j_det >= 0).float(),
                              dim=1, keepdim=True)

    return amips_energy
