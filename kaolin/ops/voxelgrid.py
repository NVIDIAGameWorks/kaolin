# Copyright (c) 2019,20 NVIDIA CORPORATION & AFFILIATES.
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
from scipy import ndimage


def downsample(voxelgrids, scale):
    r"""Downsamples a voxelgrids, given a (down)scaling factor for each
    dimension.

    .. Note::
        The voxelgrids output is not thresholded.

    Args:
        voxelgrids (torch.Tensor): voxelgrids to be downsampled, of shape
                                   :math:`(\text{batch_size}, \text{X}, \text{Y}, \text{Z})`.
        scale (list or tuple or int): List or tuple of int of length 3 to scale each dimension down.
                                      or an int to scale down for every dimension.

    Returns:
        (torch.Tensor): Downsampled voxelgrids.

    Example:
        >>> voxelgrids2 = torch.zeros((1, 4, 4, 4))
        >>> voxelgrids2[0, 0] = 1
        >>> voxelgrids2[0, 1] = 0.4
        >>> voxelgrids2[0, 3] = 0.8
        >>> downsample(voxelgrids2, 2)
        tensor([[[[0.7000, 0.7000],
                  [0.7000, 0.7000]],
        <BLANKLINE>
                 [[0.4000, 0.4000],
                  [0.4000, 0.4000]]]])
    """
    voxelgrids = _force_float(voxelgrids)

    try:
        output = F.avg_pool3d(voxelgrids.unsqueeze(1), kernel_size=scale,
                              stride=scale, padding=0)
    except RuntimeError as err:
        if isinstance(scale, list) and len(scale) != 3:
            scale_length = len(scale)
            raise ValueError(f"Expected scale to have 3 dimensions "
                             f"but got {scale_length} dimensions.")



        if voxelgrids.ndim != 4:
            voxelgrids_dim = voxelgrids.ndim
            raise ValueError(f"Expected voxelgrids to have 4 dimensions " 
                             f"but got {voxelgrids_dim} dimensions.")

        for i in range(3):
            if scale[i] < 1:
                scale_value = scale[i]
                raise ValueError(f"Downsample ratio must be at least 1 "
                                 f"along every dimension but got {scale_value} at "
                                 f"index {i}.")

            if scale[i] > voxelgrids.shape[i + 1]:
                voxelgrids_shape_val = voxelgrids.shape[i + 1]
                scale_val = scale[i]
                raise ValueError(f"Downsample ratio must be less than voxelgrids "
                                 f"shape of {voxelgrids_shape_val} at index {i}, but got {scale_val}.")
        raise err  # unknown error

    except TypeError as err:
        if not isinstance(scale, list) and not isinstance(scale, int):
            scale_type = type(scale)
            raise TypeError(f"Expected scale to be type list or int "
                            f"but got {scale_type}.")

        raise err  # unknown error

    return output.squeeze(1)


def extract_surface(voxelgrids, mode="wide"):
    r"""Removes any internal structure(s) from a voxelgrids.

    Args:
        voxelgrids (torch.Tensor): Binary voxelgrids of shape (N, X, Y ,Z)
                                   from which to extract surface
        mode (str): Either "wide" or "thin". Each voxel can be seen as a cube in a grid.
                    "wide" mode keeps each filled voxel with at least one vertex in contact
                    with an empty voxel. "thin" mode keeps each filled voxel with at least
                    one face in contact with an empty voxel.

    Returns:
        torch.BoolTensor: binary surface voxelgrids tensor

    Example:
        >>> voxelgrids = torch.ones((1, 3, 3, 3))
        >>> output = extract_surface(voxelgrids)
        >>> output[0]
        tensor([[[ True,  True,  True],
                 [ True,  True,  True],
                 [ True,  True,  True]],
        <BLANKLINE>
                [[ True,  True,  True],
                 [ True, False,  True],
                 [ True,  True,  True]],
        <BLANKLINE>
                [[ True,  True,  True],
                 [ True,  True,  True],
                 [ True,  True,  True]]])
    """
    voxelgrids = _force_float(voxelgrids)

    if voxelgrids.ndim != 4:
        voxelgrids_dim = voxelgrids.ndim
        raise ValueError(f"Expected voxelgrids to have 4 dimensions "
                         f"but got {voxelgrids_dim} dimensions.")

    if mode == "wide":
        output = F.avg_pool3d(voxelgrids.unsqueeze(1), kernel_size=(3, 3, 3), padding=1, stride=1).squeeze(1)
        output = (output < 1) * voxelgrids.bool()
    elif mode == "thin":
        output_x = F.avg_pool3d(voxelgrids.unsqueeze(1), kernel_size=(3, 1, 1), padding=(1, 0, 0), stride=1).squeeze(1)
        output_y = F.avg_pool3d(voxelgrids.unsqueeze(1), kernel_size=(1, 3, 1), padding=(0, 1, 0), stride=1).squeeze(1)
        output_z = F.avg_pool3d(voxelgrids.unsqueeze(1), kernel_size=(1, 1, 3), padding=(0, 0, 1), stride=1).squeeze(1)
        output = ((output_x < 1) | (output_y < 1) | (output_z < 1)) * voxelgrids.bool()
    else:
        raise ValueError(f'mode "{mode}" is not supported.')

    return output


def fill(voxelgrids):
    r""" Fills the internal structures in a voxelgrids grid. Used to fill holes
    and 'solidify' objects.

    .. Note::
        This function is not differentiable.

    Args:
        voxelgrids (torch.Tensor): binary voxelgrids of size (N, X, Y, Z) to be filled.

    Returns:
        torch.BoolTensor: filled, binary voxelgrids array

    Example:
        >>> voxelgrids = torch.Tensor(
        ...              [[[[0., 0., 0., 0., 0.],
        ...                 [0., 1., 1., 1., 1.],
        ...                 [0., 1., 1., 1., 1.],
        ...                 [0., 1., 1., 1., 1.]],
        ...                [[0., 0., 0., 0., 0.],
        ...                 [0., 1., 1., 1., 1.],
        ...                 [0., 1., 0., 0., 1.],
        ...                 [0., 1., 1., 1., 1.]],
        ...                [[0., 0., 0., 0., 0.],
        ...                 [0., 1., 1., 1., 1.],
        ...                 [0., 1., 1., 1., 1.],
        ...                 [0., 1., 1., 1., 1.]]]])
        >>> fill(voxelgrids)
        tensor([[[[False, False, False, False, False],
                  [False,  True,  True,  True,  True],
                  [False,  True,  True,  True,  True],
                  [False,  True,  True,  True,  True]],
        <BLANKLINE>
                 [[False, False, False, False, False],
                  [False,  True,  True,  True,  True],
                  [False,  True,  True,  True,  True],
                  [False,  True,  True,  True,  True]],
        <BLANKLINE>
                 [[False, False, False, False, False],
                  [False,  True,  True,  True,  True],
                  [False,  True,  True,  True,  True],
                  [False,  True,  True,  True,  True]]]])
    """
    if voxelgrids.ndim != 4:
        voxelgrids_dim = voxelgrids.ndim
        raise ValueError(f"Expected voxelgrids to have 4 dimensions " 
                         f"but got {voxelgrids_dim} dimensions.")


    dtype = voxelgrids.dtype
    device = voxelgrids.device

    if voxelgrids.is_cuda:
        raise NotImplementedError("Fill function is not supported on GPU yet.")

    voxelgrids = voxelgrids.data.cpu()

    output = []
    for i in range(voxelgrids.shape[0]): 
        on = ndimage.binary_fill_holes(voxelgrids[i])
        output.append(on)

    output = torch.tensor(output, dtype=torch.bool, device=device)
    return output

def extract_odms(voxelgrids):
    r"""Extracts orthographic depth maps from voxelgrids.

    Args:
        voxelgrids (torch.Tensor): Binary voxelgrids of shape (N, dim, dim, dim) from 
                                   which odms are extracted.

    Returns:
        (torch.LongTensor): Batched ODMs of shape (N, 6, dim, dim) from the 6 primary viewing angles.
        The face order is z_neg, z_pos, y_neg, y_pos, x_neg, x_pos, denoting the axis and direction
        we are looking at.

    Example:
        >>> voxelgrids = torch.ones((2, 2, 2, 2))
        >>> voxelgrids[0, :, 0, :] = 0 # Set the front face to be zeros
        >>> output = extract_odms(voxelgrids)
        >>> output
        tensor([[[[2, 0],
                  [2, 0]],
        <BLANKLINE>
                 [[2, 0],
                  [2, 0]],
        <BLANKLINE>
                 [[0, 0],
                  [0, 0]],
        <BLANKLINE>
                 [[1, 1],
                  [1, 1]],
        <BLANKLINE>
                 [[2, 2],
                  [0, 0]],
        <BLANKLINE>
                 [[2, 2],
                  [0, 0]]],
        <BLANKLINE>
        <BLANKLINE>
                [[[0, 0],
                  [0, 0]],
        <BLANKLINE>
                 [[0, 0],
                  [0, 0]],
        <BLANKLINE>
                 [[0, 0],
                  [0, 0]],
        <BLANKLINE>
                 [[0, 0],
                  [0, 0]],
        <BLANKLINE>
                 [[0, 0],
                  [0, 0]],
        <BLANKLINE>
                 [[0, 0],
                  [0, 0]]]])
    """
    # Cast input to torch.bool to make it run faster.
    voxelgrids = voxelgrids.bool()
    device = voxelgrids.device
    dtype = voxelgrids.dtype

    dim = voxelgrids.shape[-1]
    batch_num = voxelgrids.shape[0]

    multiplier = torch.arange(1, dim + 1, device=device)
    reverse_multiplier = torch.arange(dim, 0, step=-1, device=device)
    full_multiplier = torch.cat([multiplier, reverse_multiplier], dim=0)

    # z_axis
    z_axis = voxelgrids.unsqueeze(1) * full_multiplier.view(1, 2, 1, 1, -1)
    z_axis_values, _ = torch.max(z_axis, dim=4)

    # y_axis
    y_axis = voxelgrids.unsqueeze(1) * full_multiplier.view(1, 2, 1, -1, 1)
    y_axis_values, _ = torch.max(y_axis, dim=3)

    # x_axis
    x_axis = voxelgrids.unsqueeze(1) * full_multiplier.view(1, 2, -1, 1, 1)
    x_axis_values, _ = torch.max(x_axis, dim=2)
    return dim - torch.cat([z_axis_values, y_axis_values, x_axis_values], dim=1)

def _force_float(input_tensor):
    r""" Cast the tensor to the smallest floating point dtype if it's a torch.BoolTensor.
         If it's a torch.BoolTensor on cpu then cast to torch.float,
         If it's a torch.cuda.BoolTensor then cast to torch.half,
         otherwise don't cast.

    Args:
        input_tensor (torch.Tensor)

    Returns:
        torch.Tensor: The cast tensor of either type torch.half or torch.float if input
                      is of type torch.bool, depending on the device. Else, voxelgrids
                      type is unchanged.
    """
    input_dtype = input_tensor.dtype
    if input_dtype == torch.bool:
        output_dtype = torch.half if input_tensor.is_cuda else torch.float
        input_tensor = input_tensor.type(output_dtype)
    return input_tensor

def project_odms(odms, voxelgrids=None, votes=1):
    r"""Projects orthographic depth map onto voxelgrids.

    .. Note::
        If no voxelgrids is provided, we project onto a completely filled grids.

    Args:
        odms (torch.Tensor): Batched ODMs of shape (N, 6, dim, dim) from the 6 primary viewing angles.
            The face order is z_neg, z_pos, y_neg, y_pos, x_neg, x_pos, denoting the axis and direction
            we are looking at.

        voxelgrids (torch.Tensor): Binary voxelgrids onto which ODMs are projected.

        votes (int): int from range(0, 7). Votes needed to substract a voxel to 0.

    Returns:
        (torch.BoolTensor): Updated binary voxel grid.

    Example:
        >>> odms = torch.zeros((1, 6, 2, 2))  # empty odms
        >>> odms[0, 1, 1, 1] = 2  # Change z_pos surface
        >>> project_odms(odms)
        tensor([[[[ True,  True],
                  [ True,  True]],
        <BLANKLINE>
                 [[ True,  True],
                  [False, False]]]])
        >>> project_odms(odms, votes=2)
        tensor([[[[True, True],
                  [True, True]],
        <BLANKLINE>
                 [[True, True],
                  [True, True]]]])
    """
    # Check the second dimension of odms
    if odms.shape[1] != 6:
        raise ValueError(f"Expected odms' second dimension to be 6, "
                         f"but got {odms.shape[1]} instead.")

    device = odms.device
    dtype = odms.dtype
    batch_size = odms.shape[0]
    dim = odms.shape[-1]

    if voxelgrids is None:
        voxelgrids = torch.ones((batch_size, dim, dim, dim), dtype=torch.bool, device=device)
    else:
        voxel_batch = voxelgrids.shape[0]
        if batch_size != voxelgrids.shape[0]:
            raise ValueError(f"Expected voxelgrids and odms' batch size to be the same, "
                             f"but got {batch_size} for odms and {voxel_batch} for voxelgrid.")

        for i in voxelgrids.shape[1:]:
            if i != dim:
                raise ValueError(f"Expected voxelgrids and odms' dimension size to be the same, "
                                 f"but got {dim} for odms and {i} for voxelgrid.")

    updated_odms = odms.clone()
    updated_odms = updated_odms.view(batch_size, 3, 2, dim, dim)
    updated_odms[:, :, 0] = dim - updated_odms[:, :, 0]
    updated_odms = updated_odms.view(batch_size, 6, dim, dim)

    base_idx = torch.arange(dim, device=device)

    pos0 = updated_odms[:, 0]
    pos1 = updated_odms[:, 1]
    pos2 = updated_odms[:, 2]
    pos3 = updated_odms[:, 3]
    pos4 = updated_odms[:, 4]
    pos5 = updated_odms[:, 5]

    z_neg_mask = (base_idx.view(1, 1, 1, -1) >= pos0.unsqueeze(-1)).byte()  # shape (2, 3, 3, 3)
    z_pos_mask = (base_idx.view(1, 1, 1, -1) < pos1.unsqueeze(-1)).byte()

    y_neg_mask = (base_idx.view(1, 1, -1, 1) >= pos2.unsqueeze(-2)).byte()
    y_pos_mask = (base_idx.view(1, 1, -1, 1) < pos3.unsqueeze(-2)).byte()

    x_neg_mask = (base_idx.view(1, -1, 1, 1) >= pos4.unsqueeze(-3)).byte()
    x_pos_mask = (base_idx.view(1, -1, 1, 1) < pos5.unsqueeze(-3)).byte()

    sum_of_mask = z_neg_mask + z_pos_mask + y_neg_mask + y_pos_mask + x_neg_mask + x_pos_mask

    voxelgrids = (voxelgrids * votes - sum_of_mask) > 0
    return voxelgrids
