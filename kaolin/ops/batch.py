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
from kaolin import _C

class _TileToPackedCuda(torch.autograd.Function):
    """torch.autograd.function wrapper for :func:`tile_to_packed` CUDA implementations"""
    @staticmethod
    def forward(ctx, inputs, numel_per_tensor, total_numel):
        inputs = inputs.contiguous()
        numel_per_tensor = numel_per_tensor.contiguous()
        output = _C.ops.tile_to_packed_cuda(inputs, numel_per_tensor, total_numel)
        ctx.save_for_backward(numel_per_tensor)
        ctx.inputs_dtype = inputs.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        numel_per_tensor, = ctx.saved_tensors
        grad_inputs = _C.ops.packed_simple_sum_cuda(grad_output, numel_per_tensor)
        return grad_inputs.to(ctx.inputs_dtype), None, None

def get_shape_per_tensor(tensor_list):
    r"""Returns the shape of each tensor in the tensor list except the last dimension.

    See shape_per_tensor for :ref:`packed<packed_shape_per_tensor>` or :ref:`padded<padded_shape_per_tensor>`
    for more information.

    Args:
        tensor_list (sequence of torch.Tensor): any python sequence of tensors of the identical type,
            number of dimensions, and last dimension size, e.g. :math:`[(H_0, W_0, C), (H_1, W_1, C)]`.

    Returns:
        (torch.Tensor):
            the shape of each subtensor (except for the last dim),
            of shape :math:`(len(\text{tensor_list}), \text{tensor_list[0].ndim} - 1)`.

    Examples:
        >>> tensor_list = [
        ...         torch.zeros((1, 3, 4, 2)),
        ...         torch.ones((2, 5, 3, 2))
        ... ]
        >>> get_shape_per_tensor(tensor_list)
        tensor([[1, 3, 4],
                [2, 5, 3]])
    """
    try:
        shape_per_tensor = torch.tensor([t.shape[:-1] for t in tensor_list], dtype=torch.long)
    except ValueError as err:
        ndim = tensor_list[0].ndim
        for i, t in enumerate(tensor_list):
            if t.ndim != ndim:
                raise ValueError(f"Expected all tensors to have {ndim} dimensions "
                                 f"but got {t.ndim} at index {i}")
        raise err  # Unknown error
    return shape_per_tensor

def list_to_packed(tensor_list):
    r"""Converts a sequence of torch.Tensor into a single :ref:`packed tensor<packed>`.

    torch.Tensor of same type, number of dimensions and last dimension size
    will be reshaped to :math:`(-1, \text{last_dim})` and concatenated on first axis.
    E.g.:
    With input of shapes :math:`[(X_0, Y_0, Z_0, C), (X_1, Y_1, Z_1, C)]` the output packed tensor will be
    of shape :math:`((X_0 * Y_0 * Z_0 + X_1 * Y_1 * Z_1), C)`.
    The output shape_per_tensor will be the tensor: :math:`[[X_0, Y_0, Z_0], [X_1, Y_1, Z_1]]`.

    Args:
        tensor_list (sequence of torch.Tensor): any python sequence of tensors of identical type,
            number of dimensions, and last dimension size, e.g. :math:`[(H_0, W_0, C), (H_1, W_1, C)]`.

    Returns:
        (torch.Tensor, torch.LongTensor):
            the :ref:`packed tensor<packed>` and the associated :ref:`shape_per_tensor<padded_shape_per_tensor>`

    Example:
        >>> a = torch.LongTensor([[0, 1, 2],
        ...                       [1, 2, 3]])
        >>> b = torch.LongTensor([[2, 4, 5]])
        >>> packed_tensor, shape_per_tensor = list_to_packed([a, b])
        >>> packed_tensor
        tensor([[0, 1, 2],
                [1, 2, 3],
                [2, 4, 5]])
        >>> shape_per_tensor
        tensor([[2],
                [1]])
    """
    shape_per_tensor = get_shape_per_tensor(tensor_list)
    try:
        output = torch.cat([t.reshape(-1, t.shape[-1]) for t in tensor_list], dim=0)
    except RuntimeError as err:
        last_dim = tensor_list[0].shape[-1]
        t_type = tensor_list[0].type()
        for i, t in enumerate(tensor_list):
            if t.shape[-1] != last_dim:
                raise ValueError(f"Expected all tensor to have last dimension {last_dim} "
                                 f"but got {t.shape[-1]} at index {i}")
            if t.type() != t_type:
                raise ValueError(f"Expected all tensor to have type {t_type} "
                                 f"but got {t.type()} at index {i}")
        raise err  # Unknown error

    return output, shape_per_tensor

def get_first_idx(numel_per_tensor):
    """Returns the first indices of each tensor in the :ref:`packed tensor <packed>`.

    See :ref:`first_idx definition <packed_first_idx>` for more information.

    Args:
        numel_per_tensor (torch.LongTensor): The number of elements
            (vertices, faces, points...) in each unbatched tensor, as a 1D tensor.

    Returns:
        (torch.LongTensor):
            first indices for each unbatched tensor in the packed tensor,
            and the last index + 1, as 1D tensor.

    Example:
        >>> numel_per_tensor = torch.LongTensor([2, 3, 5])
        >>> get_first_idx(numel_per_tensor)
        tensor([ 0,  2,  5, 10])
    """
    output = torch.zeros((numel_per_tensor.shape[0] + 1,), dtype=torch.long,
                         device=numel_per_tensor.device)
    torch.cumsum(numel_per_tensor, dim=0, out=output[1:])
    return output

def tile_to_packed(values, numel_per_tensor):
    r"""Tiles values to a packed representation of numel_per_tensor,

    Args:
        values (torch.Tensor): tensor of shape :math:`(\text{batch_size},)` of values to be tiled.
        numel_per_tensor (torch.LongTensor): number of elements per tensor of the output packed tensor.

    Return:
        torch.Tensor:
            The :ref:`packed tensor<packed>` of tiled values of shape
            :math:`(sum(\text{numel_per_tensor}), 1)`.

    Example:
        >>> values = torch.tensor([0., 6., 7.])
        >>> numel_per_tensor = torch.LongTensor([2, 2, 3])
        >>> tile_to_packed(values, numel_per_tensor)
        tensor([[0.],
                [0.],
                [6.],
                [6.],
                [7.],
                [7.],
                [7.]])
    """
    if torch.cuda.is_available() and values.is_cuda and not numel_per_tensor.is_cuda:
        # TODO(cfujitsang): this could be externalized with lazy initialization
        #                   currently kept inside as the slowdown is still reasonable
        total_numel = torch.sum(numel_per_tensor)
        tiled_packed_tensor = _TileToPackedCuda.apply(values, numel_per_tensor, total_numel)
    else:
        tiled_packed_tensor = torch.cat(
            [torch.full((int(numel),), fill_value=value.item(), dtype=values.dtype, device=values.device)
             for value, numel in zip(values, numel_per_tensor)], dim=0).unsqueeze(-1)
    return tiled_packed_tensor

def packed_to_list(packed_tensor, shape_per_tensor, first_idx):
    """Converts a single packed tensor into a sequence of torch.Tensor.

    Args:
        packed_tensor (torch.Tensor): input packed tensor.
        shape_per_tensor (torch.LongTensor): :ref:`shape_per_tensor<packed_shape_per_tensor>` associated to the packed tensor.
        first_idx (torch.LongTensor): :ref:`first_idx<packed_first_idx>` associated to the packed tensor.

    Return:
        list of torch.Tensor: list of tensor unbatched from packed_tensor

    Example:
        >>> packed_tensor = torch.arange(16).reshape(8, 2)
        >>> packed_tensor
        tensor([[ 0,  1],
                [ 2,  3],
                [ 4,  5],
                [ 6,  7],
                [ 8,  9],
                [10, 11],
                [12, 13],
                [14, 15]])
        >>> shape_per_tensor = torch.LongTensor([[3], [4], [1]])
        >>> first_idx = torch.LongTensor([0, 3, 7, 8])
        >>> packed_to_list(packed_tensor, shape_per_tensor, first_idx)
        [tensor([[0, 1],
                [2, 3],
                [4, 5]]), tensor([[ 6,  7],
                [ 8,  9],
                [10, 11],
                [12, 13]]), tensor([[14, 15]])]
    """
    last_dim = packed_tensor.shape[-1]
    return [packed_tensor[first_id:last_id].reshape(*shape, last_dim)
            for first_id, last_id, shape in zip(first_idx[:-1], first_idx[1:], shape_per_tensor)]

def fill_max_shape(shape_per_tensor, partial_max_shape=None):
    r"""Fills partial definition of shape to be at least as big as each shape in shape_per_tensor.

    if the i-th dimension is -1 then the i-th output will be ``shape_per_tensor[:,i].max()``.

    Args:
        shape_per_tensor (torch.Tensor): Input :ref:`shape_per_tensor<packed_shape_per_tensor>`,
                                         of shape :math:`(\text{N}, \text{ndim})`.
        partial_max_shape (tuple, list or torch.Tensor): partially defined maximum shape,
                                                         of size ``ndim``.

    Returns:
        (torch.Tensor): the max_shape fully defined, of same size than ``partial_max_shape``.

    Example:
        >>> partial_max_shape = (6, -1, -1)
        >>> shape_per_tensor = torch.LongTensor([[2, 3, 5],
        ...                                      [3, 4, 2]])
        >>> fill_max_shape(shape_per_tensor, partial_max_shape)
        tensor([6, 4, 5])
    """
    list_max_shape, idx_max_shape = torch.max(shape_per_tensor, dim=0)

    if partial_max_shape is None:
        max_shape = list_max_shape
    else:
        # Avoid inplace modification of mutable argument
        if torch.is_tensor(partial_max_shape):
            max_shape = partial_max_shape.clone()
        else:
            max_shape = torch.LongTensor(partial_max_shape)
        for i, max_dim in enumerate(list_max_shape):
            if max_shape[i] == -1:
                max_shape[i] = max_dim
            elif max_shape[i] < max_dim:
                raise ValueError(f"dim {i} of max_shape ({max_shape[i]} is smaller than "
                                 f"for tensor {idx_max_shape[i]} ({max_dim})")
    return max_shape

def list_to_padded(tensor_list, padding_value, max_shape=None):
    r"""Converts a sequence of torch.Tensor into a single :ref:`padded tensor<padded>`.

    torch.Tensor of same type, number of dimensions and last dimension size
    will be padded and stacked on first axis.
    E.g.:
    With input of shapes :math:`[(X_0, Y_0, Z_0, C), (X_1, Y_1, Z_1, C)]`
    the output padded tensor will be of shape
    :math:`(2, max(X_0, X_1, \text{max_shape}[0]),
    max(Y_0, Y_1, \text{max_shape}[1]), max(Z_0, Z_1, \text{max_shape}[2]), C)`
    The output shape_per_tensor with be the tensor: :math:`[[X_0, Y_0, Z_0], [X_1, Y_1, Z_1]].`

    Args:
        tensor_list (sequence of torch.Tensor): any python sequence of tensors of identical type,
            number of dimensions, and last dimension size, e.g. :math:`[(H_0, W_0, C), (H_1, W_1, C)]`.
        padding_value (float): the value that will be used as padding.
        max_shape (list, tuple or torch.LongTensor): list of maximum value for each dim
            of the output shape (except batch and last axis), if a value is set to None
            then it will be the maximum value among the tensors.
            Default: All maximum values among the tensors.

    Return:
        (torch.Tensor, torch.LongTensor):
            the :ref:`padded tensor<padded>` and the associated :ref:`shape_per_tensor<padded_shape_per_tensor>`.

    Example:
        >>> a = torch.LongTensor([[0, 1, 2],
        ...                       [1, 2, 3]])
        >>> b = torch.LongTensor([[2, 4, 5]])
        >>> padded_tensor, shape_per_tensor = list_to_padded([a, b], -1, [3])
        >>> padded_tensor
        tensor([[[ 0,  1,  2],
                 [ 1,  2,  3],
                 [-1, -1, -1]],
        <BLANKLINE>
                [[ 2,  4,  5],
                 [-1, -1, -1],
                 [-1, -1, -1]]])
        >>> shape_per_tensor
        tensor([[2],
                [1]])
    """
    shape_per_tensor = get_shape_per_tensor(tensor_list)
    batch_size = shape_per_tensor.shape[0]
    last_dim = tensor_list[0].shape[-1]
    max_shape = fill_max_shape(shape_per_tensor, max_shape)
    output = torch.full((batch_size, *max_shape, last_dim), fill_value=padding_value,
                        device=tensor_list[0].device, dtype=tensor_list[0].dtype)
    for i, t in enumerate(tensor_list):
        output[[i] + [slice(elem_dim) for elem_dim in t.shape]] = t
    return output, shape_per_tensor

def padded_to_list(padded_tensor, shape_per_tensor):
    """Converts a single padded tensor into a sequence of torch.Tensor.

    Args:
        padded_tensor (torch.Tensor): a :ref:`padded tensor<padded>`.
        shape_per_tensor (torch.LongTensor): the :ref:`shape_per_tensor<padded_shape_per_tensor>`
            tensor associated to the padded tensor.

    Return:
        list of torch.Tensor: list of tensor unbatched from padded_tensor

    Example:
        >>> padded_tensor = torch.LongTensor([[[0, 1, 2],
        ...                                    [1, 2, 3],
        ...                                    [-1, -1, -1]],
        ...                                   [[2, 4, 5],
        ...                                    [-1, -1, -1],
        ...                                    [-1, -1, -1]]])
        >>> shape_per_tensor = torch.LongTensor([[2], [1]])
        >>> padded_to_list(padded_tensor, shape_per_tensor)
        [tensor([[0, 1, 2],
                [1, 2, 3]]), tensor([[2, 4, 5]])]
    """
    return [padded_tensor[[i] + [slice(dim) for dim in shape]]
            for i, shape in enumerate(shape_per_tensor)]

def packed_to_padded(packed_tensor, shape_per_tensor, first_idx, padding_value, max_shape=None):
    """Converts a single packed tensor into a padded tensor.

    Args:
        packed_tensor (torch.Tensor): a :ref:`packed tensor<packed>`.
        shape_per_tensor (torch.LongTensor): the :ref:`shape_per_tensor<packed_shape_per_tensor>`
            tensor associated to the padded tensor.
        first_idx (torch.LongTensor): :ref:`first_idx<packed_first_idx>` associated to the packed tensor.
        padding_value (float): the value that will be used as padding.
        max_shape (list, tuple or torch.LongTensor): list of maximum value for each dim
            of the output shape (except batch and last axis), if a value is set to None
            then it will be the maximum value among the tensors.
            Default: All maximum values among the tensors.

    Returns:
        (torch.Tensor): the :ref:`padded tensor<padded>`.

    """
    batch_size = shape_per_tensor.shape[0]
    last_dim = packed_tensor.shape[1]
    max_shape = fill_max_shape(shape_per_tensor, max_shape)
    output = torch.full((batch_size, *max_shape, last_dim), fill_value=padding_value,
                        device=packed_tensor.device, dtype=packed_tensor.dtype)
    for i, shape in enumerate(shape_per_tensor):
        output[[i] + [slice(elem_dim) for elem_dim in shape]] = \
            packed_tensor[first_idx[i]:first_idx[i + 1]].reshape(*shape, last_dim)
    return output

def padded_to_packed(padded_tensor, shape_per_tensor):
    """Converts a single padded tensor into a packed tensor.

    Args:
        padded_tensor (torch.Tensor): a :ref:`padded tensor<padded>`.
        shape_per_tensor (torch.LongTensor): the :ref:`shape_per_tensor<padded_shape_per_tensor>`
            tensor associated to the padded tensor.

    Returns:
        (torch.Tensor): the :ref:`packed tensor<packed>`.
    """
    return torch.cat([t.reshape(-1, padded_tensor.shape[-1])
                      for t in padded_to_list(padded_tensor, shape_per_tensor)], dim=0)
