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

import math
from torch import nn
from torch.autograd import Function
import torch

from kaolin import _C
from kaolin.rep import Spc

__all__ = [
    'conv3d',
    'Conv3d',
    'conv_transpose3d',
    'ConvTranspose3d',
]

class Conv3dFunction(Function):
    @staticmethod
    def forward(ctx, octrees, point_hierarchies, level, pyramids, exsum,
                inputs, params, kernel_vectors, jump):
        octrees = octrees.contiguous()
        point_hierarchies = point_hierarchies.contiguous()
        pyramids = pyramids.contiguous()
        exsum = exsum.contiguous()
        inputs = inputs.contiguous()
        params = params.contiguous()
        kernel_vectors = kernel_vectors.contiguous()

        ctx.save_for_backward(octrees, point_hierarchies, pyramids, exsum,
                              inputs, params, kernel_vectors)
        ctx.jump = jump  # jump is an int, not a tensor

        outputs, level = _C.ops.spc.Conv3d_forward(
            octrees, point_hierarchies, level, pyramids, exsum,
            inputs, params, kernel_vectors, jump)
        ctx.level = level

        level = torch.tensor([level])
        ctx.mark_non_differentiable(level)
        return outputs, level

    @staticmethod
    def backward(ctx, grad_outputs, grad_level):
        grad_outputs = grad_outputs.contiguous()

        octrees, point_hierarchies, pyramids, exsum, inputs, params, kernel_vectors = ctx.saved_tensors

        d_inputs, d_params = _C.ops.spc.Conv3d_backward(
            octrees, point_hierarchies, ctx.level, pyramids, exsum, inputs,
            grad_outputs, params, kernel_vectors, ctx.jump)

        return None, None, None, None, None, d_inputs, d_params, None, None

def conv3d(octrees, point_hierarchies, level, pyramids, exsum, input,
           weight, kernel_vectors, jump=0, bias=None, **kwargs):
    r"""Convolution over a structured point cloud. The inputs :math:`X` are mapped
    to outputs :math:`Y` by the following:

    .. math::

        Y_i = \sum_k w_k \cdot X_{n(i,k)} + b \quad\text{for}\; i \in 0,\ldots,|Y|-1,

    where :math:`w_k` are weights associated with the kernel, and :math:`n(i,k)` is the
    neighborhood function described :ref:`here <neighborhood-text>`.

    Args:
        octrees (torch.ByteTensor):
            :ref:`packed` octrees of shape :math:`(\text{num_bytes})`.
            See :ref:`octree <spc_octree>`.
        point_hierarchies (torch.ShortTensor):
            :ref:`packed` point hierarchies of shape :math:`(\text{num_points})`.
            See :ref:`point_hierarchies <spc_points>`.
        level (int):
            level at which the ``input`` features are associated to.
        pyramids (torch.IntTensor):
            Batched tensor containing point hierarchy structural information
            of shape :math:`(\text{batch_size}, 2, \text{max_level}+2)`.
            See :ref:`pyramids <spc_pyramids>`.
        exsum (torch.IntTensor):
            Tensor containing the :ref:`packed` exclusive sum of the bit
            counts of individual octrees of shape :math:`(\text{num_bytes} + \text{batch_size})`.
            See :ref:`exsum <spc_exsum>`.
        input (torch.FloatTensor):
            :ref:`packed` input feature data of the octrees,
            of shape :math:`(\text{total_num_inputs}, \text{in_channels})`,
            where ``total_num_inputs`` correspond to the number of nodes of the octrees at ``level``,
            and ``in_channels`` is the input feature dimension (for instance 3 for RGB color).
        weight (torch.FloatTensor):
            filter of shape :math:`(\text{kernel_vectors.shape[0]}, \text{in_channels},
            \text{self.out_channels})`.
        kernel_vectors (torch.ShortTensor):
            A tensor of 3D offsets that define the shape of the kernel,
            of shape :math:`(\text{num_weights}, 3)`.
            See :ref:`kernel creation <kernel-text>`.
        jump (int, optional):
            The difference between the input and output levels for the convolution.
            A non-zero value implies downsampling. Value must be positive and refer to a valid level of
            the structured point cloud. Default: 0.
        bias (torch.FloatTensor, optional):
            optional bias tensor of shape :math:`(\text{out_channel})`.

    Returns:
        (torch.FloatTensor, int):

            - Output of convolution. Number of outputs will correspond
              to level in the hierachy determined by **jump**.

            - the level associated to the output features.
    """
    remaining_kwargs = kwargs.keys() - Spc.KEYS
    if len(remaining_kwargs) > 0:
        raise TypeError("conv3d got an unexpected keyword argument "
                        f"{list(remaining_kwargs)[0]}")

    if (weight.shape[0] == 1 and jump == 0):
        outputs = input.mm(weight.squeeze(0))
    else:
        outputs, level = Conv3dFunction.apply(octrees, point_hierarchies, level,
                                              pyramids, exsum, input, weight,
                                              kernel_vectors, jump)
    if bias is not None:
        outputs += bias.unsqueeze(0)

    return outputs, int(level)

class Conv3d(nn.Module):
    r"""Convolution layer for a structured point cloud. The inputs :math:`X` are mapped
    to outputs :math:`Y` by the following:

    .. math::

        Y_i = \sum_k w_k \cdot X_{n(i,k)} + b \quad\text{for}\; i \in 0,\ldots,|Y|-1,

    where :math:`w_k` are weights associated with the kernel, and :math:`n(i,k)` is the
    neighborhood function described :ref:`here <neighborhood-text>`.

    Args:
        in_channels (int):
            The number of channels in the input tensor.
        out_channels (int):
            The number of channels in the output tensor.
        kernel_vectors (torch.ShortTensor):
            A tensor of 3D offsets that define the shape of the kernel,
            of shape :math:`(\text{num_weights}, 3)`.
            See :ref:`kernel creation <kernel-text>`.
        jump (int, optional):
            The difference between the input and output levels for the convolution.
            A non-zero value implies downsampling. Value must be positive and refer to a valid level of
            the structured point cloud. Default: 0.
        bias (bool, optional):
            If True, the convolution layer has a bias. Default: True.
    """
    def __init__(self, in_channels, out_channels, kernel_vectors, jump=0, bias=True):
        super(Conv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_vectors_size = kernel_vectors.shape[0]
        self.jump = jump

        self.register_buffer('kernel_vectors', kernel_vectors)


        self.kernel_shape = (self.kernel_vectors_size,
                             self.in_channels, self.out_channels)

        self.weight = nn.Parameter(torch.empty(*self.kernel_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = (self.in_channels) * self.kernel_vectors_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, octrees, point_hierarchies, level, pyramids, exsum, input, **kwargs):
        r"""
        Args:
            octrees (torch.ByteTensor):
                :ref:`packed` octrees of shape :math:`(\text{num_bytes})`.
                See :ref:`octree <spc_octree>`.

            point_hierarchies (torch.ShortTensor):
                :ref:`packed` point hierarchies of shape :math:`(\text{num_points})`.
                See :ref:`point_hierarchies <spc_points>`.

            level (int):
                level at which the ``input`` features are associated to.

            pyramids (torch.IntTensor):
                Batched tensor containing point hierarchy structural information
                of shape :math:`(\text{batch_size}, 2, \text{max_level}+2)`.
                See :ref:`pyramids <spc_pyramids>`.

            exsum (torch.IntTensor):
                Tensor containing the :ref:`packed` exclusive sum of the bit
                counts of individual octrees of shape :math:`(\text{num_bytes} + \text{batch_size})`.
                See :ref:`exsum <spc_exsum>`.

            input (torch.FloatTensor):
                :ref:`packed` input feature data of the octrees,
                of shape :math:`(\text{total_num_inputs}, \text{in_channels})`,
                where ``total_num_inputs`` correspond to the number of nodes of the octrees at ``level``,
                and ``in_channels`` is the input feature dimension (for instance 3 for RGB color).

        Returns:
            (torch.FloatTensor, int):

                - Output of convolution. Number of outputs will correspond
                  to level in the hierachy determined by **jump**.

                - the level associated to the output features.
        """
        remaining_kwargs = kwargs.keys() - Spc.KEYS
        if len(remaining_kwargs) > 0:
            raise TypeError("Conv3d got an unexpected keyword argument "
                            f"{list(remaining_kwargs)[0]}")

        return conv3d(octrees, point_hierarchies, level, pyramids, exsum, input,
                      self.weight, self.kernel_vectors, self.jump, self.bias)

    def __repr__(self):
        s = '(in={}, out={}, kernel_vector_size={})'.format(
            self.in_channels, self.out_channels,
            self.kernel_vectors_size)
        return self.__class__.__name__ + s

class ConvTranspose3dFunction(Function):
    @staticmethod
    def forward(ctx, octrees, point_hierarchies, level, pyramids, exsum,
                inputs, params, kernel_vectors, jump):
        octrees = octrees.contiguous()
        point_hierarchies = point_hierarchies.contiguous()
        pyramids = pyramids.contiguous()
        exsum = exsum.contiguous()
        inputs = inputs.contiguous()
        params = params.contiguous()
        kernel_vectors = kernel_vectors.contiguous()

        ctx.save_for_backward(octrees, point_hierarchies, pyramids, exsum, inputs,
                              params, kernel_vectors)
        ctx.jump = jump

        outputs, level = _C.ops.spc.ConvTranspose3d_forward(octrees, point_hierarchies,
                                                            level, pyramids, exsum,
                                                            inputs, params, kernel_vectors, jump)
        ctx.level = level

        level = torch.tensor([level])
        ctx.mark_non_differentiable(level)
        return outputs, level

    @staticmethod
    def backward(ctx, grad_outputs, grad_level):
        grad_outputs = grad_outputs.contiguous()

        octrees, point_hierarchies, pyramids, exsum, inputs, params, kernel_vectors = \
            ctx.saved_tensors

        d_inputs, d_params = _C.ops.spc.ConvTranspose3d_backward(
            octrees, point_hierarchies, ctx.level, pyramids, exsum, inputs,
            grad_outputs, params, kernel_vectors, ctx.jump)

        return None, None, None, None, None, d_inputs, d_params, None, None

def conv_transpose3d(octrees, point_hierarchies, level, pyramids, exsum,
                     input, weight, kernel_vectors, jump=0, bias=None, **kwargs):
    r"""Transposed convolution over a structured point cloud. The inputs :math:`X` are mapped
    to outputs :math:`Y` by the following:

    .. math::

        Y_i = \sum_k w_k \cdot X_{n^T(i,k)} + b \quad\text{for}\; i \in 0,\ldots,|Y|-1,

    where :math:`w_k` are weights associated with the kernel, and :math:`n^T(i,k)` is the
    transpose neighborhood function described :ref:`here <neighborhood-text>`.


    Args:
        octrees (torch.ByteTensor):
            :ref:`packed` octrees of shape :math:`(\text{num_bytes})`.
            See :ref:`octree <spc_octree>`.

        point_hierarchies (torch.ShortTensor):
            :ref:`packed` point hierarchies of shape :math:`(\text{num_points})`.
            See :ref:`point_hierarchies <spc_points>`.

        level (int):
            level at which the ``input`` features are associated to.

        pyramids (torch.IntTensor):
            Batched tensor containing point hierarchy structural information
            of shape :math:`(\text{batch_size}, 2, \text{max_level}+2)`.
            See :ref:`pyramids <spc_pyramids>`.

        exsum (torch.IntTensor):
            Tensor containing the :ref:`packed` exclusive sum of the bit
            counts of individual octrees of shape :math:`(\text{num_bytes} + \text{batch_size})`.
            See :ref:`exsum <spc_exsum>`.

        input (torch.FloatTensor):
            :ref:`packed` input feature data of the octrees,
            of shape :math:`(\text{total_num_inputs}, \text{in_channels})`,
            where ``total_num_inputs`` correspond to the number of nodes of the octrees at ``level``,
            and ``in_channels`` is the input feature dimension (for instance 3 for RGB color).

        weight (torch.FloatTensor):
            filter of shape :math:`(\text{kernel_vectors.shape[0]}, \text{in_channels},
            \text{self.out_channels})`.

        kernel_vectors (torch.ShortTensor):
            A tensor of 3D offsets that define the shape of the kernel,
            of shape :math:`(\text{num_weights}, 3)`.
            See :ref:`kernel creation <kernel-text>`.

        jump (int, optional):
            The difference between the input and output levels for the convolution.
            A non-zero value implies downsampling. Value must be positive and refer to a valid level of
            the structured point cloud. Default: 0.

        bias (torch.FloatTensor, optional):
            optional bias tensor of shape :math:`(\text{out_channel})`.
    """
    remaining_kwargs = kwargs.keys() - Spc.KEYS
    if len(remaining_kwargs) > 0:
        raise TypeError("conv_transpose3d got an unexpected keyword argument "
                        f"{list(remaining_kwargs)[0]}")

    if (weight.shape[0] == 1 and jump == 0):
        outputs = input.mm(weight.squeeze(0))
    else:
        outputs, level = ConvTranspose3dFunction.apply(octrees, point_hierarchies, level, pyramids,
                                                       exsum, input, weight, kernel_vectors, jump)
    if bias is not None:
        outputs += bias.unsqueeze(0)

    return outputs, int(level)

class ConvTranspose3d(nn.Module):
    r"""Transposed convolution layer for a structured point cloud. The inputs :math:`X` are mapped
    to outputs :math:`Y` by the following:

    .. math::

        Y_i = \sum_k w_k \cdot X_{n^T(i,k)} + b \quad\text{for}\; i \in 0,\ldots,|Y|-1,

    where :math:`w_k` are weights associated with the kernel, and :math:`n^T(i,k)` is the
    transpose neighborhood function described :ref:`here <neighborhood-text>`.

    Args:
        in_channels (int):
            The number of channels in the input tensor.

        out_channels (int):
            The number of channels in the output tensor.

        kernel_vectors (torch.ShortTensor):
            A tensor of 3D offsets that define the shape of the kernel,
            of shape :math:`(\text{num_weights}, 3)`.
            See :ref:`kernel creation <kernel-text>`.

        jump (int, optional):
            The difference between the input and output levels for the convolution. Default: 0.
            A non-zero value implies upsampling. Value must be positive and refer to a valid level of
            the structured point cloud.

        bias (bool, optional):
            If True, the convolution layer has a bias. Default: True.
    """
    def __init__(self, in_channels, out_channels, kernel_vectors, jump=0, bias=True):
        super(ConvTranspose3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_vectors_size = kernel_vectors.shape[0]
        self.jump = jump

        self.register_buffer('kernel_vectors', kernel_vectors)

        self.kernel_shape = (self.kernel_vectors_size,
                             self.in_channels, self.out_channels)

        self.weight = nn.Parameter(torch.empty(*self.kernel_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = (self.out_channels) * self.kernel_vectors_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, octrees, point_hierarchies, level, pyramids, exsum, input, **kwargs):
        r"""
        Args:
            octrees (torch.ByteTensor):
                :ref:`packed` octrees of shape :math:`(\text{num_bytes})`.
                See :ref:`octree <spc_octree>`.

            point_hierarchies (torch.ShortTensor):
                :ref:`packed` point hierarchies of shape :math:`(\text{num_points})`.
                See :ref:`point_hierarchies <spc_points>`.

            level (int):
                level at which the ``input`` features are associated to.

            pyramids (torch.IntTensor):
                Batched tensor containing point hierarchy structural information
                of shape :math:`(\text{batch_size}, 2, \text{max_level}+2)`.
                See :ref:`pyramids <spc_pyramids>`.

            exsum (torch.IntTensor):
                Tensor containing the :ref:`packed` exclusive sum of the bit
                counts of individual octrees of shape :math:`(\text{num_bytes} + \text{batch_size})`.
                See :ref:`exsum <spc_exsum>`.

            input (torch.FloatTensor):
                :ref:`packed` input feature data of the octrees,
                of shape :math:`(\text{total_num_inputs}, \text{in_channels})`,
                where ``total_num_inputs`` correspond to the number of nodes of the octrees at ``level``,
                and ``in_channels`` is the input feature dimension (for instance 3 for RGB color).

        Returns:
            (torch.FloatTensor, int):

                - Output of transpose convolution. Number of outputs will correspond
                  to level in the hierachy determined by **jump**.

                - the level associated to the output features.
        """
        remaining_kwargs = kwargs.keys() - Spc.KEYS
        if len(remaining_kwargs) > 0:
            raise TypeError("ConvTranspose3d got an unexpected keyword argument "
                            f"{list(remaining_kwargs)[0]}")
        return conv_transpose3d(octrees, point_hierarchies, level, pyramids, exsum, input,
                                self.weight, self.kernel_vectors, self.jump, self.bias)

    def __repr__(self):
        s = '(in={}, out={}, kernel_vector_size={})'.format(
            self.in_channels, self.out_channels,
            self.kernel_vectors_size)
        return self.__class__.__name__ + s
