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

class _PackedSimpleSumCuda(torch.autograd.Function):
    """torch.autograd.function wrapper for :func:`tile_to_packed` CUDA implementations"""

    @staticmethod
    def forward(ctx, inputs, numel_per_tensor):
        inputs = inputs.contiguous()
        numel_per_tensor = numel_per_tensor.contiguous()
        output = _C.ops.packed_simple_sum_cuda(inputs, numel_per_tensor)
        if inputs.dtype == torch.half:
            output = output.to(torch.half)
        ctx.save_for_backward(numel_per_tensor)
        ctx.inputs_shape = inputs.shape
        ctx.inputs_dtype = inputs.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        numel_per_tensor, = ctx.saved_tensors
        grad_inputs = torch.empty(ctx.inputs_shape, dtype=ctx.inputs_dtype, device=grad_output.device)
        _C.ops.tile_to_packed_out_cuda(grad_output, numel_per_tensor, grad_inputs)
        return grad_inputs, None

def packed_simple_sum(tensor, numel_per_tensor):
    """Sum of each subtensor in a packed tensor with last_dim=1.

    Args:
        tensor (torch.Tensor): The input :ref:`packed_tensor<packed>`
        numel_per_tensor (torch.LongTensor):
            Tensor containing the number of element per sub-tensor.

    Returns:
        (torch.Tensor):
            A 1D tensor of size ``tensor.shape[0]``,
            containing the sum of each sub-tensor in the input tensor.
    """
    assert tensor.shape[-1] == 1
    if torch.cuda.is_available() and tensor.is_cuda and not numel_per_tensor.is_cuda:
        output = _PackedSimpleSumCuda.apply(tensor, numel_per_tensor)
    else:
        output = []
        last_id = 0
        for i, numel in enumerate(numel_per_tensor):
            first_id = last_id
            last_id += int(numel)
            output.append(torch.sum(tensor[first_id:last_id]))
        output = torch.stack(output, dim=0)
    return output
