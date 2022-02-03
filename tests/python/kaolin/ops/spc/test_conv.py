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
import pytest
import os
from itertools import product

import torch
from kaolin.ops.spc.uint8 import bits_to_uint8, uint8_bits_sum, uint8_to_bits
from kaolin.ops.random import random_spc_octrees
from kaolin.rep import Spc

from kaolin.ops import spc

from kaolin.utils.testing import FLOAT_TYPES, with_seed, check_tensor

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('height,width,depth,threshold',
                         [(27, 37, 37, 0.7), (64, 64, 64, 0.)])
@pytest.mark.parametrize('in_channels', [1, 5])
@pytest.mark.parametrize('out_channels', [1, 7])
@pytest.mark.parametrize('kernel_size,kernel_offset', [(1, 0), (2, 0), (3, 0), (3, 1), (4, 0), (5, 0), (5, 2)])
@pytest.mark.parametrize('with_bias', [False, True])
class TestConv3D:
    @pytest.fixture(autouse=True)
    def sparsity_masks(self, batch_size, height, width, depth, threshold):
        return torch.rand(batch_size, height, width, depth,
                          device='cuda') > threshold

    @pytest.fixture(autouse=True)
    def feature_grids(self, sparsity_masks, batch_size, in_channels, height, width, depth):
        return torch.rand(batch_size, in_channels, height, width, depth,
                          device='cuda') * sparsity_masks.unsqueeze(1)

    @pytest.fixture(autouse=True)
    def kernel_vectors(self, kernel_size, kernel_offset):
        return torch.tensor(
            list(product(range(-kernel_offset, kernel_size - kernel_offset), repeat=3)),
            dtype=torch.int16, device='cuda')

    @pytest.fixture(autouse=True)
    def dense_weight(self, in_channels, out_channels, kernel_size):
        return torch.rand(out_channels, in_channels,
                          kernel_size, kernel_size, kernel_size,
                          device='cuda')

    @pytest.fixture(autouse=True)
    def spc_weight(self, dense_weight, in_channels, out_channels):
        return dense_weight.reshape(out_channels, in_channels, -1).permute(2, 1, 0)

    @pytest.fixture(autouse=True)
    def bias(self, with_bias, out_channels):
        if with_bias:
            return torch.rand(out_channels, device='cuda')
        else:
            return None

    @pytest.fixture(autouse=True)
    def octrees_lengths_features(self, feature_grids, sparsity_masks):
        return spc.feature_grids_to_spc(feature_grids, sparsity_masks)

    @pytest.fixture(autouse=True)
    def octrees(self, octrees_lengths_features):
        return octrees_lengths_features[0]

    @pytest.fixture(autouse=True)
    def lengths(self, octrees_lengths_features):
        return octrees_lengths_features[1]

    @pytest.fixture(autouse=True)
    def coalescent_features(self, octrees_lengths_features):
        return octrees_lengths_features[2]

    @pytest.fixture(autouse=True)
    def max_level_pyramids_exsum(self, octrees, lengths):
        return spc.scan_octrees(octrees, lengths)

    @pytest.fixture(autouse=True)
    def max_level(self, max_level_pyramids_exsum):
        return max_level_pyramids_exsum[0]

    @pytest.fixture(autouse=True)
    def pyramids(self, max_level_pyramids_exsum):
        return max_level_pyramids_exsum[1]

    @pytest.fixture(autouse=True)
    def exsum(self, max_level_pyramids_exsum):
        return max_level_pyramids_exsum[2]

    @pytest.fixture(autouse=True)
    def point_hierarchies(self, octrees, pyramids, exsum):
        return spc.generate_points(octrees, pyramids, exsum)

    @pytest.mark.parametrize('with_spc_to_dict', [False, True])
    @pytest.mark.parametrize('jump', [0, 1, 2])
    def test_conv3d(self, height, width, depth, in_channels, out_channels, kernel_size,
                    feature_grids, sparsity_masks, dense_weight, bias,
                    octrees, lengths, coalescent_features, max_level,
                    pyramids, exsum, point_hierarchies,
                    kernel_vectors, kernel_offset, spc_weight, jump, with_spc_to_dict):
        stride = 2 ** jump
        coalescent_features = coalescent_features.detach()
        coalescent_features.requires_grad = True
        spc_weight = spc_weight.detach()
        spc_weight.requires_grad = True

        if with_spc_to_dict:
            input_spc = Spc(octrees, lengths)
            output_features, output_level = spc.conv3d(
                **input_spc.to_dict(), level=input_spc.max_level, input=coalescent_features,
                weight=spc_weight, kernel_vectors=kernel_vectors, jump=jump, bias=bias)
            output = spc.to_dense(**input_spc.to_dict(), input=output_features,
                                  level=output_level)
            output_sparsity_masks = spc.to_dense(
                **input_spc.to_dict(),
                input=torch.ones_like(output_features, requires_grad=False),
                level=output_level)
        else:
            output_features, output_level = spc.conv3d(
                octrees, point_hierarchies, max_level, pyramids, exsum, coalescent_features,
                spc_weight, kernel_vectors, jump=jump, bias=bias)
            output = spc.to_dense(point_hierarchies, pyramids, output_features, output_level)
            output_sparsity_masks = spc.to_dense(
                point_hierarchies, pyramids, torch.ones_like(output_features, requires_grad=False),
                output_level)

        feature_grids = feature_grids.detach()
        feature_grids.requires_grad = True
        dense_weight = dense_weight.detach()
        dense_weight.requires_grad = True

        padded_input = torch.nn.functional.pad(feature_grids,
                                               (kernel_offset, kernel_size - 1 - kernel_offset,
                                                kernel_offset, kernel_size - 1 - kernel_offset,
                                                kernel_offset, kernel_size - 1 - kernel_offset))
        expected_output = torch.nn.functional.conv3d(padded_input, dense_weight, stride=stride, bias=bias)
        expected_height, expected_width, expected_depth = expected_output.shape[2:]
        expected_output *= output_sparsity_masks[:, :, :expected_height, :expected_width, :expected_depth]
        assert torch.allclose(output[:, :, :expected_height, :expected_width, :expected_depth],
                              expected_output, atol=1e-3, rtol=1e-3)
        grad_output = torch.rand_like(output)
        output.backward(grad_output)
        expected_output.backward(grad_output[:, :, :expected_height, :expected_width, :expected_depth])

        _, _, sparsified_grad = spc.feature_grids_to_spc(feature_grids.grad, sparsity_masks)

        assert torch.allclose(coalescent_features.grad, sparsified_grad, rtol=1e-3, atol=1e-3)
        assert torch.allclose(spc_weight.grad,
                              dense_weight.grad.reshape(out_channels, in_channels, -1).permute(2, 1, 0),
                              rtol=5e-2, atol=5e-2)

    @pytest.mark.parametrize('with_spc_to_dict', [False, True])
    @pytest.mark.parametrize('jump', [0, 1, 2])
    def test_conv_transpose3d(self, height, width, depth, in_channels, out_channels,
                              sparsity_masks, dense_weight, bias,
                              octrees, lengths, max_level, pyramids, exsum, point_hierarchies,
                              kernel_vectors, kernel_size, kernel_offset, spc_weight, jump,
                              with_spc_to_dict):
        stride = 2 ** jump

        if stride > kernel_size:
            pytest.skip('stride higher than kernel_size is not tested')

        out_sparsity_masks = sparsity_masks
        in_level = max_level - jump
        in_num_nodes = torch.sum(pyramids[:, 0, -(2 + jump)])
        coalescent_features = torch.rand((in_num_nodes, in_channels), device='cuda',
                                         requires_grad=True)

        dense_weight = dense_weight.detach()
        dense_weight.requires_grad = True
        spc_weight  = spc_weight.detach()
        spc_weight.requires_grad = True
        if with_spc_to_dict:
            input_spc = Spc(octrees, lengths)
            feature_grids = spc.to_dense(**input_spc.to_dict(), input=coalescent_features,
                                         level=in_level)
        else:
            feature_grids = spc.to_dense(point_hierarchies, pyramids, coalescent_features, in_level)
        feature_grids = feature_grids[:, :, :math.ceil(height / stride),
                                      :math.ceil(width / stride), :math.ceil(depth / stride)]
        feature_grids = feature_grids.detach()
        feature_grids.requires_grad = True
        if with_spc_to_dict:
            sparsity_masks = spc.to_dense(
                **input_spc.to_dict(), input=torch.ones_like(coalescent_features),
                level=in_level).bool()
        else:
            sparsity_masks = spc.to_dense(point_hierarchies, pyramids,
                                          torch.ones_like(coalescent_features),
                                          in_level).bool()
        sparsity_masks = sparsity_masks[:, 0, :math.ceil(height / stride),
                                        :math.ceil(width / stride), :math.ceil(depth / stride)]

        # test forward
        if with_spc_to_dict:
            output_features, output_level = spc.conv_transpose3d(
                **input_spc.to_dict(), level=in_level, input=coalescent_features,
                weight=spc_weight, kernel_vectors=kernel_vectors, jump=jump, bias=bias)
            output = spc.to_dense(**input_spc.to_dict(), input=output_features, level=output_level)
        else:
            output_features, output_level = spc.conv_transpose3d(
                octrees, point_hierarchies, in_level, pyramids, exsum,
                coalescent_features,
                spc_weight, kernel_vectors, jump=jump, bias=bias)
            output = spc.to_dense(point_hierarchies, pyramids, output_features, output_level)

        output = output[:, :, :height, :width, :depth]

        expected_output = torch.nn.functional.conv_transpose3d(
                feature_grids, dense_weight.permute(1, 0, 2, 3, 4),
                stride=stride, bias=bias,
                output_padding=stride - 1)[:, :,
                                           kernel_offset:height + kernel_offset,
                                           kernel_offset:width + kernel_offset,
                                           kernel_offset:depth + kernel_offset]
        expected_output *= out_sparsity_masks.unsqueeze(1)
        assert output_level == max_level
        assert torch.allclose(output, expected_output, rtol=1e-3, atol=1e-3)
        # test backward
        grad_out = torch.rand_like(expected_output)
        expected_output.backward(grad_out)
        output.backward(grad_out)
        _, _, sparsified_grad = spc.feature_grids_to_spc(feature_grids.grad, sparsity_masks)
        assert torch.allclose(coalescent_features.grad, sparsified_grad,
                              rtol=5e-2, atol=5e-2)
        assert torch.allclose(spc_weight.grad,
                              dense_weight.grad.reshape(out_channels, in_channels, -1).permute(2, 1, 0),
                              rtol=5e-2, atol=5e-2)

    @pytest.mark.parametrize('with_spc_to_dict', [False, True])
    @pytest.mark.parametrize('jump', [0, 1, 2])
    def test_module_conv3d(self, height, width, depth, in_channels, out_channels, with_bias,
                           octrees, lengths, coalescent_features, max_level, pyramids, exsum,
                           point_hierarchies, kernel_vectors, jump, with_spc_to_dict):
        conv = spc.Conv3d(in_channels, out_channels, kernel_vectors,
                          jump, bias=with_bias).cuda()
        params = dict(conv.named_parameters())
        weight = params['weight']
        check_tensor(weight, shape=(kernel_vectors.shape[0],
                                    in_channels, out_channels),
                     dtype=torch.float, device='cuda')
        if with_bias:
            assert len(params) == 2
            bias = params['bias']
            check_tensor(bias, shape=(out_channels,), dtype=torch.float,
                         device='cuda')
        else:
            assert len(params) == 1
            bias = None

        buffers = dict(conv.named_buffers())
        assert len(buffers) == 1
        assert torch.equal(buffers['kernel_vectors'], kernel_vectors)

        assert repr(conv) == f'Conv3d(in={in_channels}, out={out_channels}, ' \
                             f'kernel_vector_size={kernel_vectors.shape[0]})'

        if with_spc_to_dict:
            input_spc = Spc(octrees, lengths)
            output, output_level = conv(**input_spc.to_dict(), level=max_level,
                                        input=coalescent_features)
        else:
            output, output_level = conv(
                octrees, point_hierarchies, max_level, pyramids, exsum,
                coalescent_features)

        expected_output, expected_output_level = spc.conv3d(
            octrees, point_hierarchies, max_level, pyramids, exsum, coalescent_features,
            weight, kernel_vectors, jump=jump, bias=bias)
        assert torch.equal(output, expected_output)
        assert output_level == expected_output_level

    @pytest.mark.parametrize('with_spc_to_dict', [False, True])
    @pytest.mark.parametrize('jump', [0, 1, 2])
    def test_module_conv_transpose3d(self, height, width, depth, in_channels, out_channels, with_bias,
                                     octrees, lengths, max_level, pyramids, exsum, point_hierarchies,
                                     kernel_size, kernel_vectors, jump, with_spc_to_dict):
        stride = 2 ** jump

        if stride > kernel_size:
            pytest.skip('stride higher than kernel_size is not tested')

        in_level = max_level - jump
        in_num_nodes = torch.sum(pyramids[:, 0, -(2 + jump)])
        coalescent_features = torch.rand((in_num_nodes, in_channels), device='cuda',
                                         requires_grad=True)


        conv = spc.ConvTranspose3d(in_channels, out_channels, kernel_vectors,
                                   jump, bias=with_bias).cuda()
        params = dict(conv.named_parameters())
        weight = params['weight']
        check_tensor(weight, shape=(kernel_vectors.shape[0],
                                    in_channels, out_channels),
                     dtype=torch.float, device='cuda')
        if with_bias:
            assert len(params) == 2
            bias = params['bias']
            check_tensor(bias, shape=(out_channels,), dtype=torch.float,
                         device='cuda')
        else:
            assert len(params) == 1
            bias = None

        buffers = dict(conv.named_buffers())
        assert len(buffers) == 1
        assert torch.equal(buffers['kernel_vectors'], kernel_vectors)

        assert repr(conv) == f'ConvTranspose3d(in={in_channels}, ' \
                             f'out={out_channels}, ' \
                             f'kernel_vector_size={kernel_vectors.shape[0]})'

        if with_spc_to_dict:
            input_spc = Spc(octrees, lengths)
            output, output_level = conv(**input_spc.to_dict(), level=in_level,
                                        input=coalescent_features)
        else:
            output, output_level = conv(
                octrees, point_hierarchies, in_level, pyramids, exsum,
                coalescent_features)

        expected_output, expected_output_level = spc.conv_transpose3d(
            octrees, point_hierarchies, in_level, pyramids, exsum, coalescent_features,
            weight, kernel_vectors, jump=jump, bias=bias)
        assert torch.equal(output, expected_output)
        assert output_level == expected_output_level
