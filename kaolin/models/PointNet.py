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

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetFeatureExtractor(nn.Module):
    r"""PointNet feature extractor (extracts either global or local, i.e.,
    per-point features).

    Based on the original PointNet paper:

        .. code-block::

        @article{qi2016pointnet,
          title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
          author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
          journal={arXiv preprint arXiv:1612.00593},
          year={2016}
        }

    Args:
        in_channels (int): Number of channels in the input pointcloud
            (default: 3, for X, Y, Z coordinates respectively).
        feat_size (int): Size of the global feature vector
            (default: 1024)
        layer_dims (Iterable[int]): Sizes of fully connected layers
            to be used in the feature extractor (excluding the input and
            the output layer sizes). Note: the number of
            layers in the feature extractor is implicitly parsed from
            this variable.
        global_feat (bool): Extract global features (i.e., one feature
            for the entire pointcloud) if set to True. If set to False,
            extract per-point (local) features (default: True).
        activation (function): Nonlinearity to be used as activation
                    function after each batchnorm (default: F.relu)
        batchnorm (bool): Whether or not to use batchnorm layers
            (default: True)
        transposed_input (bool): Whether the input's second and third dimension
            is already transposed. If so, a transpose operation can be avoided,
            improving performance.
            See documentation for the forward method for more details.

    For example, to specify a PointNet feature extractor with 4 linear
    layers (sizes 6 -> 10, 10 -> 40, 40 -> 500, 500 -> 1024), with
    3 input channels in the pointcloud and a global feature vector of size
    1024, see the example below.

    Example:

        >>> pointnet = PointNetFeatureExtractor(in_channels=3, feat_size=1024,
                                           layer_dims=[10, 20, 40, 500])
        >>> x = torch.rand(2, 3, 30)
        >>> y = pointnet(x)
        print(y.shape)

    """

    def __init__(self,
                 in_channels: int = 3,
                 feat_size: int = 1024,
                 layer_dims: Iterable[int] = [64, 128],
                 global_feat: bool = True,
                 activation=F.relu,
                 batchnorm: bool = True,
                 transposed_input: bool = False):
        super(PointNetFeatureExtractor, self).__init__()

        if not isinstance(in_channels, int):
            raise TypeError('Argument in_channels expected to be of type int. '
                            'Got {0} instead.'.format(type(in_channels)))
        if not isinstance(feat_size, int):
            raise TypeError('Argument feat_size expected to be of type int. '
                            'Got {0} instead.'.format(type(feat_size)))
        if not hasattr(layer_dims, '__iter__'):
            raise TypeError('Argument layer_dims is not iterable.')
        for idx, layer_dim in enumerate(layer_dims):
            if not isinstance(layer_dim, int):
                raise TypeError('Elements of layer_dims must be of type int. '
                                'Found type {0} at index {1}.'.format(
                                    type(layer_dim), idx))
        if not isinstance(global_feat, bool):
            raise TypeError('Argument global_feat expected to be of type '
                            'bool. Got {0} instead.'.format(
                                type(global_feat)))

        # Store feat_size as a class attribute
        self.feat_size = feat_size

        # Store activation as a class attribute
        self.activation = activation

        # Store global_feat as a class attribute
        self.global_feat = global_feat

        # Add in_channels to the head of layer_dims (the first layer
        # has number of channels equal to `in_channels`). Also, add
        # feat_size to the tail of layer_dims.
        if not isinstance(layer_dims, list):
            layer_dims = list(layer_dims)
        layer_dims.insert(0, in_channels)
        layer_dims.append(feat_size)

        self.conv_layers = nn.ModuleList()
        if batchnorm:
            self.bn_layers = nn.ModuleList()
        for idx in range(len(layer_dims) - 1):
            self.conv_layers.append(nn.Conv1d(layer_dims[idx],
                                              layer_dims[idx + 1], 1))
            if batchnorm:
                self.bn_layers.append(nn.BatchNorm1d(layer_dims[idx + 1]))

        # Store whether or not to use batchnorm as a class attribute
        self.batchnorm = batchnorm

        self.transposed_input = transposed_input

    def forward(self, x: torch.Tensor):
        r"""Forward pass through the PointNet feature extractor.

        Args:
            x (torch.Tensor): Tensor representing a pointcloud
                (shape: :math:`B \times N \times D`, where :math:`B`
                is the batchsize, :math:`N` is the number of points
                in the pointcloud, and :math:`D` is the dimensionality
                of each point in the pointcloud).
                If self.transposed_input is True, then the shape is
                :math:`B \times D \times N`.

        """
        if not self.transposed_input:
            x = x.transpose(1, 2)

        # Number of points
        num_points = x.shape[2]

        # By default, initialize local features (per-point features)
        # to None.
        local_features = None

        # Apply a sequence of conv-batchnorm-nonlinearity operations

        # For the first layer, store the features, as these will be
        # used to compute local features (if specified).
        if self.batchnorm:
            x = self.activation(self.bn_layers[0](self.conv_layers[0](x)))
        else:
            x = self.activation(self.conv_layers[0](x))
        if self.global_feat is False:
            local_features = x

        # Pass through the remaining layers (until the penultimate layer).
        for idx in range(1, len(self.conv_layers) - 1):
            if self.batchnorm:
                x = self.activation(self.bn_layers[idx](
                    self.conv_layers[idx](x)))
            else:
                x = self.activation(self.conv_layers[idx](x))

        # For the last layer, do not apply nonlinearity.
        if self.batchnorm:
            x = self.bn_layers[-1](self.conv_layers[-1](x))
        else:
            x = self.conv_layers[-1](x)

        # Max pooling.
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.feat_size)

        # If extracting global features, return at this point.
        if self.global_feat:
            return x

        # If extracting local features, compute local features by
        # concatenating global features, and per-point features
        x = x.view(-1, self.feat_size, 1).repeat(1, 1, num_points)
        return torch.cat((x, local_features), dim=1)


class PointNetClassifier(nn.Module):
    r"""PointNet classifier. Uses the PointNet feature extractor, and
    adds classification layers on top.

    Based on the original PointNet paper:

        @article{qi2016pointnet,
          title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
          author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
          journal={arXiv preprint arXiv:1612.00593},
          year={2016}
        }

    Args:
        in_channels (int): Number of channels in the input pointcloud
            (default: 3, for X, Y, Z coordinates respectively).
        feat_size (int): Size of the global feature vector
            (default: 1024)
        num_classes (int): Number of classes (for the classification
            task) (default: 2).
        dropout (float): Dropout ratio to use (default: 0.). Note: If
            the ratio is set to 0., we altogether skip using a dropout
            layer.
        layer_dims (Iterable[int]): Sizes of fully connected layers
            to be used in the feature extractor (excluding the input and
            the output layer sizes). Note: the number of
            layers in the feature extractor is implicitly parsed from
            this variable.
        activation (function): Nonlinearity to be used as activation
                    function after each batchnorm (default: F.relu)
        batchnorm (bool): Whether or not to use batchnorm layers
            (default: True)
        transposed_input (bool): Whether the input's second and third dimension
            is already transposed. If so, a transpose operation can be avoided,
            improving performance.
            See documentation of PointNetFeatureExtractor for more details.

    Example:

        pointnet = PointNetClassifier(in_channels=6, feat_size=1024,
                                      feat_layer_dims=[32, 64, 256],
                                      classifier_layer_dims=[500, 200, 100])
        x = torch.rand(5, 6, 30)
        y = pointnet(x)
        print(y.shape)

    """

    def __init__(self,
                 in_channels: int = 3,
                 feat_size: int = 1024,
                 num_classes: int = 2,
                 dropout: float = 0.,
                 classifier_layer_dims: Iterable[int] = [512, 256],
                 feat_layer_dims: Iterable[int] = [64, 128],
                 activation=F.relu,
                 batchnorm: bool = True,
                 transposed_input: bool = False):

        super(PointNetClassifier, self).__init__()

        if not isinstance(num_classes, int):
            raise TypeError('Argument num_classes must be of type int. '
                            'Got {0} instead.'.format(type(num_classes)))
        if not isinstance(dropout, float):
            raise TypeError('Argument dropout must be of type float. '
                            'Got {0} instead.'.format(type(dropout)))
        if dropout < 0 or dropout > 1:
            raise ValueError('Dropout ratio must always be in the range'
                             '[0, 1]. Got {0} instead.'.format(dropout))
        if not hasattr(classifier_layer_dims, '__iter__'):
            raise TypeError('Argument classifier_layer_dims is not iterable.')
        for idx, layer_dim in enumerate(classifier_layer_dims):
            if not isinstance(layer_dim, int):
                raise TypeError('Expected classifier_layer_dims to contain '
                                'int. Found type {0} at index {1}.'.format(
                                    type(layer_dim), idx))

        # Add feat_size to the head of classifier_layer_dims (the output of
        # the PointNet feature extractor has number of elements equal to
        # has number of channels equal to `in_channels`).
        if not isinstance(classifier_layer_dims, list):
            classifier_layer_dims = list(classifier_layer_dims)
        classifier_layer_dims.insert(0, feat_size)

        # Note that `global_feat` MUST be set to True, for global
        # classification tasks.
        self.feature_extractor = PointNetFeatureExtractor(
            in_channels=in_channels, feat_size=feat_size,
            layer_dims=feat_layer_dims, global_feat=True,
            activation=activation, batchnorm=batchnorm,
            transposed_input=transposed_input
        )

        self.linear_layers = nn.ModuleList()
        if batchnorm:
            self.bn_layers = nn.ModuleList()
        for idx in range(len(classifier_layer_dims) - 1):
            self.linear_layers.append(nn.Linear(classifier_layer_dims[idx],
                                                classifier_layer_dims[idx + 1]))
            if batchnorm:
                self.bn_layers.append(nn.BatchNorm1d(
                    classifier_layer_dims[idx + 1]))

        self.last_linear_layer = nn.Linear(classifier_layer_dims[-1],
                                           num_classes)

        # Store activation as a class attribute
        self.activation = activation

        # Dropout layer (if dropout ratio is in the interval (0, 1]).
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

        else:
            self.dropout = None

        # Store whether or not to use batchnorm as a class attribute
        self.batchnorm = batchnorm

        self.transposed_input = transposed_input

    def forward(self, x):
        r"""Forward pass through the PointNet classifier.

        Args:
            x (torch.Tensor): Tensor representing a pointcloud
                (shape: :math:`B \times N \times D`, where :math:`B`
                is the batchsize, :math:`N` is the number of points
                in the pointcloud, and :math:`D` is the dimensionality
                of each point in the pointcloud).
                If self.transposed_input is True, then the shape is
                :math:`B \times D \times N`.

        """
        x = self.feature_extractor(x)
        for idx in range(len(self.linear_layers) - 1):
            if self.batchnorm:
                x = self.activation(self.bn_layers[idx](
                    self.linear_layers[idx](x)))
            else:
                x = self.activation(self.linear_layers[idx](x))
        # For penultimate linear layer, apply dropout before batchnorm
        if self.dropout:
            if self.batchnorm:
                x = self.activation(self.bn_layers[-1](self.dropout(
                    self.linear_layers[-1](x))))
            else:
                x = self.activation(self.dropout(self.linear_layers[-1](x)))
        else:
            if self.batchnorm:
                x = self.activation(self.bn_layers[-1](
                    self.linear_layers[-1](x)))
            else:
                x = self.activation(self.linear_layers[-1](x))
        # TODO: Use dropout before batchnorm of penultimate linear layer
        x = self.last_linear_layer(x)
        # return F.log_softmax(x, dim=1)
        return x


class PointNetSegmenter(nn.Module):
    r"""PointNet segmenter. Uses the PointNet feature extractor, and
    adds per-point segmentation layers on top.

    Based on the original PointNet paper:

        @article{qi2016pointnet,
          title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
          author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
          journal={arXiv preprint arXiv:1612.00593},
          year={2016}
        }

    Args:
        in_channels (int): Number of channels in the input pointcloud
            (default: 3, for X, Y, Z coordinates respectively).
        feat_size (int): Size of the global feature vector
            (default: 1024)
        num_classes (int): Number of classes (for the segmentation
            task) (default: 2).
        dropout (float): Dropout ratio to use (default: 0.). Note: If
            the ratio is set to 0., we altogether skip using a dropout
            layer.
        layer_dims (Iterable[int]): Sizes of fully connected layers
            to be used in the feature extractor (excluding the input and
            the output layer sizes). Note: the number of
            layers in the feature extractor is implicitly parsed from
            this variable.
        activation (function): Nonlinearity to be used as activation
                    function after each batchnorm (default: F.relu)
        batchnorm (bool): Whether or not to use batchnorm layers
            (default: True)
        transposed_input (bool): Whether the input's second and third dimension
            is already transposed. If so, a transpose operation can be avoided,
            improving performance.
            See documentation of PointNetFeatureExtractor for more details.

    Example:

        pointnet = PointNetSegmenter(in_channels=6, feat_size=1024,
                                         feat_layer_dims=[32, 64, 256],
                                         classifier_layer_dims=[500, 200, 100])
        x = torch.rand(5, 6, 30)
        y = pointnet(x)
        print(y.shape)

    """

    def __init__(self,
                 in_channels: int = 3,
                 feat_size: int = 1024,
                 num_classes: int = 2,
                 dropout: float = 0.,
                 classifier_layer_dims: Iterable[int] = [512, 256],
                 feat_layer_dims: Iterable[int] = [64, 128],
                 activation=F.relu,
                 batchnorm: bool = True,
                 transposed_input: bool = False):
        super(PointNetSegmenter, self).__init__()

        if not isinstance(num_classes, int):
            raise TypeError('Argument num_classes must be of type int. '
                            'Got {0} instead.'.format(type(num_classes)))
        if not isinstance(dropout, float):
            raise TypeError('Argument dropout must be of type float. '
                            'Got {0} instead.'.format(type(dropout)))
        if not hasattr(classifier_layer_dims, '__iter__'):
            raise TypeError('Argument classifier_layer_dims is not iterable.')
        for idx, layer_dim in enumerate(classifier_layer_dims):
            if not isinstance(layer_dim, int):
                raise TypeError('Expected classifier_layer_dims to contain '
                                'int. Found type {0} at index {1}.'.format(
                                    type(layer_dim), idx))

        # Add feat_size to the head of classifier_layer_dims (the output of
        # the PointNet feature extractor has number of elements equal to
        # has number of channels equal to `in_channels`).
        if not isinstance(classifier_layer_dims, list):
            classifier_layer_dims = list(classifier_layer_dims)
        classifier_layer_dims.insert(0, feat_size)

        # Note that `global_feat` MUST be set to False, for
        # segmentation tasks.
        self.feature_extractor = PointNetFeatureExtractor(
            in_channels=in_channels, feat_size=feat_size,
            layer_dims=feat_layer_dims, global_feat=False,
            activation=activation, batchnorm=batchnorm,
            transposed_input=transposed_input
        )

        # Compute the dimensionality of local features
        # Local feature size = (global feature size) + (feature size
        #       from the output of the first layer of feature extractor)
        # Note: In self.feature_extractor, we manually append in_channels
        # to the head of feat_layer_dims. So, we use index 1 of
        # feat_layer_dims in the below line, to compute local_feat_size,
        # and not index 0.
        self.local_feat_size = feat_size + feat_layer_dims[1]

        self.conv_layers = nn.ModuleList()
        if batchnorm:
            self.bn_layers = nn.ModuleList()
        # First classifier layer
        self.conv_layers.append(nn.Conv1d(self.local_feat_size,
                                          classifier_layer_dims[0], 1))
        if batchnorm:
            self.bn_layers.append(nn.BatchNorm1d(classifier_layer_dims[0]))
        for idx in range(len(classifier_layer_dims) - 1):
            self.conv_layers.append(nn.Conv1d(classifier_layer_dims[idx],
                                              classifier_layer_dims[idx + 1], 1))
            if batchnorm:
                self.bn_layers.append(nn.BatchNorm1d(
                    classifier_layer_dims[idx + 1]))

        self.last_conv_layer = nn.Conv1d(classifier_layer_dims[-1],
                                         num_classes, 1)

        # Store activation as a class attribute
        self.activation = activation

        # Store the number of classes as an attribute
        self.num_classes = num_classes

        # Store whether or not to use batchnorm as a class attribute
        self.batchnorm = batchnorm

        self.transposed_input = transposed_input

    def forward(self, x):
        r"""Forward pass through the PointNet segmentation model.

        Args:
            x (torch.Tensor): Tensor representing a pointcloud
                shape: :math:`B \times N \times D`, where :math:`B`
                is the batchsize, :math:`N` is the number of points
                in the pointcloud, and :math:`D` is the dimensionality
                of each point in the pointcloud.
                If self.transposed_input is True, then the shape is
                :math:`B \times D \times N`.

        """
        batchsize = x.shape[0]
        num_points = x.shape[2] if self.transposed_input else x.shape[1]
        x = self.feature_extractor(x)
        for idx in range(len(self.conv_layers)):
            if self.batchnorm:
                x = self.activation(self.bn_layers[idx](
                    self.conv_layers[idx](x)))
            else:
                x = self.activation(self.conv_layers[idx](x))
        x = self.last_conv_layer(x)
        x = x.transpose(2, 1).contiguous()
        # x = F.log_softmax(x.view(-1, self.num_classes), dim=-1)
        print("x.shape = {}".format(x.shape))
        return x.view(batchsize, num_points, self.num_classes)
