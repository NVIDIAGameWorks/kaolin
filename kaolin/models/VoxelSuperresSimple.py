# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    """A simple encoder-decoder style voxel superresolution network"""

    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.deconv3 = nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(16)
        self.deconv4 = nn.ConvTranspose3d(16, 8, 3, stride=2, padding=0)
        self.deconv5 = nn.ConvTranspose3d(8, 1, 3, stride=2, padding=0)

    def forward(self, x):
        # Encoder
        x = (F.relu(self.bn1(self.conv1(x))))
        x = (F.relu(self.bn2(self.conv2(x))))
        # Decoder
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.deconv4(x))
        # Superres layer
        return self.deconv5(x)
