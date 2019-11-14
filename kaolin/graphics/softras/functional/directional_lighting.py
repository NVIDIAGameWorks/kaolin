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
# Soft Rasterizer (SoftRas)
# 
# Copyright (c) 2017 Hiroharu Kato
# Copyright (c) 2018 Nikos Kolotouros
# Copyright (c) 2019 Shichen Liu
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def directional_lighting(light, normals, light_intensity=0.5, light_color=(1,1,1), 
                         light_direction=(0,1,0)):
    # normals: [nb, :, 3]

    device = light.device

    if isinstance(light_color, tuple) or isinstance(light_color, list):
        light_color = torch.tensor(light_color, dtype=torch.float32, device=device)
    elif isinstance(light_color, np.ndarray):
        light_color = torch.from_numpy(light_color).float().to(device)
    if isinstance(light_direction, tuple) or isinstance(light_direction, list):
        light_direction = torch.tensor(light_direction, dtype=torch.float32, device=device)
    elif isinstance(light_direction, np.ndarray):
        light_direction = torch.from_numpy(light_direction).float().to(device)
    if light_color.ndimension() == 1:
        light_color = light_color[None, :]
    if light_direction.ndimension() == 1:
        light_direction = light_direction[None, :] #[nb, 3]

    cosine = F.relu(torch.sum(normals * light_direction, dim=2)) #[]
    light += light_intensity * (light_color[:, None, :] * cosine[:, :, None])
    return light #[nb, :, 3]