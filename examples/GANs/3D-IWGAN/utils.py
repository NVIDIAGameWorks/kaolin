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

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def calculate_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss"""
    # Random weight for interpolation between real and fake samples
    eta = torch.rand((real_samples.size(0), 1, 1, 1), device=real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (eta * real_samples + ((1 - eta) * fake_samples)).requires_grad_(True)
    # calculate probability of interpolated examples
    d_interpolates = D(interpolates)
    # Get gradient w.r.t. interpolates
    fake = torch.ones_like(d_interpolates, device=real_samples.device, requires_grad=False)
    gradients = torch.autograd.grad(outputs=d_interpolates,
                                    inputs=interpolates,
                                    grad_outputs=fake,
                                    create_graph=True,
                                    retain_graph=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
