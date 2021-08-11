/*
* Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <vector>
#include <torch/torch.h>

at::Tensor spc_point2morton(torch::Tensor points);

at::Tensor spc_morton2point(torch::Tensor morton_codes);

at::Tensor spc_point2coeff(
    torch::Tensor x,
    torch::Tensor pts);

at::Tensor spc_point2jacobian(torch::Tensor x);

at::Tensor spc_point2corners(torch::Tensor points);

