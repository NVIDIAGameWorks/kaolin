/*
* Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES
* All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <ATen/ATen.h>

at::Tensor spc_point2morton(at::Tensor points);

at::Tensor spc_morton2point(at::Tensor morton_codes);

at::Tensor spc_point2coeff(
    at::Tensor x,
    at::Tensor pts);

at::Tensor spc_point2jacobian(at::Tensor x);

at::Tensor spc_point2corners(at::Tensor points);

