/*
* Copyright (c) 2021,22 NVIDIA CORPORATION & AFFILIATES
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

namespace kaolin {

at::Tensor points_to_morton_cuda(at::Tensor points);

at::Tensor morton_to_points_cuda(at::Tensor morton_codes);

at::Tensor interpolate_trilinear_cuda(
    at::Tensor coords,
    at::Tensor pidx,
    at::Tensor points,
    at::Tensor trinkets,
    at::Tensor feats,
    int32_t level);

at::Tensor coords_to_trilinear_cuda(
    at::Tensor coords,
    at::Tensor points);

at::Tensor coords_to_trilinear_jacobian_cuda(at::Tensor coords);

at::Tensor points_to_corners_cuda(at::Tensor points);

} // namespace kaolin

