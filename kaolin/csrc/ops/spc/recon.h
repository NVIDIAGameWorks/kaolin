// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <ATen/ATen.h>


namespace kaolin {

at::Tensor inclusive_sum(at::Tensor Inputs);
at::Tensor build_mip2d(at::Tensor imagedata, at::Tensor InInv, int mip_levels, float maxdepth, bool true_depth);

//at::Tensor compactify(at::Tensor Points, at::Tensor Exsum);
//at::Tensor subdivide(at::Tensor Points, at::Tensor Exsum);

std::vector<at::Tensor> subdivide2(at::Tensor Points, at::Tensor Insum);
std::vector<at::Tensor> compactify2(at::Tensor Points, at::Tensor Exsum);

//at::Tensor scalar_to_rgb(at::Tensor scalars);

//at::Tensor slice_image(
//    at::Tensor  octree,
//    at::Tensor  points,
//    uint32_t    level,
//    at::Tensor  pyramid,
//    at::Tensor  prefixsum,
//    uint32_t    axes,
//    uint32_t    val);
//
//at::Tensor slice_image_empty(
//    at::Tensor  octree,
//    at::Tensor  empty,
//    at::Tensor  points,
//    uint32_t    level,
//    at::Tensor  pyramid,
//    at::Tensor  prefixsum,
//    uint32_t    axes,
//    uint32_t    val);

}  // namespace kaolin