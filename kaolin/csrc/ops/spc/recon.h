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

std::vector<at::Tensor> subdivide(at::Tensor Points, at::Tensor Insum);
std::vector<at::Tensor> compactify(at::Tensor Points, at::Tensor Exsum);

}  // namespace kaolin