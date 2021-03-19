// Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KAOLIN_OPS_TILE_TO_PACKED_H_
#define KAOLIN_OPS_TILE_TO_PACKED_H_

#include <ATen/ATen.h>

namespace kaolin {

at::Tensor tile_to_packed_cuda(
    at::Tensor values_tensor,
    at::Tensor shape_per_tensor,
    int total_numel);

at::Tensor tile_to_packed_out_cuda(
    at::Tensor values_tensor,
    at::Tensor shape_per_tensor,
    at::Tensor output);

}  // namespace kaolin

#endif  // KAOLIN_OPS_TILE_TO_PACKED_H_
