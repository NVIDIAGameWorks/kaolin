// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef KAOLIN_OPS_SPC_FEATURE_GRIDS_H_
#define KAOLIN_OPS_SPC_FEATURE_GRIDS_H_

#include <ATen/ATen.h>

namespace kaolin {

at::Tensor to_dense_forward(
    at::Tensor points,
    int level,
    at::Tensor pyramid,
    at::Tensor features);

at::Tensor to_dense_backward(
    at::Tensor points,
    int level,
    at::Tensor pyramid,
    at::Tensor features,
    at::Tensor grad_outputs);

}  // namespace kaolin

#endif  // KAOLIN_OPS_SPC_FEATURE_GRIDS_H_

