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

#ifndef KAOLIN_OPS_CONVERSIONS_MESH_TO_SPC_MESH_TO_SPC_H_
#define KAOLIN_OPS_CONVERSIONS_MESH_TO_SPC_MESH_TO_SPC_H_

#include <ATen/ATen.h>

namespace kaolin {

std::vector<at::Tensor> mesh_to_spc_cuda(
    at::Tensor face_vertices,
    uint32_t Level) ;


}  // namespace kaolin

#endif  // KAOLIN_OPS_CONVERSIONS_MESH_TO_SPC_H_
