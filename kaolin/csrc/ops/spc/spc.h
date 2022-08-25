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

#ifndef KAOLIN_OPS_SPC_SPC_H_
#define KAOLIN_OPS_SPC_SPC_H_

#include <ATen/ATen.h>

#ifdef WITH_CUDA
#include "../../spc_math.h"
#endif

namespace kaolin {

at::Tensor morton_to_octree(
    at::Tensor mortons,
    uint32_t level);

at::Tensor points_to_octree(
    at::Tensor points,
    uint32_t level);

std::tuple<int, at::Tensor, at::Tensor> scan_octrees_cuda(
    at::Tensor octrees,
    at::Tensor lengths);

std::tuple<int, at::Tensor, at::Tensor> scan_octree_cuda(
    at::Tensor octree);

at::Tensor generate_points_cuda(
  at::Tensor octrees,
  at::Tensor pyramids,
  at::Tensor exsum);

std::tuple<at::Tensor, int> Conv3d_forward(
    at::Tensor octree,
    at::Tensor points,
    uint32_t level,
    at::Tensor pyramid,
    at::Tensor exsum,
    at::Tensor inputs,
    at::Tensor params,
    at::Tensor kernel_vectors,
    uint32_t jump);

std::vector<at::Tensor> Conv3d_backward(
    at::Tensor octree,
    at::Tensor points,
    uint32_t level,
    at::Tensor pyramid,
    at::Tensor exsum,
    at::Tensor inputs,
    at::Tensor grad_outputs,
    at::Tensor params,
    at::Tensor kernel_vectors,
    uint32_t jump);

std::tuple<at::Tensor, int> ConvTranspose3d_forward(
    at::Tensor octree,
    at::Tensor points,
    uint32_t level,
    at::Tensor pyramid,
    at::Tensor exsum,
    at::Tensor inputs,
    at::Tensor params,
    at::Tensor kernel_vectors,
    uint32_t jump);

std::vector<at::Tensor> ConvTranspose3d_backward(
    at::Tensor octree,
    at::Tensor points,
    uint32_t level,
    at::Tensor pyramid,
    at::Tensor exsum,
    at::Tensor inputs,
    at::Tensor grad_outputs,
    at::Tensor params,
    at::Tensor kernel_vectors,
    uint32_t jump);

}  // namespace kaolin

#endif  // KAOLIN_OPS_SPC_SPC_H_

