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

#ifndef KAOLIN_OPS_RENDER_SPC_RAYTRACE_H_
#define KAOLIN_OPS_RENDER_SPC_RAYTRACE_H_

#ifdef WITH_CUDA
#include "../../spc_math.h"
#endif

#include <ATen/ATen.h>

namespace kaolin {

std::vector<at::Tensor> generate_primary_rays_cuda(
    uint height,
    uint width,
    at::Tensor Eye,
    at::Tensor At,
    at::Tensor Up,
    float fov,
    at::Tensor World);

at::Tensor raytrace_cuda(
    at::Tensor octree,
    at::Tensor points,
    at::Tensor pyramid,
    at::Tensor exclusive_sum,
    at::Tensor ray_o,
    at::Tensor ray_d,
    uint target_level) ;


at::Tensor remove_duplicate_rays_cuda(
    at::Tensor nuggets);

at::Tensor mark_first_hit_cuda(
    at::Tensor nuggets);

std::vector<torch::Tensor> spc_ray_aabb(
    torch::Tensor nuggets,
    torch::Tensor points,
    torch::Tensor ray_query,
    torch::Tensor ray_d,
    uint targetLevel,
    torch::Tensor info,
    torch::Tensor info_idxes,
    torch::Tensor cond,
    bool init);

std::vector<at::Tensor> generate_shadow_rays_cuda(
    at::Tensor ray_o,
    at::Tensor ray_d,
    at::Tensor light,
    at::Tensor plane);

}  // namespace kaolin

#endif  // KAOLIN_OPS_RENDER_SPC_RAYTRACE_H_
