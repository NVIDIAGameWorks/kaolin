// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef KAOLIN_OPS_CONVERSIONS_GS_TO_SPC_GS_TO_SPC_H_
#define KAOLIN_OPS_CONVERSIONS_GS_TO_SPC_GS_TO_SPC_H_

#include <ATen/ATen.h>

namespace kaolin {

std::vector<at::Tensor>
gs_to_spc_cuda(
  const at::Tensor& means3D,
  const at::Tensor& scales,
  const at::Tensor& rotations,
  const float iso,
  const float tol,
  const uint32_t level);

at::Tensor integrate_gs_cuda(
  const at::Tensor& points,
  const at::Tensor& gaus_id,
  const at::Tensor& means3D,
  const at::Tensor& cov3Dinv,
  const at::Tensor& opacities,
  const uint32_t    level,
  const uint32_t    step);

}  // namespace kaolin

#endif  // KAOLIN_OPS_CONVERSIONS_GS_TO_SPC_GS_TO_SPC_H_
