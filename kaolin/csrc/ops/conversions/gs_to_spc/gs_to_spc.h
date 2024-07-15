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

#ifndef KAOLIN_OPS_CONVERSIONS_GS_TO_SPC_MESH_TO_SPC_H_
#define KAOLIN_OPS_CONVERSIONS_GS_TO_SPC_MESH_TO_SPC_H_

#include <ATen/ATen.h>

namespace kaolin::gs_to_spc {

std::vector<at::Tensor>
gs_to_spc_cuda(
	const at::Tensor& means3D,
	const at::Tensor& scales,
	const at::Tensor& rotations,
	const at::Tensor& opacities,
	const float iso,
    const float tol,
    const uint32_t level);


}  // namespace kaolin

#endif  // KAOLIN_OPS_CONVERSIONS_GS_TO_SPC_MESH_TO_SPC_H_
