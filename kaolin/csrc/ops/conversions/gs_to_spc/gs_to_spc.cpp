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


#include <ATen/ATen.h>

#include "../../../check.h"

namespace kaolin::gs_to_spc {

#ifdef WITH_CUDA

std::vector<at::Tensor> gs_to_spc_cuda_impl(
	const at::Tensor& means3D,
	const at::Tensor& scales,
	const at::Tensor& rotations,
	const at::Tensor& opacities,
	const float iso,
  const float tol,
  const uint target_level);

#endif

std::vector<at::Tensor>
gs_to_spc_cuda(
	const at::Tensor& means3D,
	const at::Tensor& scales,
	const at::Tensor& rotations,
	const at::Tensor& opacities,
	const float iso,
  const float tol,
  const uint32_t level) {
#ifdef WITH_CUDA

  CHECK_DIMS(means3D, 2);
  CHECK_DIMS(scales, 2);
  CHECK_DIMS(rotations, 2);
  CHECK_DIMS(opacities, 1);
  CHECK_FLOAT(means3D);
  CHECK_FLOAT(scales);
  CHECK_FLOAT(rotations);
  CHECK_FLOAT(opacities);

  return gs_to_spc_cuda_impl(means3D, scales, rotations, opacities, iso, tol, level);

#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif

}

}  // namespace kaolin
