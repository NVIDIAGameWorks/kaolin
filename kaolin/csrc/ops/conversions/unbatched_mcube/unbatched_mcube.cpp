// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

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

namespace kaolin {

#if WITH_CUDA
std::vector<at::Tensor> unbatched_mcube_forward_cuda_kernel_launcher(const at::Tensor voxelgrid, float iso_value);
#endif

std::vector<at::Tensor> unbatched_mcube_forward_cuda(const at::Tensor voxelgrid, float iso_value) {
#if WITH_CUDA
  return unbatched_mcube_forward_cuda_kernel_launcher(voxelgrid, iso_value);
#else
  AT_ERROR("packed_simple_sum not built with CUDA");
#endif  // WITH_CUDA
}

}  // namespace kaolin
