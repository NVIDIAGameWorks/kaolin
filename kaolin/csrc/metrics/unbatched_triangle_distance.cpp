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

#ifdef WITH_CUDA
void unbatched_triangle_distance_forward_cuda_kernel_launcher(
    const at::Tensor points,
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3,
    const at::Tensor dist1,
    const at::Tensor idx1,
    const at::Tensor type1);

void unbatched_triangle_distance_backward_cuda_kernel_launcher(
    const at::Tensor grad_output,
    const at::Tensor points,
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3,
    const at::Tensor idx,
    const at::Tensor dist_type,
    const at::Tensor grad_input_p,
    const at::Tensor grad_input_v1,
    const at::Tensor grad_input_v2,
    const at::Tensor grad_input_v3);
#endif  // WITH_CUDA

void unbatched_triangle_distance_forward_cuda(
    const at::Tensor points,
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3,
    const at::Tensor dist1,
    const at::Tensor idx1,
    const at::Tensor type1) {
#if WITH_CUDA
  unbatched_triangle_distance_forward_cuda_kernel_launcher(
      points, verts_1, verts_2, verts_3, dist1, idx1, type1
  );
#else
  AT_ERROR("unbatched_triangle_distance_forward not built with CUDA");
#endif
}

void unbatched_triangle_distance_backward_cuda(
    const at::Tensor grad_output,
    const at::Tensor points,
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3,
    const at::Tensor idx,
    const at::Tensor dist_type,
    const at::Tensor grad_input_p,
    const at::Tensor grad_input_v1,
    const at::Tensor grad_input_v2,
    const at::Tensor grad_input_v3) {
#if WITH_CUDA
  unbatched_triangle_distance_backward_cuda_kernel_launcher(
      grad_output, points, verts_1, verts_2, verts_3, idx, dist_type,
      grad_input_p, grad_input_v1, grad_input_v2, grad_input_v3
  );
#else
  AT_ERROR("unbatched_triangle_distance_backward not built with CUDA");
#endif
}

}  // namespace kaolin
