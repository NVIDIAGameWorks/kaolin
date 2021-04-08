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

#ifndef KAOLIN_METRICS_UNBATCHED_TRIANGLE_DISTANCE_H_
#define KAOLIN_METRICS_UNBATCHED_TRIANGLE_DISTANCE_H_

#include <ATen/ATen.h>

namespace kaolin {

void unbatched_triangle_distance_forward_cuda(
    const at::Tensor points,
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3,
    const at::Tensor dist1,
    const at::Tensor idx1,
    const at::Tensor type1);

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
    const at::Tensor grad_input_v3);

}  // namespace kaolin

#endif  // KAOLIN_METRICS_UNBATCHED_TRIANGLE_DISTANCE_H_
